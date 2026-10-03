# MTP depth in the storage-offload regime

**Short version:** when a model's n-gram table has to live on SSD because it does
not fit in RAM, Multi-Token Prediction depth must stay *shallow* — and the
reason is disk I/O, not FLOPs. On an M2 Max 96 GB running a 125B MoE with a
51.2B-token n-gram PLE, cost per generated token is U-shaped in depth with its
minimum at **depth 1–2**, and it degrades badly past depth 4.

Two things this repo is equally careful about, because both went wrong here
first:

- **Acceptance rate is not a quality metric.** It counts how much compute was
  saved, not how many characters were wrong. Rejected drafts are replaced by
  the backbone's own token and never reach the output. What MTP does *not*
  guarantee is byte-identical text on free-form output.
- **A silent HTTP failure is worse than a loud one.** One `401` swallowed by an
  `except Exception: pass` made an A/B compare a configuration against itself
  and report a confident 6.9% effect. §4.8 is the postmortem, and the harness
  now refuses to produce a number when a toggle does not take.

This repo contains the experiments, the raw data, and the tooling.

---

## 1. Why this is not the usual story

The textbook advice for speculative decoding is "go deeper if acceptance rate
is high." That advice assumes verification is compute-bound, so verifying *K*
candidate tokens costs barely more than verifying one.

That assumption breaks when **the model's own weights are partly on SSD.**

Qwen3.8-Flash-Next is a 125B MoE with a 51.2B n-gram prompt-lookup-embedding
(PLE) table. Fully resident it needs **111.6 GB**. The machine has **103.1 GB**
usable, so the PLE *cannot* be resident — oMLX logs this explicitly:

```
Qwen4-Exp PLE forced to SSD for Qwen3.8-Flash-Next-oQ4e-mtp:
  resident 111.6GB leaves no room to serve prompts under the 92.3GB memory ceiling
```

PLE rows are mmap'd, so every verification step performs a **random disk read**
to fetch the embedding row for the current n-gram. Verification is therefore no
longer a batched bandwidth-bound operation: each additional verified position
adds its own miss. Cost stops being sub-linear in depth and becomes
approximately linear, while the benefit (tokens per cycle) saturates as soon as
acceptance is high.

Both effects push the same way, and the optimum lands at a small depth.

Measured cost per generated token, mean over 11 tasks:

| depth | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| ms/token | 20.67 | **20.33** | 21.02 | 22.50 | 26.03 | 34.64 |

The curve is U-shaped with a **very shallow** minimum: depth 1 and depth 2 land
within 1.6% of each other, which is *below* the ±4.5% throughput noise measured
over 198 cells. Depth 3 is already worse; depth 8 is 68% worse than depth 1.

So the defensible claim is not "depth 2 is optimal". It is: **depth 1 and depth
2 are a tie, and anything past 2 is a loss.** Which of the tied pair wins is
decided by the workload — see §7.

The cost is almost entirely backbone, i.e. the full forward pass that fetches
the PLE row, not the draft head. At depth 8: backbone 167.0 ms/cycle, MTP head
10.2 ms/cycle.

### What acceptance actually predicts

Sorting tasks by α and looking at the throughput change from depth 1 to depth 2:

| α at depth 1 | tasks | depth 2 vs depth 1 |
|---|---|---|
| > 90% | repetitive code, TS component, Rust impl | **+6.6% to +17.9%** |
| 78–90% | extraction, doc rewrite, Python, JSON | −7.5% to +12.4% (mixed) |
| < 74% | translation, factual QA, creative prose | **−3.9% to −18.4%** |

High acceptance is what makes depth 2 worth it. Below roughly 75% it is
actively harmful: creative prose (α = 53.6%) loses 18.4% of its throughput.

## 2. Setup

| | |
|---|---|
| Host | MacBook Pro M2 Max, 96 GB unified memory, macOS 26.5.2 (arm64), headless |
| Metal cap | `iogpu.wired_limit_mb=88000` → 85.9 GB (default was 77.8 GB) |
| Server | oMLX 0.7.0, `launchd` system daemon, port 8091 |
| Model | `Qwen3.8-Flash-Next-oQ4e-mtp`, 106.3 GB on disk, 69.5 GB resident |
| Memory guard | 82 GB |
| Context | 32768 |
| Decoding | `burst_decode=aggressive`, `max_concurrent_requests=1` |

Two knobs dominate everything below: raising the Metal wired limit bought the
7 GB that made the model loadable at all, and the SSD cache size was raised to
200 GB so the page cache never thrashes.

## 3. Changing depth without root

If the server runs as a `launchd` **system** daemon it is owned by root, so from
an unprivileged shell `lsof` cannot see its socket, `kill` cannot stop it, and
`launchctl bootout` fails with `Operation not permitted`. All three failures
look like unrelated bugs; they are one permission wall.

Use the admin API instead — no root required:

```bash
# 1. log in (the api_key is the same one the server uses for inference)
curl -c cj -X POST http://127.0.0.1:8091/admin/api/login \
  -H 'Content-Type: application/json' \
  -d '{"api_key":"'"$OMLX_API_KEY"'"}'

# 2. write the setting
curl -b cj -X PUT \
  http://127.0.0.1:8091/admin/api/models/$OMLX_MODEL/settings \
  -H 'Content-Type: application/json' \
  -d '{"mtp_fixed_depth":2}'
```

Two things will bite you:

1. **The path prefix is `/admin/api/`.** Omitting it returns `404`, which reads
   like "the route does not exist". It does exist — see `/openapi.json`, which
   lists 119 routes.
2. **It is not a hot swap.** Any settings write makes oMLX log
   `Settings changed for loaded model, auto-unloading` and reload the entire
   model (unload 69.5 GB → load). The HTTP call blocks for the whole reload
   (tens of seconds), so client timeouts must be in minutes. Poll
   `/health` for `engine_pool.loaded_count == 1` and then settle ~45 s more:
   the health check can succeed *before* the auto-unload even begins, so
   returning early is a race.

## 4. Methodology pitfalls

These cost more time than the experiments did.

### 4.1 α must be token-weighted, and needs a minimum sample size

Acceptance rate is usually averaged per log line, or worse, over "whatever came
back". Both break on short generations. oMLX emits a statistics line per
request:

```
MTP[7] finish=stop tokens=234 cycles=78 tok/cycle=3.00 accept=154/156 (98.7%)
  depth[d1=78/78,d2=76/78] emits[...] timing[backbone=..ms mtp=..ms sample=..ms cache=..ms]
```

A response that stops after 3–4 tokens gives `tokens=3 cycles=1` and
`accept=0/1 (0.0%)`. One sample, denominator 1. Averaged into a cell mean it
produced this:

| task | T=0.0 | T=0.3 | T=0.6 | T=0.9 | **T=1.2** | T=1.6 |
|---|---|---|---|---|---|---|
| Rust impl | 91.8 | 92.3 | 93.5 | 91.8 | **41.1** | 39.5 |

That reads as "high temperature destroys acceptance on Rust". It was a
collection artifact: the raw repeats were `[0.0, 82.3]`. Re-running T=1.2 gave
**90.6%**. The effect did not exist.

The fix, implemented in `omlx_client.py`:

* α = `Σ accept_num / Σ accept_den` (token-weighted, not a mean of ratios)
* runs with `accept_den < 50` are marked `valid=False` and excluded from every
  aggregate, with `n_valid` / `n_invalid` recorded so nothing disappears silently

### 4.2 Know your noise floor before reading a trend

Across 196 cells, the median within-cell standard deviation of α is 0.28 pt,
but the pooled σ is **6.35 pt** — a handful of cells are wildly unstable. With
n=2 the 95% confidence half-width is ±12.4 pt. A 5 pt difference between two
temperature settings means nothing. The whole first dataset was discarded and
re-run at n=3 once this was known.

### 4.3 α alone is not a quality signal

At `temperature=2.0, top_p=1.0` the model emits multilingual token soup —
`十分なLineColor بالاีด deficiência requisiti Pradesh tabindex` — 332 unique
characters in 2296, no coherent structure at all. Its acceptance rate was
**80.1%, the highest of any configuration tested**, and suspiciously uniform
across tasks (80.2 / 80.2 / 79.8).

Degenerate decoding *raises* α. Any rule of the form "pick depth from α" is
therefore satisfiable by a broken configuration. Pair α with a degeneracy
check (repetition ratio, unique-token ratio, or simply whether output hit
`max_tokens` on every run).

### 4.4 Order confounds everything

Page-cache and SSD-cache state carry over between cells. An early version of
the depth sweep ran depths in ascending order, which systematically favoured
whichever depth ran first. The fix is to randomise or reverse the order, warm
up after every reload, and treat any effect smaller than the noise floor as
absent.

### 4.5 A truncated field is not a field

`~/.claude/history.jsonl` stores a `display` field that looked like full user
prompts. It is a **truncated preview**: median 68 characters, frequently ending
mid-word (`"btw,"`, `"and also wor"`). The same prompts in the transcript files
have a median of 1125 characters. Benchmarking off the truncated view produced
prompts like `…go to the website and create a` — semantically incomplete, and
the resulting measurements were uninterpretable.

### 4.6 A cell needs more than one surviving sample

The per-run filter above (`accept_den >= 25`) stops one 3-token response from
poisoning a cell. It does not stop a cell whose *only* surviving sample is
itself small. Rust impl at T=1.2 produced `runs=[52.94]` with `den=51` — one
sample — and read as a 39-point collapse. High temperature makes the model
terminate early on that prompt, so there was nothing to measure: that is
"insufficient", not "collapsed". Cells now require at least 2 valid repeats and
are printed as `ins` otherwise.

The same issue one level up: a cell mean of per-run ratios weights a
39-draft-token run the same as a 574-draft-token one. Pooling by denominator at
cell level as well turned a "−25.4 pt temperature effect" into −12.2 pt, which
then agreed with the independent depth-1 measurement of −9.5 pt.

### 4.7 Do not measure the machine with a probe that shares it

Mid-run, throughput appeared to have dropped ~48% between two otherwise
identical conditions. The diagnosis went through memory pressure, swap
(12.9 GB in use), app-level GPU contention and thermal state before landing on
the actual cause: **the diagnostic probe was hitting the same server**, which
runs `max_concurrent_requests=1`. The client measures wall time, which includes
queueing behind my own request. The instrument was the confound.

Worse, it silently killed a full five-stage sweep — every experiment request
came back `409 Conflict`, each stage crashed seconds after starting, and the
orchestrator reported "all stages complete". `omlx_client.py` now retries 409
with exponential backoff, and `master`-style orchestration should never share a
server with anything else.

### 4.8 A silent 401 is worse than a loud one

This one produced a fake result that survived into a draft writeup, so it is
worth stating plainly.

Two scripts needed to flip `mtp_enabled` between arms of an A/B. Both logged in
with a bare `urllib.request.urlopen` and then issued the settings PUT. The
admin login hands back a **session cookie**, and `urllib` keeps no cookie state
between calls, so the PUT came back:

```
401 {"detail":"Admin authentication required"}
```

The same request under `curl -c jar -b jar` returned `{"success":true}`. And
the PUT was wrapped in

```python
try:
    urllib.request.urlopen(req, timeout=1800).read()
except Exception:
    pass          # response may be dropped while the reload is in flight
```

which is the natural thing to write, because the call legitimately blocks for
the whole model reload. So the 401 became a no-op, and **both arms ran the same
configuration**. The experiment reported MTP on is 6.9% faster than MTP off,
on 4 of 4 tasks, and that number was written up as a finding.

The evidence that would have caught it was in the results the whole time: the
"MTP off" arm still emitted one MTP statistics line per request, and reported
acceptance rates identical to the on arm's to two decimals. A disabled
speculative path cannot report 90.91% acceptance.

Three rules came out of it, all in `omlx_client.set_settings`:

1. Carry the cookie (`HTTPCookieProcessor`).
2. Never swallow a settings-write exception.
3. **Assert the readback.** The PUT response echoes the full settings blob, so
   a write that did not take is detectable before any measurement.

The experiment scripts go further and refuse to produce a number at all: the
MTP-off arm must emit zero MTP statistics lines, or the run aborts.

This also invalidated a published reproduction path. `run_sweep.py` in this
repo had been rewritten during packaging and reintroduced the same bug, so
anyone running it would have "measured" depths 1, 2, 3, 4, 6 and 8 against a
server sitting at depth 1. It is fixed, and the numbers in `data/` are from the
correct client — the v2 sweep log records a ~36 s reload for every depth change
and `0.0 s` for the one depth that was already set, which is what a real
change and a no-op look like.

### 4.9 The validator is the instrument, and it was wrong three times

Every quality number in this repo is produced by a regex or an AST predicate
written against a prompt. Those predicates are **instruments**, and this project
broke them three separate ways before auditing them properly. Each failure
looked identical in the output — a task that appeared to degrade with
temperature — because a *correct* answer scored as a failure:

| What the prompt said | What the validator checked | Result |
|---|---|---|
| `useRef` + `useEffect` | `useState` | every compliant run scored as a failure; the task came out as the most temperature-sensitive one in the set, which is the opposite of the truth |
| 「使用 `collections.OrderedDict` 实现」 | only `from collections import OrderedDict` | the model's `import collections` + `collections.OrderedDict()` — the prompt's own wording — was rejected |
| a `Shape` base class with 3 subclasses and `total_area()` | class name, `area`, `describe` | 100% at every temperature, a flat line drawn by the ruler |

The shared root cause is not carelessness: **requirements that can be written
as a regex get checked, requirements that require reading the artifact get
skipped.** The first three are cheap and satisfying. "超出时淘汰最久未使用的键"
— the entire point of an LRU cache — is neither, so it was never checked at
all, and the task reported a perfect score.

Two metrics were wrong in the same silent way. The self-consistency projection
for code tasks kept only the *names* of defined functions, which the prompt
itself enumerates, so it was a constant: 100% at every temperature, unrelated to
the model. And the JSON projection's bracket counter evaluated
`depth += depth + 1` on every `[`, so it never closed and every sample silently
fell back to raw text — exact-text comparison wearing a projection's name.

`AUDIT_CHECKLIST.md` classifies all ten of these failure modes. Three mechanical
guards came out of it, and they are what keep this from recurring:

- **A requirement ledger.** Every requirement a prompt states appears in
  `quality_checks.VALIDATION_SPEC` as either *checked* or *explicitly
  unmeasured with a reason*. Silence is not an option — a silently unchecked
  requirement is invisible in every table downstream. 56 requirements: 45
  checked, 11 declared unmeasured.
- **Single-violation differential fixtures.** For each checked requirement, a
  sample that breaks *only that one* must be rejected **naming that
  requirement**. A fixture rejected for the wrong reason proves nothing, so the
  reason is matched, not just the verdict.
- **A no-op mutation guard.** If a mutation leaves the text unchanged, the
  fixture tests nothing and says so. Two T6 fixtures were silently passing this
  way after their parent sample was edited without them.

Run them with `python3 src/audit_validators.py` and `python3 src/audit_metrics.py`.
The metrics are tested on *constructed* inputs, where the right answer is known
by construction: a pair that must differ must differ, a pair that means the same
must match.

### 4.10 When a regex cannot decide it, execute the artifact

Three of the requirements were "semantic": no pattern decides them. Two of them
are the *core* of their task, and a presence-only checker reports a perfect
score on both, forever.

| Requirement | Decided by |
|---|---|
| 「超出时淘汰最久未使用的键」 | executing the class |
| 「打印函数耗时」 | executing `timed` |
| `pi*r^2`, `w*h`, `b*h/2` arithmetically right | executing the shapes |
| 「用户手动上滚后暂停自动滚动」 | **not measured** — needs a JSX runtime |

For the Python ones, `exec` settles it. `src/exec_probes.py` compiles each
generation and drives it: `put A,B,C; get A; put D` — an LRU has just refreshed
A, so B is coldest and B is evicted; a FIFO evicts A. A static check cannot
separate those two, which is exactly why the task scored 100% for as long as it
did.

**Result: 26/26 for LRU eviction, 26/26 for `timed`, 26/26 for the areas** — at
every temperature including T=1.5. The model writes a genuinely correct LRU
every time. That claim is now *measured*, where before it was *assumed by a
checker that only looked for method names*.

The load-bearing negative control is a FIFO wearing an LRU's name; the probes
reject it by name (`this is FIFO, not LRU`), and every wrong-area variant is
rejected too. `python3 src/test_exec_checks.py` runs all 18.

`T10` stays unmeasured. Deciding it needs a JSX runtime and an effect scheduler,
there is no offline transpiler on this machine, and hand-writing one would mean
building another unverified instrument inside an audit whose entire subject is
unverified instruments. The honest move was to leave it on the ledger.

**11 requirements remain unmeasured**, and the list still contains the hardest
requirement in the set.

### 4.11 `tools/qa_audit` — the part that is not about this model

The mechanisms above are task-agnostic, so they are extracted into
`tools/qa_audit`: a requirement ledger, differential fixtures with
reason-matching and a no-op guard, metric degeneracy tests, and an isolated
subprocess runner for behavioural probes. No dependencies.

It audits itself. `python3 tools/qa_audit/selftest.py` feeds every guard a
deliberately broken input and **fails if the guard does not fire**, so the
guards are regression-protected rather than merely present. One guard caught a
bug in its own test while being written: a fixture named "unchecked rule
accepts violation" reused a task whose checker *did* inspect the rule, so the
condition it claimed to construct was never built. Same error class it exists
to detect, one level up.

## 5. Acceptance rate is a compute metric, not a quality metric

The most common misreading of α is "90% acceptance means one character in ten
is wrong, so my code has bugs in it".

α does not count errors. A speculative draft is only ever a **candidate**; when
verification rejects it, the character is replaced by the one the backbone
computed itself.

### How verification decides

From oMLX's decode path, `patches/mlx_lm_mtp/batch_generator.py`:

```python
if is_greedy:
    accept = verify_id == draft_id
else:
    log_accept = (verify_accept_lp[0, draft_id].item()
                  - draft_accept_lp[draft_id].item())
    accept = mx.random.uniform() < exp(log_accept)
```

Two regimes:

* **Greedy** (T=0, or `top_k=1`): exact token-id equality against what the
  backbone sampled at that position.
* **Stochastic**: the standard speculative-sampling rule
  `min(1, p_backbone(x) / p_draft(x))`, computed on *sampler-filtered*
  logprobs so it matches the distribution actually being drawn from — `top_p`
  and `top_k` included — rather than the raw softmax. A worse draft can still
  be accepted if its probability ratio clears the roll.

Either way the backbone is the reference, every cycle. Nothing is compared
against a stored answer key.

### The output is unchanged, but not bit-exactly

oMLX's Qwen4-Exp decoder sets `_omlx_mtp_row_exact_verify = True`, which makes
the multi-row verify forward run with the arithmetic of a serial one-row
decode, so that *"greedy MTP output equals MTP-off output"*.

Measured, same prompt, T=0, depth 1, `mtp_enabled` flipped between arms.
The toggle is verified — the off arm emits zero MTP statistics lines and no
acceptance figures at all:

| Task | α while on | chars on / off | first divergence | chars differing after it |
|---|---|---|---|---|
| Rust impl | 91.8% | 3096 / 3096 | — | identical |
| JSON output | 93.4% | 1741 / 1723 | char 685 | 950 |
| Creative prose | 56.7% | 682 / 681 | char 631 | 34 |

So the guarantee is real but **best-effort, not a proof**. The structured Rust
task holds exactly. The other two diverge, and the excerpts say why — the
divergence is a near-tie between two tokens the model genuinely could have
picked:

```
prose   ON  …莫名的感动。深秋的雨，洗净了尘埃…
        OFF …无名的感动。深秋的雨，洗净了尘埃…

JSON    ON  …"name": "Smart Lock",   "price": 249.99, "category": "Security"
        OFF …"name": "Robot Vacuum", "price": 299.99, "category": "Cleaning"
```

Both readings are correct. The verify forward and the serial decode forward
round the logits in a different order, so at a position where the top two tokens
are nearly tied the argmax can resolve either way — and once one character
differs, everything downstream re-tokenises. That is the 950: a single tie at
character 685 replaced one product record and restructured the rest of the
document. The prose task diverges later and cascades far less, but it hits
those ties constantly, because at α = 57% it lives near the decision boundary.

Two useful corollaries:

* **This is not corruption.** The rejected drafts never reach the output; both
  arms produced a valid answer. But do not promise byte-identical text, and do
  not use MTP where you need it.
* **α drifts between runs that produce identical text.** The same three tasks,
  same config, T=0, measured twice, gave α = 91.8 / 93.4 / 56.7 and
  α = 90.9 / 94.4 / 57.8 — with byte-identical output both times. α is a
  property of the draft sampling, not of the answer.

`OMLX_MTP_ROW_EXACT_VERIFY=0` restores the faster verify kernels and gives up
even the partial guarantee.

### Consequence

α is about how much compute was saved. Never use it as a quality signal, and
never pick depth from α alone without a degeneracy check (§4.3).

## 6. Temperature: acceptance barely moves, quality moves a lot

The first temperature sweep (§ below, 861 generations) measured the effect on
**acceptance rate** and found it small: 11 tasks × 6 temperatures × 2 depths ×
3 repeats, pooled within-cell σ = 3.62 pt, so at n=3 the 95% confidence
half-width is **±5.8 pt**. Against that threshold, most tasks are flat, and
only translation (−12.7 pt), TypeScript component (−11.0 pt) and Rust
(−9.5 pt) decline measurably, and only at the top of the range. The community
claim that T=1.0 drops α to 40–55% does not reproduce.

That is an **efficiency** result. It says nothing about whether the output is
still right, because α was never a quality metric (§5). So the second sweep
asks the question the router actually needed: 7 tasks × 6 temperatures × 5
replicates, scored by objective validators written from each prompt's own stated
requirements (JSON schema and field types, Python AST, the arithmetic answer,
required points, brace balance, stated length and simile count).

| T | mean valid | mean self-consistency | mean α |
|---|---|---|---|
| 0.0 | 100% | — | 85.1% |
| 0.3 | 100% | 74% | 85.3% |
| 0.6 | 97% | 63% | 85.6% |
| 0.9 | 91% | 54% | 84.9% |
| 1.2 | 83% | 54% | 83.4% |
| 1.5 | 83% | 85%* | 81.3% |

\* the T=1.5 self-consistency figure is over a mix of cells that include four
truncated runs; the valid column excludes those and the full per-task table is
in `RESULTS_temp_quality.md`.

**The comparison that matters is the last column against the first two.** Over
T=0 → 1.5, acceptance moves 3.8 pt. Over the same range, self-consistency falls
from 100% to 54% and objective validity from 100% to 83%. **α understates what
temperature does to the output by roughly an order of magnitude**, which is
exactly why the router's temperature values could not be derived from it.

Two further readings:

* **Code generation is the most temperature-robust thing here, and the
  TCP explainer is the least.** The two Python tasks stay 5/5 on content at
  every temperature through T=1.5. The TCP explainer is 5/5 only up to T=0.6 and
  then intermittently stops covering 「为什么不是四次」. So what temperature costs
  you is usually *coverage of a stated requirement*, not arithmetic, not syntax,
  and not prose.
* **JSON is flat.** Content validity is 4/5 at T=0.3 and 5/5 from 0.6 to 1.2,
  returning to 4/5 at 1.5. The residual failure is an array the model stopped
  writing the closing bracket on — not gradual decay. The earlier claim that
  "JSON breaks above 0.9 and the React component is the most fragile of all"
  came from a validator that checked `useState` when the prompt asks for
  `useRef`+`useEffect`; it scored compliant output as failure. See §4.9.
* **React's T=1.5 failures are truncation and collapse, not missing
  requirements.** Of the three failures, one hit the 1800-token cap mid-effect
  and two collapsed to 18 and 120 tokens of word salad. Note that `distinct-2`
  rates the 18-token sample at 1.000 — the metric is length-dependent and is
  blind to exactly the case where an output has collapsed to nothing.
* **The model ignores 「不要 markdown 标记」, at every temperature.** Code
  outputs arrive wrapped in ``` fences 73% (LRU), 81% (Shape) and 69% (React) of
  the time, including at T=0 where there is no sampling noise to blame. There is
  no temperature that fixes it. Strip fences client-side.
* **The prose row is not a quality result.** Prose "validity" checks that the
  output has ~600 Chinese characters, at least two simile markers, and avoids
  the two banned words. It cannot tell you 0.8 is nicer than 0.4, and the
  figure draws that line dashed for exactly this reason.

A control confirms temperature reached the sampler: `top_k=1` overrides
temperature and forces greedy, reproducing the T=0 numbers (71.7% vs 71.8%).

Two methodological notes from building this, both of which nearly invalidated it:

1. **Token budgets have to clear the task, and a truncated run is not a quality
   result.** T1, T2 and T7 were 100% truncated at their original budgets, which
   would have read as "temperature destroys structured output" while measuring
   nothing but a character limit. Budgets are now set from observed token counts
   with headroom, and `truncated` is recorded per run so the analysis can
   separate length from quality.
2. **A sweep that dies at run 31 of 182 must not start over.** One request came
   back with a response shape that had no `message.content`, with nothing in the
   server log past the previous completion — the request never reached the
   model. `omlx_client.generate()` now retries shape faults, records
   `transport_error`, and returns an empty result rather than raising; the
   runner resumes from whatever is on disk.

**Long context does not raise α.** Growing the prompt from 123 to 5299 tokens on
a fixed generation task leaves α slightly *lower* (95.1% → 92.5%) and
throughput slightly worse (49.3 → 46.9 tok/s). The n-gram PLE predicts from the
generated stream, not from the prompt, so extra context buys nothing for MTP
and costs cache bookkeeping.

### What this did and did not settle

It bounded where output breaks per task, which is what the router needs for
safe ceilings (`temp_measured_ceiling` in `data/router_profiles.json`). It did
**not** find the best temperature, because that needs a quality judge for
open-ended output and this repo has none. The ceilings were re-derived after
the validator audit and now carry the validator version that produced them, so
a future change to a check cannot silently leave a stale number in place
(`python3 src/check_router_staleness.py`).

The substantive reading, on content validity at n=5 per cell: code generation
holds to T=1.5, the TCP explainer degrades from T≈0.9 by dropping a stated
point, JSON and arithmetic are flat to T≈1.2, and React only fails at T=1.5 and
then by truncating or collapsing rather than by omitting a requirement.
"Measured safe to 1.5" is still not an argument for "1.5 is better than 0.3",
and for the prose row it is not a quality statement at all.

## 7. Results

The numeric tables in `RESULTS.md` and the figures in `figures/` are generated
from `data/` by `src/analyze.py`. They are not written by hand.

- `RESULTS.md` — throughput, cost decomposition, α vs depth, α vs temperature,
  noise floor, real-workload comparison, output identity, MTP on/off
- `figures/cost_per_token_vs_depth.png` — the U-shaped cost curve
- `figures/alpha_vs_temperature.png` — α in temperature
- `figures/alpha_vs_depth.png` — slow decay of α with depth
- `figures/real_workload_alpha_by_depth.png` — real-prompt distribution by depth
- `figures/mtp_on_off_vs_alpha.png` — the depth-1 on/off delta against α
- `figures/valid_rate_vs_temperature.png` — validity, self-consistency and α
  on one axis: the first two fall, the third does not
- `RESULTS_temp_quality.md` — the temperature × quality tables

### The workload matters more than the task type

Synthetic tasks and a real workload disagree, and the real workload is the one
that counts.

On the synthetic set the two tied depths split the tasks: depth 1 wins 5 of 11,
depth 2 wins 4, and depth 3/4 win one each. Six tasks gain meaningfully from
depth 2 (mean +10.5%, max +17.9% for the TypeScript component task) and five are
hurt. The per-task winners are not a robust signal — an earlier run of the same
experiment gave depth 2 a 6–4 lead instead, because the margins are inside the
±4.5% noise band.

On **16 prompts sampled from a real coding-agent history** — 56%
conversational/meta, 19% code, 19% research, 6% ops, matching the observed
distribution — **depth 1 wins cleanly**: higher α on 14/16 prompts (exact sign
test *p* = 0.0021) and higher throughput on 12/16, at a mean α advantage of
+2.3 pt. This reproduced across two independent full runs.

The reason is the task mix, not the model. Depth 2 only pays when acceptance is
high enough to keep the backbone's extra cost amortised, and most real
conversational turns sit in the middle of the α range (65–83%) where the added
verification is not repaid. A tuning decision made on synthetic benchmarks would
have picked the slower setting for this user's actual work.

### Is depth 1 worth having at all?

The depth sweep compares depths against each other. It never asks whether the
speculative path is worth switching on. Same prompt, T=0, `mtp_enabled` toggled,
3 repeats per arm:

| Task | α while on | MTP ON | MTP OFF | Δ | verdict |
|---|---|---|---|---|---|
| Repetitive code | 98.3% | 53.01 | 42.87 | **+23.6%** | faster |
| Python function | 96.1% | 45.48 | 39.41 | **+15.4%** | faster |
| Factual QA | 65.3% | 40.97 | 42.06 | −2.6% | within noise |
| Creative prose | 56.7% | 36.24 | 42.44 | **−14.6%** | slower |

Mean +5.5%, median +6.4%. The ordering is monotone in α, which is the point:
at depth 1 MTP is a large win on predictable text and a real loss on free text.
The per-run triples do not overlap in either direction — repetitive code ON
runs 51.5 / 53.8 / 53.8 against OFF 42.2 / 43.2 / 43.3, and prose ON
37.8 / 36.7 / 34.1 against OFF 42.5 / 42.9 / 41.9.

This also killed a confident piece of reasoning. The argument for switching MTP
off at depth 1 was: the head drafts one token, the backbone verifies one, so a
cycle emits one token whether the draft is accepted or not, and the head's
forward is pure overhead. Measured `tok/cycle` at depth 1 is **1.83** on
average (range 1.57–2.00), not 1.0 — the verify step covers the draft plus a
free bonus row, so an accepted cycle emits two. The number was in the server
log the whole time.

**Caveat that matters for reading the table.** `mtp_enabled` is one switch, and
oMLX offers no way to disable only the draft head, so the two arms also differ
in which decode kernel runs. The PLE block is not part of the toggle at all —
it executes as `if "ple" in self:` inside the decoder layer, adding the n-gram
embedding to hidden states every step regardless — so this is not a "with PLE
vs without PLE" comparison either. The effect is real and toggle-verified; its
attribution to the draft head specifically is not established by this design.

### Recommended settings

| Situation | depth |
|---|---|
| PLE resident in RAM (no offload) | 3–6; this whole analysis does not apply |
| PLE on SSD, workload α < 75% | **1**, or off entirely (prose loses 14.6%) |
| PLE on SSD, workload α 75–90% | 1 or 2, measure your own prompts |
| PLE on SSD, workload α > 90% and mostly code | **2** (up to +18% over depth 1) |
| Unsure | 1 — it is the safe end of the tied pair |

If you can only set one number: **depth 1**. It is never significantly worse
than depth 2 in aggregate, and it is the better choice for mixed workloads.
Below roughly α 60%, turning `mtp_enabled` off entirely beats any depth.

## 8. A router: which settings, for which request

Everything above picks one setting for the whole server. The measurements say
that is a compromise nobody chose: acceptance rate runs from 56.7% to 98.3%
across tasks, and the depth-1 on/off delta runs from −14.6% to +23.6% over
that range. A fixed temperature and a fixed depth gives up one end of that.

`src/router.py` classifies a prompt and returns parameters for it. It is
keyword-based on purpose — an LLM router would spend a generation to save a
few milliseconds, and would be harder to debug when it picks wrong.

**The hard constraint, which shapes the whole design:**

| parameter | routable per request? | why |
|---|---|---|
| `temperature`, `top_p`, `top_k`, `max_tokens`, `enable_thinking` | **yes** | sent in the request body; no reload |
| `mtp_fixed_depth`, `mtp_enabled` | **no** | model settings; any write unloads and reloads the model — 47 s measured |

So the router returns two different things: `params` for this request, and
`session_advice` for a decision you make at a session boundary.

```python
import router
r = router.route("帮我写个 Python 函数解析 CSV 文件")
print(r.explain())
# [code_function] 单函数 / 脚本实现  — strong
#   matched: 函数 / 脚本 / 具体语言实现
#   expected α: 70–99% (n=1 — a guess with a number attached)
#   per-request: temperature=0.3, top_p=0.95, top_k=0
#   session-level: mtp_enabled=True, recommended_depth=1
```

Two implementation details that turned out to matter:

* **Match the instruction, not the payload.** A prompt that ships a document
  ("extract the key points from these minutes: … 核心搜索模块重构已进入联调 …")
  contains the *document's* vocabulary. Matching the whole string routed that
  task under repetitive-code rewriting, because the minutes happened to say
  重构. `split_instruction()` prefers matches before the first blank line.
* **CJK must not carry `\b`.** In Python's Unicode `\b`, Chinese characters are
  word characters, so `批量|遍历所有` with a trailing `\b` never matched
  anything. Chinese alternatives go unanchored; English ones keep `\b`.

### What the router is actually worth, honestly

Two validators, because the first one is a trap:

| | before any fitting | after fitting to the same data |
|---|---|---|
| 11 synthetic tasks | 7/11 | 11/11 |
| **16 real prompts (holdout)** | **8/16 (50%)** | 16/16 — meaningless |

The in-sample 11/11 is noise. The **8/16** is the real reading, and the three
genuine errors it found were a rule bug (payload matching), a 20 pt band error
on translation, and a 20 pt band error on math. `src/fit_bands.py` now fits the
bands from measurements and prints the sample size behind each one, because a
band supported by a single observation is a guess with a number attached — and
six of the nine profiles currently have fewer than three.

**The finding that matters more than the score:** across all 16 real
coding-agent prompts, α runs **69.8% to 87.5%, median 75.5%**, and *none of them
drops below 60%*. So for this workload the depth and MTP advice is uniform and
unambiguous — **MTP on, depth 1 — for every single request.** The router's
depth routing has nothing to do here.

Which leaves temperature, and that is precisely the part this repo has **not**
measured. The temperature values in `router_profiles.json` are conventional
defaults; the experiments established temperature's effect on *efficiency* and
its degeneracy boundary (T≈2.0 collapses into token soup), not a
quality-versus-temperature curve. Building that curve needs a quality judge,
which this repo does not have. If the router is going to be trusted for
temperature, that measurement is the missing piece.

The one lever that *is* measured and closes the loop: oMLX logs
`accept=NNN/MMM` about a second into any request, so a workload's real α is
available after one call. `router.recommend_depth(alpha)` turns that number
into depth advice — at session boundaries, never per turn.

## 9. Reproducing

```bash
git clone <this repo> && cd mtp-depth-lab

# Point at your oMLX server. The api_key is only needed if your server
# enforces one; leave it empty otherwise.
export OMLX_BASE=http://127.0.0.1:8091
export OMLX_API_KEY=...
export OMLX_MODEL=Qwen3.8-Flash-Next-oQ4e-mtp
export OMLX_LOG=$HOME/.omlx/logs/server.log

cd src
python3 run_sweep.py --depths 1,2,3,4,6,8 --temps 0.0 --reps 3 -o ../data/depth_sweep.json
python3 run_sweep.py --depths 1,2 --temps 0.0,0.3,0.6,0.9,1.2,1.6 --reps 3 -o ../data/temp_sweep.json
python3 analyze.py --data-dir ../data --figure-dir ../figures
```

Two more experiments toggle `mtp_enabled` and each triggers a full model
reload, so they take several minutes apiece:

```bash
python3 verify_quality_identity.py ../data/quality_identity.json
python3 ab_mtp_on_off.py ../data/ab_mtp_on_off.json
python3 analyze.py --data-dir ../data --figure-dir ../figures   # re-emit RESULTS.md
```

The temperature × quality sweep scores 7 tasks against objective validators.
Run `test_quality_checks.py` first — a validator that accepts bad output would
make the whole sweep report 100% and measure nothing:

```bash
python3 test_quality_checks.py          # 29 known-good / known-bad expectations
python3 run_temp_quality.py --out ../data/temp_quality.json
python3 analyze_temp_quality.py
```

It is resumable (`--restart` to force a clean run) and takes about an hour.

Both abort rather than report a number if the toggle did not take — see §4.8.

The router and its validators need no server at all:

```bash
python3 router.py "写一个关于深夜便利店的短篇散文"    # explain a routing
python3 validate_router.py                           # 11 synthetic tasks
python3 fit_bands.py --results <real.json>           # refit α bands, dry run
python3 validate_router_holdout.py --prompts <p.json> --results <r.json>
```

`omlx_client.py` parses the server's own MTP statistics lines out of its log, so
the only requirement is that `OMLX_LOG` points at a file the server writes.
If your build does not emit those lines, the measurement path needs a different
source; the parsing regex is a single place to adapt.

**Do not point anything else at the server while a sweep runs.** It is
configured `max_concurrent_requests=1`; a second caller does not slow the
experiment down politely, it gets a 409 and the sweep dies (§4.7).

## 10. Data and privacy

`data/depth_sweep.json`, `data/temp_sweep.json`, `data/sampling.json`,
`data/context.json`, `data/quality_identity.json` and `data/ab_mtp_on_off.json`
are derived from the synthetic task set in `src/tasks.py` and contain no
personal data. `quality_identity.json` stores short excerpts around the first
divergence between the two arms, drawn from the same synthetic prompts.

`data/workload_aggregate.json` contains **only aggregate statistics** (α and
throughput per depth, plus the sign-test result) from a benchmark built on 16
real prompts from one person's coding-agent history. The prompts themselves,
and the scripts that mine them, are deliberately **not** redistributed — they
contain project names and personal context that are not theirs to publish.

The synthetic task set in `src/tasks.py` includes Rust and TypeScript tasks
because the original workload was Rust/TypeScript-heavy; that shape is
reproduced, the content is not.

## 11. Related

* [Jetson AGX Thor + vLLM, same model, PLE mmap'd](https://github.com/Leibniz-HBI/Qwen3.8-Flash-Next-Jetson-Thor-vLLM) — independently recommends `MTP=1`, which agrees with the shallow optimum found here.
* oMLX: <https://github.com/jundot/omlx>
