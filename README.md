# judgekit

**Most eval tools measure your model. This one measures your judge.**

[![CI](https://github.com/Esammy/llm-as-a-judge/actions/workflows/ci.yml/badge.svg)](https://github.com/Esammy/llm-as-a-judge/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

> **Status: v0.1.0.** Library, CLI, providers, service, Kubernetes manifests and
> dashboard are all implemented. 436 tests, no API key and no network required.

---

## The problem

Using an LLM to grade another LLM is now standard practice. Trusting that judge without
measuring it is also standard practice, and it should not be.

A judge that quietly disagrees with your humans is worse than no judge at all, because it
produces a number that looks like evidence. Three failure modes cause most of it:

- **Position bias.** Judges prefer whichever answer came first. In pairwise comparison this
  alone can decide the result.
- **Verbosity bias.** Judges reward length. Published measurements put the inflation at
  15 to 30 points of preference for the longer answer, regardless of whether it is better.
- **Self-preference.** Judges score outputs from their own model family higher.

And one that is less discussed but bites hardest in production:

- **Missing evidence.** A judge asked "is this figure hallucinated?" cannot answer honestly if
  it never sees the tool call the figure came from. It guesses, and it guesses conservatively,
  so correct answers get marked as fabrications.

On top of that, judges drift. A rubric that agreed with your reviewers in January may not in
April, so calibration is a recurring job rather than a one-off.

## What judgekit does differently

| | |
| --- | --- |
| **Version-pinned rubrics** | Editing a scoring prompt changes its content hash and bumps its version. The runner **refuses** to compare scores across rubric versions, because two runs graded under different rubrics are not comparable. Without this, a quality number tracked over time means nothing. |
| **Evidence-aware judging** | The judge receives the tool calls and retrieved context that produced an answer, not just the answer. This removes the most common false negative in production eval suites. |
| **Calibration you can fail a build on** | `judgekit calibrate` scores your judge against human labels and reports Cohen's kappa, Krippendorff's alpha, Spearman correlation and a confusion matrix. Set a floor; drop below it and CI fails. |
| **Bias measured, not just mitigated** | Position-swap, length normalisation and cross-family self-preference checks each report a **magnitude**, so you know how big the problem was, not merely that you addressed it. |
| **Runs with no API key** | The default provider is a deterministic stub. The whole suite runs offline, which makes CI free, hermetic and reproducible. |

## Quickstart

```bash
# no API key needed - the default provider is deterministic
uv sync --extra dev
uv run pytest          # 436 tests, offline, in about ten seconds
```

Judge a dataset from Python today:

```python
import asyncio
from judgekit.core.judge import Judge
from judgekit.core.models import Dataset
from judgekit.core.rubric import Rubric
from judgekit.core.runner import Runner, compare
from judgekit.providers.stub import StubProvider

dataset = Dataset.from_file("datasets/example.jsonl")
rubric = Rubric.from_file("rubrics/answer-quality.v2.yaml")

result = asyncio.run(Runner(Judge(rubric, StubProvider())).run(dataset))
print(f"{result.pass_rate:.0%} passed, mean {result.mean_score:.2f}")

# Comparing across a rubric version raises rather than returning a
# number that looks meaningful and is not.
v1 = Rubric.from_file("rubrics/answer-quality.v1.yaml")
old = asyncio.run(Runner(Judge(v1, StubProvider())).run(dataset))
compare(old, result)   # IncomparableScoresError: rubric differs in version
```

Or from the command line:

```bash
# Judge a dataset, write a standalone HTML report, save the run for later
judgekit run datasets/example.jsonl -r rubrics/answer-quality.v2.yaml     --out report.html --save run.json

# Fail CI if a published rubric was edited without a version bump
judgekit rubric verify              # exit 1 on violation

# See exactly why two rubrics are not comparable
judgekit rubric diff rubrics/answer-quality.v1.yaml rubrics/answer-quality.v2.yaml

# Diff two runs; refuses rather than printing a meaningless delta
judgekit compare baseline.json candidate.json --max-drop 0.02
```

The guard, as it appears in a CI log:

```
$ judgekit rubric verify
  [edited-without-bump] answer-quality@v2: content changed since it was locked
  (714e2ddba037 -> 929e015dce9b). Bump the version rather than editing a
  published rubric.
x 1 rubric violation(s)
$ echo $?
1
```

### Measuring the judge

This is the part nothing else ships. Give it cases carrying `human_label` and
it tells you whether your judge can be trusted at all:

```bash
$ judgekit calibrate datasets/example.jsonl -r rubrics/answer-quality.v2.yaml

metric              value  reading
quadratic kappa     0.702  substantial
cohen kappa         0.360  exact matches only; harsh on ordinal scales
krippendorff alpha  0.759  interval reliability
spearman            0.927  does it rank cases the way humans do?
mean abs error       0.70  scale points
systematic offset   +0.06  positive means generous
exact agreement       50%
within one point      62%

confusion matrix (rows: human, columns: judge)
human\judge     1     2     3     4     5
          1     0     1     1     0     0
          3     0     0     2     0     0
          4     0     0     1     1     0
          5     0     0     0     1     1
```

Two things in that table are worth dwelling on.

**Quadratic kappa says 0.702; plain Cohen's kappa says 0.360.** Same judge, same
data. Plain kappa only asks whether the two raters matched exactly, so it
punishes a 4-where-a-human-said-5 exactly as hard as a 1-where-a-human-said-5.
On an ordinal quality scale that is the wrong model, and it makes usable judges
look broken. Quadratic weighting is the headline for that reason.

**Spearman is 0.927 but exact agreement is only 50%.** The judge ranks cases
almost exactly as the humans do while frequently landing on a different number.
That is a threshold problem, not a rubric problem, and the two have different
fixes - which is why both are reported.

Add `--min-kappa 0.6` to make it a CI gate.

### Measuring bias

Every detector returns a **magnitude in a stated unit**, not a verdict. A
mitigation you cannot measure is one you cannot justify keeping or dropping.

```bash
$ judgekit bias datasets/example.jsonl -r rubrics/answer-quality.v2.yaml

ok position +0.000 scale points (n=8)
    the same answer scored +0.00 points differently between slots;
    0 of 8 cases moved at all
ok verbosity -0.038 correlation (n=8)
    length correlates -0.04 with judge-minus-human residual
ok self_preference +0.000 scale points (n=0)
    needs cases from the judge's own family (stub) and from others, tagged in
    metadata['generator_family']; found 0 own and 0 other
```

- **Position bias** scores identical content twice, once in each slot. Nothing
  but slot order changes, so any difference is attributable to order alone.
- **Verbosity bias** is measured against the *residual* - judge minus human -
  not the raw score. Long answers are often genuinely better, so correlating
  length with score just rediscovers that and would flag a well-calibrated
  judge. Only the residual answers the actual question.
- **Self-preference** says it could not be measured rather than returning a
  reassuring zero. "Not measured" and "measured as unbiased" are different
  claims about a judge.

The detectors are validated by injecting a known bias into the stub and
confirming they find it - which only works because the stub is neutral by
construction:

| | neutral stub | bias injected |
| --- | --- | --- |
| position | +0.000, 0/8 moved | **+0.922 pts, 8/8 moved, first slot** |
| verbosity | -0.038 | **+0.563 correlation** |

## Providers

```bash
judgekit run datasets/example.jsonl -r rubrics/answer-quality.v2.yaml   # stub, offline
GEMINI_API_KEY=... judgekit run ... -p gemini
GROQ_API_KEY=...   judgekit run ... -p groq
```

**No provider SDKs.** Each of these APIs is a single HTTP POST and `httpx` is
already a dependency, so importing two large transitive dependency trees would
buy nothing. It also means the Groq adapter speaks the OpenAI chat-completions
shape, and therefore works against any OpenAI-compatible host - Together,
Fireworks, OpenRouter, vLLM, Ollama, OpenAI itself - by changing one URL:

```python
GroqProvider(base_url="http://localhost:11434/v1", family="local")
```

**The default is always the stub.** Nothing reaches the network unless a
provider is named, so a mistyped flag or an unset variable degrades to a
deterministic offline judge rather than quietly spending money. Every provider
test runs through `httpx.MockTransport`, so the whole suite is offline; the two
tests that hit a real endpoint are marked `live` and **deselected by default**,
because a developer with a key exported should still be able to run `pytest`
without it costing anything.

What the shared HTTP layer handles:

- **Client-side rate limiting** via a shared token bucket. Free tiers are tight
  (~15 rpm on Gemini, ~30 on Groq) and the runner fans out, so without this the
  first thing a new user sees is a wall of 429s they read as a bug.
- **Backoff that honours `Retry-After`.** Guessing a delay the server already
  told you is both ruder and slower.
- **Refusing to retry what cannot succeed.** A 401 is not transient; retrying it
  three times turns "your key is wrong" into a slow, confusing failure.
- **Token and cost accounting** per model, so a run reports what it spent.

## Running it as a service

The CLI is enough for CI. The service exists for the case the CLI cannot serve:
storing eval history so it can be looked at over time, and scaling the work
horizontally.

```bash
# Standalone. SQLite, the stub provider, jobs run in-process.
# Nothing else needs to be installed.
make api

# The full stack: Postgres, Redis, API and a worker pool.
docker compose -f deploy/docker-compose.yml up --build

curl -X POST localhost:8000/runs -H 'content-type: application/json'      -d '{"dataset":"example.jsonl","rubric":"answer-quality.v2.yaml"}'
# {"run_id":"01cbbc57d33e","status":"queued"}

curl localhost:8000/runs/01cbbc57d33e
```

| | |
| --- | --- |
| `POST /runs` | Queue a dataset. 202 with a run id |
| `GET /runs` | Recent runs; filter by `rubric_fingerprint`, dataset, status |
| `GET /runs/{id}` | One run with every judgement |
| `GET /runs/{id}/history` | Runs that are **safe to chart alongside** this one |
| `POST /compare` | Diff two runs; **409** if they are not comparable |
| `/healthz` `/readyz` `/metrics` | Liveness, readiness, Prometheus |

Four decisions worth naming:

**The fingerprint is stored on every row.** Once results live in a database they
outlive the rubric file that produced them, and the first thing anyone does with
stored history is plot it. Without the fingerprint travelling alongside the
score, that chart will eventually splice together runs graded under different
rules - the same failure this project prevents at the CLI, reappearing at the
storage layer. `GET /runs/{id}/history` returns only same-dataset, same-
fingerprint runs for exactly that reason.

**`/healthz` and `/readyz` are not the same endpoint.** Liveness answers "is this
process alive" and deliberately touches nothing; readiness checks the database.
Wiring both to one handler is how a healthy pod gets restarted because Postgres
was briefly slow.

**`POST /compare` returns 409, not a number.** The request was well formed and
both runs exist - but answering it across a rubric change would produce
something that looks like evidence and is not.

**A failed run records why.** Jobs write `failed` with the reason rather than
vanishing, because a job that disappears is far harder to diagnose than one that
explains itself.

Logs are JSON by default, since a container's stdout is read by a machine first:

```json
{"ts":"2026-09-13T19:59:08","level":"INFO","logger":"judgekit.worker.tasks",
 "message":"run completed","run_id":"01cbbc57d33e","pass_rate":0.125,
 "mean_score":3.306,"errored":0,"cost_usd":0.0,"duration_s":0.0}
```

## Kubernetes

```bash
./deploy/k8s/demo.sh    # build, deploy to minikube, queue 30 runs, watch it scale
```

Plenty of projects attach an HPA to something that cannot usefully scale
horizontally. This one can, for a specific reason: **judging spends almost all
of its time waiting on a provider.** Runs are slow, I/O-bound and embarrassingly
parallel, so "the queue is deep" really is answered by "add pods". If that were
not true these manifests would be decoration, and the honest move would be to
delete them.

That same property makes **CPU the wrong scaling signal.** An overloaded worker
sits near-idle while saturating a rate limit, so a CPU-target HPA refuses to
scale exactly when the backlog is worst. The base HPA scales on
`judgekit_queue_depth` instead, delivered as an external metric through
prometheus-adapter. The minikube overlay patches in a CPU-based HPA because a
default cluster has no such pipeline - a documented compromise for the demo, not
the recommendation.

Also in there: separate liveness and readiness probes, a 300-second termination
grace period so a worker is never killed mid-run, fast scale-up with slow
scale-down, default-deny NetworkPolicies, a migration Job, and a nightly
calibration CronJob - because judges drift, so calibration is recurring work.

Details and the reasoning: [deploy/k8s/README.md](deploy/k8s/README.md).

> **Verification status.** Both images build, `docker compose up` brings the
> stack up and runs an eval end to end, and the local overlay has been deployed
> to a live minikube: a cold `apply -k` reaches a running API, worker, Postgres
> and Redis with a completed migration Job, 630 queued runs drain through the
> worker pool, and the HPA scales it 1 -> 4 -> 6 under the backlog before holding
> at 6 through the scale-down stabilisation window. `deploy/k8s/demo.sh`
> reproduces all of it in one command.

## Dashboard

```bash
cd dashboard && npm install && npm run dev      # http://localhost:3000
```

Run history, per-case drill-down with flags and judge reasoning, and a pass-rate
trend.

The trend is the part worth noting. It draws `GET /runs/{id}/history`, which the
API scopes to **one dataset and one rubric fingerprint** - and the page says so
underneath the chart. A quality line that silently spans a rubric change is the
single most convincing wrong answer an eval dashboard can give you, so the
scoping is enforced server-side rather than left as a filter somebody might
forget to apply.

The run list shows the fingerprint in its own column for the same reason: two
rows under the same rubric id and version but different fingerprints are not
comparable, and that is what a reader needs to notice before drawing a
conclusion from the column beside it.

## Documentation

- [Measuring the judge](docs/measuring-the-judge.md) - calibration, the four
  bias modes, and why plain Cohen's kappa is the wrong headline
- [Architecture](docs/architecture.md) - the layers and the decisions behind them
- [Kubernetes](deploy/k8s/README.md) - why the HPA is justified, and why CPU is
  the wrong signal
- [Contributing](CONTRIBUTING.md) &middot; [Security](SECURITY.md)

## Design

Evaluation is slow, I/O-bound and embarrassingly parallel, which is the whole reason the
worker pool scales horizontally.

```
  CLI  /  CI job  /  Dashboard
              |  REST
         +----v-----+
         |   API    |  FastAPI - submit runs, query results
         +----+-----+
              |  enqueue
         +----v-----+
         |  Redis   |  arq queue
         +----+-----+
              |
     +--------v---------+
     |   Worker pool    |  HPA 2 -> 20 pods, scales on queue depth
     |  (async, I/O)    |
     +--+------------+--+
        |            |
  +-----v----+   +---v--------------+
  | Postgres |   | Providers        |
  | runs,    |   | stub . Gemini    |
  | scores,  |   | Groq             |
  | rubrics  |   +------------------+
  +----------+
```

## Roadmap

- [x] **Phase 0** - Foundations: packaging, ruff, mypy strict, pytest, CI
- [x] **Phase 1** - Core library: models, rubrics, checks, judge, stub provider, runner
- [x] **Phase 2** - CLI and self-contained HTML report
- [x] **Phase 3** - Bias controls and human calibration
- [x] **Phase 4** - Gemini and Groq providers
- [x] **Phase 5** - Postgres, FastAPI, arq worker, docker-compose
- [x] **Phase 6** - Kubernetes: Kustomize, HPA, CronJob
- [x] **Phase 7** - Next.js dashboard, docs, first release

## Development

```bash
uv sync --all-extras
make check          # lint, typecheck, tests, coverage gate
.\make.ps1 check    # same, on Windows
```

The project uses ruff, mypy in strict mode, and an 85% coverage floor. `pytest` must always
pass with no API key and no network access; anything that needs a real provider is marked
`@pytest.mark.live` and skipped by default.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
