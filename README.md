# judgekit

**Most eval tools measure your model. This one measures your judge.**

[![CI](https://github.com/Esammy/llm-as-a-judge/actions/workflows/ci.yml/badge.svg)](https://github.com/Esammy/llm-as-a-judge/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

> **Status: early development.** The core library and CLI are being built in the open.
> Sections marked _planned_ are not implemented yet. See [Roadmap](#roadmap).

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
uv run pytest          # 256 tests, offline, in about two seconds
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

_Planned (phase 3):_

```bash
judgekit calibrate --labels datasets/example.jsonl --min-kappa 0.6
judgekit bias --dataset datasets/example.jsonl
```

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
- [ ] **Phase 3** - Bias controls and human calibration
- [ ] **Phase 4** - Gemini and Groq providers
- [ ] **Phase 5** - Postgres, FastAPI, arq worker, docker-compose
- [ ] **Phase 6** - Kubernetes: Kustomize, HPA, CronJob
- [ ] **Phase 7** - Next.js dashboard, docs, first release

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
