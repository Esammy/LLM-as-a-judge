# Architecture

judgekit is a library first, a CLI second, and a service third. Each layer is
usable without the ones above it, which is why `pip install judgekit` gets you
something useful and `docker compose up` gets you something else.

```
  CLI  /  CI job  /  Dashboard
              |  REST
         +----v-----+
         |   API    |  FastAPI - submit runs, query results, refuse bad diffs
         +----+-----+
              |  enqueue
         +----v-----+
         |  Redis   |  arq queue
         +----+-----+
              |
     +--------v---------+
     |   Worker pool    |  HPA 2 -> 20, scales on queue depth
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

## The layers

| Layer | Module | Depends on |
| --- | --- | --- |
| Core | `judgekit.core` | pydantic, numpy, pyyaml |
| Providers | `judgekit.providers` | httpx |
| CLI | `judgekit.cli` | typer, rich |
| Storage | `judgekit.storage` | sqlalchemy |
| Service | `judgekit.api`, `judgekit.worker` | fastapi, arq |

Only the last two need the `service` extra. The core imports nothing that talks
to a network.

## Decisions that shaped it

### The rubric fingerprint is the spine

A `Rubric` hashes its own grading-relevant fields — scale, criteria,
instructions, evidence flag — and nothing else. `description` and `version` are
excluded deliberately: renaming or re-documenting a rubric must not invalidate
historical scores, and the version is the *label for* the hash rather than part
of it.

That fingerprint then travels everywhere a score does: onto every `Judgement`,
into every `RunResult`, into a stored database row, out through the API, onto
the dashboard.

It exists so that `compare()` can refuse. Three failures, three different
remedies:

- **different rubric ids** — unrelated measurements, nothing to compare
- **different versions** — an intentional change; re-run the baseline
- **same version, different fingerprint** — a published rubric was **edited in
  place**, and every score on either side of that edit is quietly incompatible

The third is the dangerous one, because nothing else in a normal pipeline would
notice. `rubrics.lock.json` pins `version → fingerprint` so `judgekit rubric
verify` fails CI the way an unexpected dependency change does.

### Providers are a port, and the default reaches nothing

`Provider` is a `Protocol`, so an adapter has the right shape rather than the
right base class. The stub is the default everywhere — CLI, API, worker — which
means a mistyped flag or an unset variable degrades to a deterministic offline
judge instead of quietly spending money.

Real providers speak REST over `httpx` rather than importing vendor SDKs. Each
API is a single POST, and the Groq adapter uses the OpenAI chat-completions
shape, so it works against any OpenAI-compatible host by changing one URL.

### Errors are data, not exceptions

`Judge.judge()` never raises. A provider outage or an unparseable response comes
back as an `ERROR` judgement carrying its reason. One bad call cannot lose the
other 599 results in a run.

That choice propagates: errored cases are excluded from `mean_score` — an
unparseable response is a *missing* measurement, not a bad one — but they still
count against `pass_rate`, because a run that crashed did not pass. Both the
HTML report and the dashboard say so in words rather than leaving it implicit.

### Concurrency is bounded on purpose

The runner fans out through a semaphore rather than an unbounded `gather`. Every
real provider rate-limits, and an unbounded fan-out converts a fast run into a
cascade of 429s. Providers additionally hold a shared token bucket, so raising
`--concurrency` cannot exceed the per-minute allowance.

### Inline and queued execution share one implementation

`execute_run()` is called identically whether a job was dispatched to an arq
worker or run in-process. Two implementations would drift — the usual outcome
being that the queued path grows a behaviour the inline one silently lacks, and
only production notices.

The default queue is inline for the same reason the default provider is the
stub: `make api` should work with nothing else installed.

### Storage carries provenance, and history is scoped

Every run row stores the fingerprint, indexed. Stored results outlive the rubric
file that produced them, and the first thing anyone does with eval history is
plot it — so `GET /runs/{id}/history` returns only same-dataset,
same-fingerprint runs. A trend line that silently spans a rubric change is worse
than no trend line.

The dashboard's chart draws exactly that endpoint, and says so underneath.

### Kubernetes has to earn its place

Judging spends nearly all of its time waiting on a provider: slow, I/O-bound and
embarrassingly parallel. "Add pods" is therefore a genuine answer to "the queue
is deep", which is not true of most workloads people attach an HPA to.

The same property makes CPU the wrong scaling signal — an overloaded worker sits
near-idle while saturating a rate limit — so the HPA scales on
`judgekit_queue_depth` as an external metric. See
[deploy/k8s/README.md](../deploy/k8s/README.md).

## Testing

The suite runs with no API key and no network, and CI proves it rather than
asserting it: one job blackholes DNS and runs the tests again. Provider adapters
are covered through `httpx.MockTransport`; the API runs against an ASGI
transport and in-memory SQLite; tests that need a real provider are marked
`live` and **deselected by default**, so a developer with a key exported can
still run `pytest` without spending money.
