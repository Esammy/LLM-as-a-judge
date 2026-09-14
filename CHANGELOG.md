# Changelog

All notable changes are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Core** — version-pinned rubrics with content fingerprints, evidence-aware
  judging, deterministic checks, an async runner with bounded concurrency, and
  `compare()` that refuses runs graded under differing rubrics.
- **Calibration** — quadratic weighted kappa, Cohen's kappa, Krippendorff's
  alpha, Spearman, Pearson, MAE, systematic offset and a confusion matrix, with
  `--min-kappa` as a CI gate.
- **Bias** — position, verbosity and self-preference detectors, each reporting a
  magnitude in a stated unit. Verbosity is measured against the judge-minus-human
  residual so a well-calibrated judge is not flagged for agreeing with people.
- **CLI** — `run`, `validate`, `compare`, `calibrate`, `bias`, and
  `rubric list/lock/verify/diff`, with exit codes meant for CI.
- **Providers** — Gemini and Groq over REST, no vendor SDKs. The Groq adapter is
  OpenAI-shaped, so it works against any compatible host. A deterministic stub is
  the default everywhere.
- **Service** — FastAPI, Postgres via SQLAlchemy with Alembic migrations, an arq
  worker pool, Prometheus metrics and structured JSON logs.
- **Kubernetes** — Kustomize base with local and prod overlays, an HPA scaling on
  queue depth, a migration Job, a nightly calibration CronJob and default-deny
  NetworkPolicies.
- **Dashboard** — Next.js run history, a fingerprint-scoped pass-rate trend and
  per-case drill-down.

### Fixed

First deploy to real Docker and a live minikube. Every one of these was
invisible to a green test suite.

- **The arq worker never started.** `WorkerSettings.redis_settings` was a
  `@staticmethod`, and arq reads its configuration from `settings_cls.__dict__`
  rather than with `getattr`, so it received the descriptor instead of the
  value and died on `'staticmethod' object has no attribute 'host'`. The API
  kept accepting runs the whole time; they simply queued forever with no
  consumer.
- **Source changes did not reach the container.** The uv cache mount outlives a
  build and uv resolves the local project by name and version, so while this
  stayed `judgekit==0.1.0` every rebuild reinstalled the wheel built from the
  first-ever source tree. `--reinstall-package judgekit` forces the rebuild.
- **`.dockerignore` was never read.** It sat in `deploy/docker/`, next to the
  Dockerfile, which is neither location a builder looks in. Moving it to the
  build-context root cut the context from 699MB to 1.4MB and stopped `.venv/`
  and `.git/` being shipped to the daemon.
- **The dashboard could never pass its health check.** Next's standalone server
  binds to `$HOSTNAME`, which Docker sets to the container id, so it listened
  only on the container's own IP. Published ports still worked, which is what
  hid it. Pinned to `0.0.0.0`.
- **The migration Job raced Postgres.** On a cold cluster it spent three of its
  four attempts failing DNS resolution; a slightly slower database would have
  failed the deploy outright. It now waits on an init container.
- **`demo.sh` demonstrated nothing.** Thirty runs against the stub judge is
  about six seconds of work, which drained before the HPA's 15-second sync
  period sampled it, so the pool sat at one pod. Raised to 600 runs submitted in
  parallel, which scales it 1 -> 4 -> 6.

### Notes

- Reports are self-contained HTML with no external requests.
- The test suite runs with no API key and no network; CI proves it by
  blackholing DNS.
