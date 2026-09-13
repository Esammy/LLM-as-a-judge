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

### Notes

- Reports are self-contained HTML with no external requests.
- The test suite runs with no API key and no network; CI proves it by
  blackholing DNS.
