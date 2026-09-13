# Security

## Reporting

Open a [private security advisory](https://github.com/Esammy/llm-as-a-judge/security/advisories/new).
Please do not open a public issue for a vulnerability.

## What this project handles

Evaluation data is frequently production data. A dataset can contain customer
questions, account figures and retrieved documents; judge reasoning can quote
any of it back. Several design choices follow from that.

**Reports are self-contained.** The generated HTML makes no external request —
no CDN, no webfont, no analytics. A report that phones out cannot be attached to
a CI artifact or opened in a regulated environment without a review nobody wants
to do.

**Untrusted input is escaped.** Case ids and judge reasoning come from datasets
and model output, both of which can contain anything. They are HTML-escaped
before rendering; without that, an eval report is a script-injection vector
aimed at whoever opens it.

**File names from HTTP are resolved, not concatenated.** `POST /runs` takes a
dataset and rubric name. Both are resolved and checked to fall inside their
configured directory, so `../../etc/passwd`, an absolute path and a symlink are
all rejected.

**Secrets are never committed.** `deploy/k8s/base` ships a Secret with empty
values so `kubectl apply -k` works from a clean checkout and the required shape
is documented. The production overlay deletes it, and CI fails if the prod
overlay renders a Secret at all — so a missing secret store is a deployment
failure rather than a silent fallback to empty credentials.

**Containers run unprivileged.** Non-root user, read-only root filesystem, all
capabilities dropped, `seccompProfile: RuntimeDefault`, no privilege escalation.

**Images are scanned.** Trivy runs on every build; findings are reported to the
security tab rather than blocking, because a base-image CVE with no available
fix should not stop a release that does not depend on it.

**Secret scanning runs pre-commit.** gitleaks is in the hook set.

## What it does not do

There is **no authentication on the API**. It is designed to sit behind an
ingress that handles authn/authz, inside a cluster, on a private network. Do not
expose it directly to the internet.

Rate limiting is client-side only, aimed at staying inside a provider's
allowance. It is not a defence against an abusive caller.
