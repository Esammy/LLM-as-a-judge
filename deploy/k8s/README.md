# Kubernetes

```bash
./deploy/k8s/demo.sh                        # build, deploy, load, watch it scale
kubectl apply -k deploy/k8s/overlays/local  # or just deploy
kubectl kustomize deploy/k8s/overlays/prod  # render production without applying
```

## Why Kubernetes is here at all

Plenty of projects attach an HPA to a workload that cannot usefully scale
horizontally. This one can, and the reason is specific: **judging spends almost
all of its time waiting on a provider.** A run is slow, I/O-bound and
embarrassingly parallel, so "the queue is deep" genuinely is answered by "add
pods". If that were not true, these manifests would be decoration and the
honest thing would be to delete them.

That same property makes **CPU the wrong scaling signal.** An overloaded worker
here sits near-idle while saturating a rate limit, so a CPU-target HPA would
refuse to scale at exactly the moment the backlog is worst. The base HPA
therefore scales on `judgekit_queue_depth`, exported by the API and delivered as
an external metric through prometheus-adapter.

The local overlay patches in a CPU-based HPA instead, because a default
minikube has no such metrics pipeline. That is a documented compromise for the
demo, not the recommended configuration.

## Layout

```
base/
  namespace.yaml   config.yaml            namespace, ConfigMap, placeholder Secret
  postgres.yaml    redis.yaml             StatefulSet with a PVC; Redis
  api.yaml                                Deployment, Service, PDB, probes
  worker.yaml                             Deployment, HPA (2 -> 20), PDB
  jobs.yaml                               migration Job; nightly calibration CronJob
  network.yaml                            default-deny, allow-lists, Ingress
overlays/
  local/                                  minikube: 1 replica, CPU HPA, no ingress policy
  prod/                                   3 API replicas, pinned tags, no committed Secret
```

## Decisions worth reading

**Liveness and readiness are different probes.** `/healthz` touches nothing;
`/readyz` checks the database. Pointing liveness at readiness is how a brief
Postgres blip turns into every API pod being restarted.

**Workers get a 300-second termination grace period.** A run takes minutes and
has already paid for its provider calls; killing it mid-flight wastes them.

**Scale-up is fast, scale-down is slow.** Backlog is the problem being solved,
so the HPA reacts in 30 seconds. It waits 5 minutes before removing a pod,
because a queue that just drained is often about to fill again.

**One job per worker pod.** Each run already fans out internally against the
provider's rate limit. Stacking runs inside a pod would multiply concurrency in
a way nothing accounts for, so capacity is added with pods, not job slots.

**Default-deny comes first.** Without it the other NetworkPolicies do nothing:
Kubernetes allows all traffic to a pod that no policy selects.

**The committed Secret is a placeholder.** It exists so `apply -k` works from a
clean checkout and so the required shape is documented. The prod overlay
deletes it, which makes a missing secret store a deployment-time failure rather
than a silent fallback to empty credentials.

## Status

Deployed and exercised on minikube (Kubernetes v1.35.1, Docker driver, 4 CPUs),
not merely rendered. From a cold `kubectl delete namespace judgekit`:

| | |
| --- | --- |
| `apply -k overlays/local` | 18 objects, no deprecation warnings |
| migration Job | completes on the first attempt |
| API / worker / Postgres / Redis | all Ready, zero restarts |
| 630 runs submitted | all `completed` |
| HPA under backlog | CPU 204% of request, replicas 1 -> 4 -> 6 |
| after the queue drained | CPU 7%, pool held at 6 for the 300s window |

Three things that deploy found, all fixed:

- the migration Job raced Postgres on a cold cluster and burned three of its
  four attempts on DNS failures - it now waits on an init container
- the worker crash-looped once against a not-yet-ready Redis, which Kubernetes
  recovered on its own; the init container above incidentally removed that too
- `demo.sh` submitted 30 runs, about six seconds of work, which drained before
  the HPA sampled it at all - the autoscaler looked broken when the load was
  simply too small to be a backlog

`demo.sh` reproduces the whole sequence in one command.
