#!/usr/bin/env bash
# One command to stand the stack up on minikube and watch the worker pool scale.
#
#   ./deploy/k8s/demo.sh
#
# Needs a running Docker daemon (minikube's default driver) and minikube.
set -euo pipefail

NS=judgekit
RUNS="${RUNS:-600}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

say "Starting minikube"
minikube status >/dev/null 2>&1 || minikube start --cpus=4 --memory=4096
minikube addons enable metrics-server

say "Building images inside minikube's Docker daemon"
# Build straight into the cluster's daemon: no registry, no push, no pull.
eval "$(minikube docker-env --shell bash)"
docker build -f "$ROOT/deploy/docker/Dockerfile" --target api    -t judgekit-api:dev    "$ROOT"
docker build -f "$ROOT/deploy/docker/Dockerfile" --target worker -t judgekit-worker:dev "$ROOT"

say "Applying the local overlay"
kubectl apply -k "$ROOT/deploy/k8s/overlays/local"

say "Waiting for Postgres and Redis"
kubectl -n "$NS" rollout status statefulset/postgres --timeout=180s
kubectl -n "$NS" rollout status deployment/redis --timeout=120s

say "Waiting for the migration job"
kubectl -n "$NS" wait --for=condition=complete job/migrate --timeout=180s

say "Waiting for the API and worker"
kubectl -n "$NS" rollout status deployment/api --timeout=180s
kubectl -n "$NS" rollout status deployment/worker --timeout=180s

say "Cluster state"
kubectl -n "$NS" get pods,hpa

say "Submitting $RUNS runs to build a backlog"
kubectl -n "$NS" port-forward svc/api 8000:80 >/dev/null 2>&1 &
PF_PID=$!
trap 'kill $PF_PID 2>/dev/null || true' EXIT
sleep 3

# The size here is load-bearing. An earlier version of this script submitted 30
# runs; against the stub judge that is roughly six seconds of work, so the whole
# backlog drained before the HPA's 15-second sync period even sampled it, and
# the demo sat at one pod looking like the autoscaler was broken. A backlog has
# to outlive the controller's reaction time to be a backlog at all.
#
# Submitted in parallel batches because 600 sequential round-trips through a
# port-forward take longer than the work itself.
for i in $(seq 1 "$RUNS"); do
  curl -sS -X POST localhost:8000/runs     -H 'content-type: application/json'     -d '{"dataset":"example.jsonl","rubric":"answer-quality.v2.yaml"}' >/dev/null &
  if [ $((i % 40)) -eq 0 ]; then wait; fi
done
wait
echo "$RUNS runs queued"

say "Watching the worker pool scale (Ctrl-C to stop)"
cat <<'NOTE'
This is the part worth recording: judging is I/O-bound and parallel, so a
deeper queue really is answered by more pods.

Expect roughly, on a 4-CPU minikube:

  ~30s   CPU crosses the 60% target  (it reached 204% here)
  ~45s   REPLICAS 1 -> 4
  ~165s  REPLICAS    -> 6            (maxReplicas in the local overlay)
  then   CPU falls to single digits and the pool HOLDS at 6 for five
         minutes before scaling in - that is the scale-down stabilisation
         window doing its job, not a stuck autoscaler.

Note the local overlay scales on CPU, which is the WRONG signal for this
workload and only works here because the stub judge does its scoring locally
instead of waiting on a provider. The base manifests scale on queue depth.
See overlays/local/hpa-cpu.yaml.
NOTE
kubectl -n "$NS" get hpa worker -w
