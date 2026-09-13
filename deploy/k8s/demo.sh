#!/usr/bin/env bash
# One command to stand the stack up on minikube and watch the worker pool scale.
#
#   ./deploy/k8s/demo.sh
#
# Needs a running Docker daemon (minikube's default driver) and minikube.
set -euo pipefail

NS=judgekit
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

say "Submitting 30 runs to build a backlog"
kubectl -n "$NS" port-forward svc/api 8000:80 >/dev/null 2>&1 &
PF_PID=$!
trap 'kill $PF_PID 2>/dev/null || true' EXIT
sleep 3

for _ in $(seq 1 30); do
  curl -sS -X POST localhost:8000/runs \
    -H 'content-type: application/json' \
    -d '{"dataset":"example.jsonl","rubric":"answer-quality.v2.yaml"}' >/dev/null
done
echo "30 runs queued"

say "Watching the worker pool scale (Ctrl-C to stop)"
echo "This is the part worth recording: judging is I/O-bound and parallel,"
echo "so a deeper queue really is answered by more pods."
kubectl -n "$NS" get hpa worker -w
