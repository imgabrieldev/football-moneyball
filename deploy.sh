#!/bin/bash
# Deploy completo — build, push, restart, port-forward
set -e

TAG="${1:-latest}"
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "=== Building backend (moneyball:$TAG) ==="
docker build --no-cache -t "moneyball:$TAG" .

echo "=== Building frontend (moneyball-frontend:$TAG) ==="
cd frontend && npm run build 2>&1 | tail -3 && cd ..
docker build --no-cache -t "moneyball-frontend:$TAG" frontend/

echo "=== Loading images into minikube ==="
minikube image load "moneyball:$TAG"
minikube image load "moneyball-frontend:$TAG"

echo "=== Deploying to K8s ==="
kubectl -n football-moneyball set image deployment/moneyball-api "moneyball-api=moneyball:$TAG"
kubectl -n football-moneyball set image deployment/moneyball-frontend "frontend=moneyball-frontend:$TAG"
kubectl -n football-moneyball rollout status deployment/moneyball-api --timeout=90s
kubectl -n football-moneyball rollout status deployment/moneyball-frontend --timeout=60s

echo "=== Restarting port-forwards ==="
pkill -f "kubectl port-forward.*football-moneyball" 2>/dev/null || true
sleep 1
kubectl port-forward -n football-moneyball svc/moneyball-frontend 3000:3000 &
kubectl port-forward -n football-moneyball svc/postgres 5432:5432 &
sleep 2

echo ""
echo "=== DONE ==="
echo "Frontend: http://localhost:3000"
echo "API docs: http://localhost:3000/api/docs"
kubectl -n football-moneyball get pods
