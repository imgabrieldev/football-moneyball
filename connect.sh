#!/bin/bash
# Reconecta port-forwards sem rebuild
pkill -f "kubectl port-forward.*football-moneyball" 2>/dev/null
pkill -f "node.*vite" 2>/dev/null
sleep 1

kubectl port-forward -n football-moneyball svc/moneyball-frontend 3000:3000 &
kubectl port-forward -n football-moneyball svc/postgres 5432:5432 &
sleep 2

echo "Frontend: http://localhost:3000"
kubectl -n football-moneyball get pods --no-headers
