---
tags:
  - roadmap
---

# Roadmap — Football Moneyball

## Shipped

### v0.1.0 — Core Analytics (2026-04-03)

- [x] Extração de ~30 métricas individuais do StatsBomb
- [x] Rede de passes com networkx (centralidade, parcerias)
- [x] Embeddings PCA + clustering de arquétipos táticos
- [x] RAPM (Regularized Adjusted Plus-Minus) via Ridge
- [x] Busca por similaridade via pgvector (HNSW, cosine distance)
- [x] Visualizações: pass network no campo, radar, heatmap, RAPM ranking, sinergia
- [x] Scout reports em markdown/JSON
- [x] CLI Typer com 8 comandos e output Rich
- [x] Infra K8s: PostgreSQL 16 + pgvector no Minikube
- [x] `.claude/` com skills (pitch, research, postmortem, lint, check-arch) e rules

## Next

### v0.2.0 — Data Quality & Testing

- [ ] Testes unitários com pytest + fixtures StatsBomb
- [ ] Testes de integração com PostgreSQL
- [ ] Validação de dados (métricas fora de range, NaN handling)
- [ ] Logging estruturado
- [ ] Cache inteligente (TTL por competição)

### v0.3.0 — Analytics Avançado

- [ ] Expected Threat (xT) model
- [ ] Possession Value framework
- [ ] Player aging curves
- [ ] Team-level tactical profiles
- [ ] Multi-season embedding comparison

### v0.4.0 — UX & Export

- [ ] Dashboard web (Streamlit ou FastAPI + HTMX)
- [ ] Export PDF com gráficos inline
- [ ] Comparação de elencos completos
- [ ] Scouting filters (idade, posição, liga, preço estimado)
