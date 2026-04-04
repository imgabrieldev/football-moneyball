---
tags:
  - pitch
  - frontend
  - react
  - vite
  - dashboard
  - k8s
---

# Pitch — Frontend Dashboard (v0.7.0)

## Problema

O sistema inteiro é CLI-only. Tem 15 comandos, 7 endpoints de API, dados de 87 partidas, previsões Monte Carlo e value bets — mas tudo só acessível via terminal. Não tem visualização em tempo real, não tem como compartilhar análises, não tem interface pra acompanhar a rodada.

O backend está completo (v0.1.0 a v0.6.0):
- API FastAPI com 7 endpoints funcionando
- Previsões Dixon-Coles com Brier 0.70
- Value bets com odds reais de 39 casas
- Backtesting com 87 partidas do Brasileirão 2026
- K8s com PostgreSQL + CronJobs de automação

Falta a camada visual.

## Solução

SPA React + Vite com 6 páginas consumindo a API FastAPI. Design dark theme (consistente com os plots matplotlib que já temos). Deploy como container nginx no Minikube.

### Páginas

#### 1. Dashboard (Home) — `/`
Visão geral da rodada atual:
- Cards com próximos jogos e previsões (P(H) / P(D) / P(A))
- Destaque de value bets da rodada (edge > 3%)
- Resumo: partidas analisadas, Brier score, ROI do backtest

#### 2. Previsões — `/predictions`
Todas as previsões da rodada em tabela:
- Time Casa vs Time Fora
- xG esperado (home/away)
- Probabilidades 1X2 com barra visual
- Over/Under 2.5, BTTS
- Placar mais provável
- Score matrix expandível

#### 3. Value Bets — `/value-bets`
Scanner de apostas com valor:
- Tabela: partida, mercado, aposta, modelo%, odds, edge%, EV, stake Kelly
- Filtros: edge mínimo, mercado (1X2, O/U), bankroll
- Ordenação por edge ou EV
- Cores: verde (edge alto), amarelo (moderado)

#### 4. Jogadores — `/players`
Tabela de jogadores com métricas:
- Filtro por time
- Colunas: jogos, minutos, gols, xG, assists, xA, passes, tackles
- Ordenação por qualquer coluna
- Barra visual xG vs gols reais (over/underperformance)

#### 5. Backtest — `/backtest`
Resultados do backtesting:
- Panel de métricas: ROI, hit rate, Brier, drawdown
- Gráfico de bankroll ao longo do tempo (line chart)
- Tabela de apostas recentes (ganhou/perdeu)

#### 6. Verificação — `/verify`
Modelo vs realidade:
- Tabela: partida, placar, previsão, resultado, acertou?
- Métricas: accuracy 1X2, accuracy O/U, Brier score
- Calibração visual (predicted prob vs actual frequency)

### Tech Stack

```
React 19 + Vite 6
├── React Router (client-side routing)
├── TanStack Query (data fetching + cache)
├── Tailwind CSS (styling — dark theme)
├── Recharts (gráficos — bankroll, calibração)
└── Lucide React (ícones)
```

Sem backend adicional — consome direto a API FastAPI (`/api/*`).

## Arquitetura

### Estrutura do frontend

```
frontend/
├── index.html
├── package.json
├── vite.config.ts
├── tailwind.config.ts
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── api/
│   │   └── client.ts          # fetch wrapper pra API
│   ├── components/
│   │   ├── Layout.tsx          # sidebar + header
│   │   ├── MatchCard.tsx       # card de partida com previsão
│   │   ├── ProbabilityBar.tsx  # barra visual H/D/A
│   │   ├── ValueBetRow.tsx     # linha de value bet
│   │   ├── PlayerTable.tsx     # tabela de jogadores
│   │   └── BankrollChart.tsx   # gráfico de evolução
│   ├── pages/
│   │   ├── Dashboard.tsx
│   │   ├── Predictions.tsx
│   │   ├── ValueBets.tsx
│   │   ├── Players.tsx
│   │   ├── Backtest.tsx
│   │   └── Verify.tsx
│   └── lib/
│       └── utils.ts            # formatação, cores, helpers
├── Dockerfile
└── nginx.conf                  # serve SPA + proxy /api → moneyball-api
```

### Módulos backend afetados

Nenhum módulo Python precisa mudar. O frontend consome a API existente.

Possível melhoria: adicionar CORS origin específico no `api.py` ao invés de `*`.

### Schema

Sem mudanças no PostgreSQL.

### Infra (K8s)

```
k8s/
├── (existentes)
├── frontend-deployment.yaml    # nginx + SPA
├── frontend-service.yaml       # porta 3000
└── kustomization.yaml          # + frontend resources
```

Nginx config:
```nginx
server {
    listen 3000;
    root /usr/share/nginx/html;
    
    location /api/ {
        proxy_pass http://moneyball-api:8000;
    }
    
    location / {
        try_files $uri /index.html;
    }
}
```

Acesso: `kubectl port-forward -n football-moneyball svc/moneyball-frontend 3000:3000`

## Escopo

### Dentro do Escopo

- [ ] Setup React + Vite + Tailwind + React Router
- [ ] API client (`src/api/client.ts`)
- [ ] Layout com sidebar dark theme
- [ ] Página Dashboard (cards de jogos + value bets)
- [ ] Página Predictions (tabela com probabilidades)
- [ ] Página Value Bets (tabela com filtros)
- [ ] Página Players (tabela com métricas, filtro por time)
- [ ] Página Backtest (métricas + gráfico bankroll)
- [ ] Página Verify (tabela + accuracy)
- [ ] Dockerfile (nginx + SPA build)
- [ ] K8s manifests (deployment + service)
- [ ] Proxy nginx /api → moneyball-api

### Fora do Escopo

- Autenticação / login
- Modo mobile (desktop-first, responsive básico)
- Real-time updates (WebSocket)
- Testes E2E (Cypress, Playwright)
- CI/CD pipeline
- PWA / offline
- Internacionalização (PT-BR only)

## Research Necessária

- [ ] Confirmar que Vite 6 + React 19 funcionam com Node disponível no Arch
- [ ] Escolher entre Recharts vs Chart.js pra gráficos

## Estratégia de Testes

### Frontend
- Manual: navegar cada página e verificar dados
- Verificar que API retorna dados corretos (já testado na v0.6.0)

### Integração
- `docker build` do frontend
- `kubectl apply` e verificar que nginx serve SPA
- Verificar que proxy `/api` funciona

## Critérios de Sucesso

- [ ] 6 páginas navegáveis com dados reais do Brasileirão
- [ ] Dashboard mostra previsões da rodada
- [ ] Value bets listadas com edge, odds e stake
- [ ] Tabela de jogadores com filtro por time
- [ ] Backtest mostra gráfico de bankroll
- [ ] Runs no Minikube via `kubectl port-forward`
- [ ] Tempo de carregamento < 2s por página
- [ ] Dark theme consistente
