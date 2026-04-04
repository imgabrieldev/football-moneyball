import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { ProbabilityBar } from '../components/ProbabilityBar';
import { TrendingUp, BarChart3, Target, Loader2 } from 'lucide-react';

export function Dashboard() {
  const { data: predictions, isLoading: loadingPreds } = useQuery({
    queryKey: ['predictions'],
    queryFn: () => api.predictions(),
  });

  const { data: valueBets } = useQuery({
    queryKey: ['valueBets'],
    queryFn: () => api.valueBets(),
  });

  const { data: backtest } = useQuery({
    queryKey: ['backtest'],
    queryFn: () => api.backtest(),
  });

  const preds = predictions?.predictions || [];
  const vbets = valueBets?.value_bets || [];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Dashboard — Brasileirão 2026</h1>

      {/* Stats cards */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div className="flex items-center gap-2 text-gray-400 text-sm mb-1">
            <BarChart3 size={16} /> Previsões
          </div>
          <div className="text-2xl font-bold">{preds.length}</div>
          <div className="text-xs text-gray-500">jogos na rodada</div>
        </div>
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div className="flex items-center gap-2 text-gray-400 text-sm mb-1">
            <TrendingUp size={16} /> Value Bets
          </div>
          <div className="text-2xl font-bold text-green-400">{vbets.length}</div>
          <div className="text-xs text-gray-500">apostas com edge positivo</div>
        </div>
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div className="flex items-center gap-2 text-gray-400 text-sm mb-1">
            <Target size={16} /> Backtest ROI
          </div>
          <div className={`text-2xl font-bold ${(backtest?.roi || 0) > 0 ? 'text-green-400' : 'text-red-400'}`}>
            {backtest?.roi !== undefined ? `${backtest.roi > 0 ? '+' : ''}${backtest.roi}%` : '—'}
          </div>
          <div className="text-xs text-gray-500">Brier: {backtest?.brier_score?.toFixed(3) || '—'}</div>
        </div>
      </div>

      {/* Predictions cards */}
      <div className="bg-gray-900 rounded-lg border border-gray-800">
        <div className="p-4 border-b border-gray-800">
          <h2 className="text-lg font-semibold">Próximos Jogos</h2>
        </div>
        {loadingPreds ? (
          <div className="p-8 text-center text-gray-500 flex items-center justify-center gap-2">
            <Loader2 size={20} className="animate-spin" /> Carregando...
          </div>
        ) : preds.length === 0 ? (
          <div className="p-8 text-center text-gray-500">
            Sem previsões. Vá em Previsões e clique "Computar".
          </div>
        ) : (
          <div className="divide-y divide-gray-800">
            {preds.slice(0, 8).map((pred: any, i: number) => {
              const conf = pred.confidence;
              const confColor = conf === 'alta' ? 'text-green-400' : conf === 'media' ? 'text-yellow-400' : 'text-gray-500';
              return (
                <div key={i} className="p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex-1">
                      <div className="flex items-center gap-3">
                        <span className="font-bold text-base">{pred.home_team}</span>
                        <span className="text-gray-600 text-sm">vs</span>
                        <span className="font-bold text-base">{pred.away_team}</span>
                      </div>
                      <div className="text-sm text-gray-500 mt-0.5">
                        xG: {pred.home_xg?.toFixed(2)} - {pred.away_xg?.toFixed(2)}
                        {pred.goals_hint && <span className="ml-2 text-yellow-600">• {pred.goals_hint}</span>}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-mono text-lg">{pred.most_likely_score}</div>
                      <div className={`text-xs ${confColor}`}>
                        {conf === 'alta' ? '● Alta confiança' : conf === 'media' ? '● Confiança média' : '○ Baixa confiança'}
                      </div>
                    </div>
                  </div>
                  {pred.interpretation && (
                    <div className="text-sm text-cyan-400 mb-2">{pred.interpretation}</div>
                  )}
                  <div className="w-full">
                    <ProbabilityBar
                      home={pred.home_win_prob || 0}
                      draw={pred.draw_prob || 0}
                      away={pred.away_win_prob || 0}
                    />
                    <div className="flex justify-between text-xs text-gray-600 mt-1">
                      <span>{pred.home_team}</span>
                      <span>Empate</span>
                      <span>{pred.away_team}</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Top Value Bets */}
      {vbets.length > 0 && (
        <div className="bg-gray-900 rounded-lg border border-gray-800">
          <div className="p-4 border-b border-gray-800">
            <h2 className="text-lg font-semibold text-green-400">Top Value Bets</h2>
          </div>
          <div className="divide-y divide-gray-800">
            {vbets.slice(0, 5).map((vb: any, i: number) => (
              <div key={i} className="p-4 flex items-center justify-between">
                <div>
                  <div className="font-medium">{vb.match}</div>
                  <div className="text-sm text-gray-400">{vb.market} — {vb.outcome}</div>
                </div>
                <div className="flex items-center gap-6 text-sm">
                  <div className="text-right">
                    <div className="text-gray-500">Odds</div>
                    <div className="font-mono">{vb.best_odds?.toFixed(2)}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-gray-500">Edge</div>
                    <div className="text-green-400 font-bold">+{(vb.edge * 100).toFixed(1)}%</div>
                  </div>
                  <div className="text-right">
                    <div className="text-gray-500">Casa</div>
                    <div>{vb.bookmaker}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-gray-500">Stake</div>
                    <div className="font-medium">R$ {vb.stake?.toFixed(2)}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
