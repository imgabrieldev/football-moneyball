import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { ProbabilityBar } from '../components/ProbabilityBar';
import { TrendingUp, BarChart3, Target } from 'lucide-react';

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

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Dashboard — Brasileirão 2026</h1>

      {/* Stats cards */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div className="flex items-center gap-2 text-gray-400 text-sm mb-1">
            <BarChart3 size={16} /> Previsões
          </div>
          <div className="text-2xl font-bold">{predictions?.total || 0}</div>
          <div className="text-xs text-gray-500">jogos na rodada</div>
        </div>
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div className="flex items-center gap-2 text-gray-400 text-sm mb-1">
            <TrendingUp size={16} /> Value Bets
          </div>
          <div className="text-2xl font-bold text-green-400">{valueBets?.value_bets?.length || 0}</div>
          <div className="text-xs text-gray-500">oportunidades encontradas</div>
        </div>
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div className="flex items-center gap-2 text-gray-400 text-sm mb-1">
            <Target size={16} /> Backtest ROI
          </div>
          <div className={`text-2xl font-bold ${backtest?.roi > 0 ? 'text-green-400' : 'text-red-400'}`}>
            {backtest?.roi !== undefined ? `${backtest.roi > 0 ? '+' : ''}${backtest.roi}%` : '—'}
          </div>
          <div className="text-xs text-gray-500">Brier: {backtest?.brier_score?.toFixed(3) || '—'}</div>
        </div>
      </div>

      {/* Predictions */}
      <div className="bg-gray-900 rounded-lg border border-gray-800">
        <div className="p-4 border-b border-gray-800">
          <h2 className="text-lg font-semibold">Próximos Jogos</h2>
        </div>
        {loadingPreds ? (
          <div className="p-8 text-center text-gray-500">Carregando previsões...</div>
        ) : (
          <div className="divide-y divide-gray-800">
            {predictions?.predictions?.slice(0, 8).map((pred: any, i: number) => (
              <div key={i} className="p-4 flex items-center gap-4">
                <div className="flex-1">
                  <div className="font-medium">{pred.home_team} vs {pred.away_team}</div>
                  <div className="text-sm text-gray-500">
                    xG: {pred.home_xg?.toFixed(2)} - {pred.away_xg?.toFixed(2)}
                  </div>
                </div>
                <div className="w-64">
                  <ProbabilityBar
                    home={pred.home_win_prob || 0}
                    draw={pred.draw_prob || 0}
                    away={pred.away_win_prob || 0}
                  />
                </div>
                <div className="text-sm text-gray-400 w-20 text-right">
                  {pred.most_likely_score}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Top Value Bets */}
      {valueBets?.value_bets && valueBets.value_bets.length > 0 && (
        <div className="bg-gray-900 rounded-lg border border-gray-800">
          <div className="p-4 border-b border-gray-800">
            <h2 className="text-lg font-semibold text-green-400">Top Value Bets</h2>
          </div>
          <table className="w-full text-sm">
            <thead className="text-gray-500 border-b border-gray-800">
              <tr>
                <th className="p-3 text-left">Partida</th>
                <th className="p-3 text-left">Aposta</th>
                <th className="p-3 text-right">Odds</th>
                <th className="p-3 text-right">Edge</th>
                <th className="p-3 text-right">Stake</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {valueBets.value_bets.slice(0, 5).map((vb: any, i: number) => (
                <tr key={i} className="hover:bg-gray-800/50">
                  <td className="p-3">{vb.match || `${vb.home_team} vs ${vb.away_team}`}</td>
                  <td className="p-3">{vb.market} — {vb.outcome}</td>
                  <td className="p-3 text-right">{vb.best_odds?.toFixed(2)}</td>
                  <td className="p-3 text-right text-green-400">+{(vb.edge * 100).toFixed(1)}%</td>
                  <td className="p-3 text-right">R$ {vb.stake?.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
