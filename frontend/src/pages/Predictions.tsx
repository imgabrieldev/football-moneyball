import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';
import { ProbabilityBar } from '../components/ProbabilityBar';
import { RefreshCw, Loader2 } from 'lucide-react';

export function Predictions() {
  const queryClient = useQueryClient();
  const { data, isLoading } = useQuery({
    queryKey: ['predictions'],
    queryFn: () => api.predictions(),
  });

  const recompute = useMutation({
    mutationFn: () => api.recomputePredictions(),
    onSuccess: () => {
      // Refetch after 30s (time for background computation)
      setTimeout(() => queryClient.invalidateQueries({ queryKey: ['predictions'] }), 30_000);
    },
  });

  const predictions = data?.predictions || [];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Previsões — Monte Carlo</h1>
          <p className="text-gray-500">{predictions.length} partidas previstas via Dixon-Coles + Poisson</p>
        </div>
        <button
          onClick={() => recompute.mutate()}
          disabled={recompute.isPending}
          className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-700 rounded-lg text-sm font-medium transition-colors"
        >
          {recompute.isPending ? (
            <><Loader2 size={16} className="animate-spin" /> Recomputando...</>
          ) : (
            <><RefreshCw size={16} /> Recomputar</>
          )}
        </button>
      </div>

      {recompute.isSuccess && (
        <div className="bg-green-900/30 border border-green-800 rounded-lg p-3 text-sm text-green-400">
          Previsões sendo recalculadas em background. A página atualiza automaticamente em ~30s.
        </div>
      )}

      {isLoading ? (
        <div className="flex items-center gap-3 justify-center py-20 text-gray-500">
          <Loader2 size={24} className="animate-spin" />
          Carregando previsões...
        </div>
      ) : predictions.length === 0 ? (
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-12 text-center">
          <p className="text-gray-400 mb-4">Nenhuma previsão pre-computada.</p>
          <button
            onClick={() => recompute.mutate()}
            disabled={recompute.isPending}
            className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg text-sm font-medium"
          >
            Computar Previsões
          </button>
        </div>
      ) : (
        <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="text-gray-500 bg-gray-900 border-b border-gray-800">
              <tr>
                <th className="p-3 text-left">Partida</th>
                <th className="p-3 text-right">xG H</th>
                <th className="p-3 text-right">xG A</th>
                <th className="p-3 w-52">Probabilidades</th>
                <th className="p-3 text-right">Over 2.5</th>
                <th className="p-3 text-right">BTTS</th>
                <th className="p-3 text-center">Placar</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {predictions.map((pred: any, i: number) => (
                <tr key={i} className="hover:bg-gray-800/50">
                  <td className="p-3 font-medium">
                    {pred.home_team} <span className="text-gray-500">vs</span> {pred.away_team}
                  </td>
                  <td className="p-3 text-right">{pred.home_xg?.toFixed(2)}</td>
                  <td className="p-3 text-right">{pred.away_xg?.toFixed(2)}</td>
                  <td className="p-3">
                    <ProbabilityBar
                      home={pred.home_win_prob || 0}
                      draw={pred.draw_prob || 0}
                      away={pred.away_win_prob || 0}
                    />
                  </td>
                  <td className="p-3 text-right">{pred.over_25 ? `${(pred.over_25 * 100).toFixed(0)}%` : '—'}</td>
                  <td className="p-3 text-right">{pred.btts_prob ? `${(pred.btts_prob * 100).toFixed(0)}%` : '—'}</td>
                  <td className="p-3 text-center font-mono">{pred.most_likely_score || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
