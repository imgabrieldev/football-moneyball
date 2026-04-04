import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { ProbabilityBar } from '../components/ProbabilityBar';

export function Predictions() {
  const { data, isLoading } = useQuery({
    queryKey: ['predictions'],
    queryFn: () => api.predictions(),
  });

  if (isLoading) return <div className="text-gray-500">Carregando previsões...</div>;

  const predictions = data?.predictions || [];

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Previsões — Monte Carlo</h1>
      <p className="text-gray-500">{predictions.length} partidas previstas via Dixon-Coles + Poisson</p>

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
    </div>
  );
}
