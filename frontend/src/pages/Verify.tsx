import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { CheckCircle, XCircle } from 'lucide-react';

export function Verify() {
  const { data, isLoading } = useQuery({
    queryKey: ['verify'],
    queryFn: () => api.verify(),
  });

  if (isLoading) return <div className="text-gray-500">Verifying predictions...</div>;
  if (data?.error) return <div className="text-yellow-400">{data.error}</div>;

  const predictions = data?.predictions || [];

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Verification — Model vs Reality</h1>

      {/* Summary */}
      {data?.total_matches > 0 && (
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
            <div className="text-xs text-gray-500">Accuracy 1X2</div>
            <div className={`text-2xl font-bold ${data.accuracy_1x2 > 40 ? 'text-green-400' : 'text-red-400'}`}>
              {data.accuracy_1x2}%
            </div>
            <div className="text-xs text-gray-500">{data.correct_1x2}/{data.total_matches} correct</div>
          </div>
          <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
            <div className="text-xs text-gray-500">Accuracy Over/Under</div>
            <div className={`text-2xl font-bold ${data.accuracy_over_under > 50 ? 'text-green-400' : 'text-yellow-400'}`}>
              {data.accuracy_over_under}%
            </div>
            <div className="text-xs text-gray-500">{data.correct_over_under}/{data.total_matches}</div>
          </div>
          <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
            <div className="text-xs text-gray-500">Brier Score</div>
            <div className={`text-2xl font-bold ${data.avg_brier_score < 0.25 ? 'text-green-400' : 'text-yellow-400'}`}>
              {data.avg_brier_score?.toFixed(4)}
            </div>
            <div className="text-xs text-gray-500">{'<'} 0.25 = good</div>
          </div>
        </div>
      )}

      {/* Detail table */}
      {predictions.length > 0 && (
        <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="text-gray-500 border-b border-gray-800">
              <tr>
                <th className="p-3 text-left">Match</th>
                <th className="p-3 text-center">Score</th>
                <th className="p-3 text-center">Prediction</th>
                <th className="p-3 text-center">Actual</th>
                <th className="p-3 text-center">1X2</th>
                <th className="p-3 text-right">P(H)</th>
                <th className="p-3 text-right">P(D)</th>
                <th className="p-3 text-right">P(A)</th>
                <th className="p-3 text-center">O/U</th>
                <th className="p-3 text-right">Brier</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {predictions.map((p: any, i: number) => (
                <tr key={i} className="hover:bg-gray-800/50">
                  <td className="p-3 font-medium">{p.match}</td>
                  <td className="p-3 text-center font-mono">{p.score}</td>
                  <td className="p-3 text-center">{p.predicted}</td>
                  <td className="p-3 text-center">{p.actual}</td>
                  <td className="p-3 text-center">
                    {p.correct_1x2 ? <CheckCircle size={16} className="text-green-400 inline" /> : <XCircle size={16} className="text-red-400 inline" />}
                  </td>
                  <td className="p-3 text-right">{(p.home_prob * 100).toFixed(0)}%</td>
                  <td className="p-3 text-right">{(p.draw_prob * 100).toFixed(0)}%</td>
                  <td className="p-3 text-right">{(p.away_prob * 100).toFixed(0)}%</td>
                  <td className="p-3 text-center">
                    {p.correct_over ? <CheckCircle size={16} className="text-green-400 inline" /> : <XCircle size={16} className="text-red-400 inline" />}
                  </td>
                  <td className="p-3 text-right">{p.brier?.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {predictions.length === 0 && (
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-8 text-center text-gray-500">
          No verifiable matches. Games have not been played yet or data has not been ingested.
        </div>
      )}
    </div>
  );
}
