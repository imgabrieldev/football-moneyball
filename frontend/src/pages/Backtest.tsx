import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export function Backtest() {
  const { data, isLoading } = useQuery({
    queryKey: ['backtest'],
    queryFn: () => api.backtest(),
  });

  if (isLoading) return <div className="text-gray-500">Running backtesting...</div>;
  if (data?.error) return <div className="text-yellow-400">{data.error}</div>;

  const stats = [
    { label: 'Initial Bankroll', value: `R$ ${data?.initial_bankroll?.toFixed(2)}`, color: '' },
    { label: 'Final Bankroll', value: `R$ ${data?.final_bankroll?.toFixed(2)}`, color: data?.final_bankroll > data?.initial_bankroll ? 'text-green-400' : 'text-red-400' },
    { label: 'ROI', value: `${data?.roi > 0 ? '+' : ''}${data?.roi?.toFixed(1)}%`, color: data?.roi > 0 ? 'text-green-400' : 'text-red-400' },
    { label: 'Matches', value: data?.matches_analyzed, color: '' },
    { label: 'Bets', value: `${data?.bets_won || 0}/${data?.bets_placed || 0} (${data?.hit_rate?.toFixed(0)}%)`, color: '' },
    { label: 'Brier Score', value: data?.brier_score?.toFixed(4), color: data?.brier_score < 0.25 ? 'text-green-400' : 'text-yellow-400' },
    { label: 'Max Drawdown', value: `${data?.max_drawdown?.toFixed(1)}%`, color: 'text-red-400' },
    { label: 'Average Edge', value: `${data?.avg_edge?.toFixed(2)}%`, color: 'text-green-400' },
  ];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Backtesting</h1>
      <p className="text-gray-500">Simulation with {data?.matches_analyzed || 0} Brasileirão 2026 matches</p>

      {/* Stats grid */}
      <div className="grid grid-cols-4 gap-3">
        {stats.map((s, i) => (
          <div key={i} className="bg-gray-900 rounded-lg p-3 border border-gray-800">
            <div className="text-xs text-gray-500">{s.label}</div>
            <div className={`text-lg font-bold ${s.color}`}>{s.value}</div>
          </div>
        ))}
      </div>

      {/* Bankroll chart placeholder */}
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
        <h2 className="text-lg font-semibold mb-4">Bankroll Evolution</h2>
        <div className="h-64 flex items-center justify-center text-gray-500">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data?.bankroll_history?.map((v: number, i: number) => ({ bet: i, value: v })) || []}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="bet" stroke="#6b7280" fontSize={12} />
              <YAxis stroke="#6b7280" fontSize={12} />
              <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }} />
              <Line type="monotone" dataKey="value" stroke="#22c55e" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
