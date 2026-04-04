import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { api } from '../api/client';

export function ValueBets() {
  const [minEdge, setMinEdge] = useState(0.03);
  const [bankroll, setBankroll] = useState(1000);

  const { data, isLoading } = useQuery({
    queryKey: ['valueBets', bankroll, minEdge],
    queryFn: () => api.valueBets(bankroll, minEdge),
  });

  const bets = data?.value_bets || [];

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Value Bets</h1>

      {/* Filters */}
      <div className="flex gap-4 items-end">
        <div>
          <label className="block text-xs text-gray-500 mb-1">Bankroll (R$)</label>
          <input
            type="number"
            value={bankroll}
            onChange={(e) => setBankroll(Number(e.target.value))}
            className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm w-32"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">Edge mínimo</label>
          <select
            value={minEdge}
            onChange={(e) => setMinEdge(Number(e.target.value))}
            className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm"
          >
            <option value={0.02}>2%</option>
            <option value={0.03}>3%</option>
            <option value={0.05}>5%</option>
            <option value={0.10}>10%</option>
          </select>
        </div>
        <div className="text-sm text-gray-500">
          {data?.total_matches || 0} partidas analisadas · {bets.length} value bets
        </div>
      </div>

      {isLoading ? (
        <div className="text-gray-500">Buscando value bets...</div>
      ) : (
        <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="text-gray-500 border-b border-gray-800">
              <tr>
                <th className="p-3 text-left">Partida</th>
                <th className="p-3 text-left">Mercado</th>
                <th className="p-3 text-left">Aposta</th>
                <th className="p-3 text-right">Modelo</th>
                <th className="p-3 text-right">Odds</th>
                <th className="p-3 text-right">Casa</th>
                <th className="p-3 text-right">Edge</th>
                <th className="p-3 text-right">EV</th>
                <th className="p-3 text-right">Stake</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {bets.map((vb: any, i: number) => (
                <tr key={i} className="hover:bg-gray-800/50">
                  <td className="p-3 font-medium">{vb.match || ''}</td>
                  <td className="p-3 text-gray-400">{vb.market}</td>
                  <td className="p-3">{vb.outcome}</td>
                  <td className="p-3 text-right">{(vb.model_prob * 100).toFixed(0)}%</td>
                  <td className="p-3 text-right font-mono">{vb.best_odds?.toFixed(2)}</td>
                  <td className="p-3 text-right text-gray-400">{vb.bookmaker}</td>
                  <td className="p-3 text-right text-green-400 font-medium">+{(vb.edge * 100).toFixed(1)}%</td>
                  <td className="p-3 text-right">{vb.ev?.toFixed(3)}</td>
                  <td className="p-3 text-right font-medium">R$ {vb.stake?.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {bets.length === 0 && (
            <div className="p-8 text-center text-gray-500">
              Nenhuma value bet com edge {'>'} {(minEdge * 100).toFixed(0)}%
            </div>
          )}
        </div>
      )}
    </div>
  );
}
