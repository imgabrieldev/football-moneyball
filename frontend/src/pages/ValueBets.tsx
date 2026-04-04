import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { api } from '../api/client';
import { Loader2 } from 'lucide-react';

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
      <p className="text-gray-500">Apostas onde nosso modelo encontra vantagem sobre as casas (melhor odd por aposta)</p>

      {/* Filters */}
      <div className="flex gap-4 items-end">
        <div>
          <label className="block text-xs text-gray-500 mb-1">Bankroll (R$)</label>
          <input type="number" value={bankroll} onChange={(e) => setBankroll(Number(e.target.value))}
            className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm w-32" />
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">Edge mínimo</label>
          <select value={minEdge} onChange={(e) => setMinEdge(Number(e.target.value))}
            className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm">
            <option value={0.02}>2%</option>
            <option value={0.03}>3%</option>
            <option value={0.05}>5%</option>
            <option value={0.10}>10%</option>
            <option value={0.15}>15%</option>
          </select>
        </div>
        <div className="text-sm text-gray-500">
          {data?.total_matches || 0} partidas • {bets.length} value bets
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center gap-3 justify-center py-20 text-gray-500">
          <Loader2 size={24} className="animate-spin" /> Buscando value bets...
        </div>
      ) : bets.length === 0 ? (
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-12 text-center text-gray-500">
          Nenhuma value bet com edge {'>'} {(minEdge * 100).toFixed(0)}%
        </div>
      ) : (
        <div className="space-y-2">
          {bets.map((vb: any, i: number) => {
            const edgePct = (vb.edge * 100).toFixed(1);
            const edgeColor = vb.edge >= 0.15 ? 'text-green-400' : vb.edge >= 0.08 ? 'text-green-300' : 'text-yellow-300';
            return (
              <div key={i} className="bg-gray-900 rounded-lg border border-gray-800 p-4 flex items-center justify-between">
                <div className="flex-1">
                  <div className="font-medium text-base">{vb.match}</div>
                  <div className="flex items-center gap-3 mt-1">
                    <span className="bg-gray-800 rounded px-2 py-0.5 text-xs">{vb.market === 'h2h' ? '1X2' : vb.market === 'totals' ? 'Gols' : vb.market}</span>
                    <span className="text-sm font-medium">
                      {vb.outcome === 'Over' ? 'Mais de 2.5 gols' :
                       vb.outcome === 'Under' ? 'Menos de 2.5 gols' :
                       vb.outcome === 'Draw' ? 'Empate' :
                       `Vitória ${vb.outcome}`}
                    </span>
                    <span className="text-xs text-gray-500">Modelo: {(vb.model_prob * 100).toFixed(0)}%</span>
                  </div>
                </div>
                <div className="flex items-center gap-5 text-sm">
                  <div className="text-center">
                    <div className="text-gray-500 text-xs">Odds</div>
                    <div className="font-mono text-lg">{vb.best_odds?.toFixed(2)}</div>
                  </div>
                  <div className="text-center">
                    <div className="text-gray-500 text-xs">Casa</div>
                    <div className="text-xs">{vb.bookmaker}</div>
                  </div>
                  <div className="text-center">
                    <div className="text-gray-500 text-xs">Edge</div>
                    <div className={`font-bold text-lg ${edgeColor}`}>+{edgePct}%</div>
                  </div>
                  <div className="text-center">
                    <div className="text-gray-500 text-xs">Stake</div>
                    <div className="font-medium">R$ {vb.stake?.toFixed(2)}</div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
