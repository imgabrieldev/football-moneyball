import { useQuery } from '@tanstack/react-query';
import { useState, useEffect } from 'react';
import { api } from '../api/client';
import { ProbabilityBar } from '../components/ProbabilityBar';
import { RefreshCw, Loader2 } from 'lucide-react';

export function Predictions() {
  const [computing, setComputing] = useState(false);
  const [error, setError] = useState('');

  const { data, isLoading } = useQuery({
    queryKey: ['predictions'],
    queryFn: () => api.predictions(),
    refetchInterval: computing ? 5_000 : false, // Poll every 5s while computing
  });

  const predictions = data?.predictions || [];

  // Stop polling when predictions arrive
  useEffect(() => {
    if (computing && predictions.length > 0) {
      setComputing(false);
    }
  }, [predictions.length, computing]);

  async function handleRecompute() {
    setComputing(true);
    setError('');
    try {
      const res = await fetch('/api/predictions/recompute', { method: 'POST' });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }
    } catch (e: any) {
      setError(`Erro: ${e.message}`);
      setComputing(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Previsões — Monte Carlo</h1>
          <p className="text-gray-500">
            {predictions.length} partidas • Dixon-Coles + Poisson (10K simulações)
          </p>
        </div>
        <button
          onClick={handleRecompute}
          disabled={computing}
          className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-700 disabled:cursor-wait rounded-lg text-sm font-medium transition-colors"
        >
          {computing ? (
            <><Loader2 size={16} className="animate-spin" /> Computando...</>
          ) : (
            <><RefreshCw size={16} /> Recomputar</>
          )}
        </button>
      </div>

      {computing && (
        <div className="bg-yellow-900/30 border border-yellow-800 rounded-lg p-4 flex items-center gap-3">
          <Loader2 size={20} className="animate-spin text-yellow-400" />
          <div>
            <p className="text-yellow-400 font-medium">Computando previsões...</p>
            <p className="text-yellow-600 text-xs">Monte Carlo rodando em background. A tabela atualiza automaticamente.</p>
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-900/30 border border-red-800 rounded-lg p-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {isLoading ? (
        <div className="flex items-center gap-3 justify-center py-20 text-gray-500">
          <Loader2 size={24} className="animate-spin" />
          Carregando...
        </div>
      ) : predictions.length === 0 && !computing ? (
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-12 text-center">
          <p className="text-gray-400 text-lg mb-2">Nenhuma previsão pre-computada</p>
          <p className="text-gray-600 text-sm mb-6">Clique no botão abaixo para rodar o Monte Carlo pela primeira vez.</p>
          <button
            onClick={handleRecompute}
            className="px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg font-medium transition-colors"
          >
            Computar Previsões
          </button>
        </div>
      ) : predictions.length > 0 ? (
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
      ) : null}
    </div>
  );
}
