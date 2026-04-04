import { useQuery, useMutation } from '@tanstack/react-query';
import { useState } from 'react';
import { fetchAPI, postAPI } from '../api/client';
import { CheckCircle, XCircle, Clock, RefreshCw, Loader2 } from 'lucide-react';

export function TrackRecord() {
  const [roundFilter, setRoundFilter] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<string>('');

  const { data: summary, isLoading: loadingSummary } = useQuery({
    queryKey: ['trackRecord'],
    queryFn: () => fetchAPI<any>('/track-record'),
  });

  const params: Record<string, string> = {};
  if (roundFilter) params.round = roundFilter;
  if (statusFilter) params.status = statusFilter;

  const { data: predictions, isLoading: loadingPreds, refetch } = useQuery({
    queryKey: ['trackRecordPreds', roundFilter, statusFilter],
    queryFn: () => fetchAPI<any[]>('/track-record/predictions', params),
  });

  const { data: valueBets } = useQuery({
    queryKey: ['trackRecordBets'],
    queryFn: () => fetchAPI<any[]>('/track-record/value-bets'),
  });

  const resolve = useMutation({
    mutationFn: () => postAPI<any>('/resolve'),
    onSuccess: () => { refetch(); },
  });

  const preds = predictions || [];
  const bets = valueBets || [];
  const rounds = [...new Set(preds.map((p: any) => p.round).filter(Boolean))].sort((a: number, b: number) => b - a);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Track Record</h1>
          <p className="text-gray-500">Histórico de previsões vs resultados reais</p>
        </div>
        <button onClick={() => resolve.mutate()} disabled={resolve.isPending}
          className="flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-700 disabled:bg-gray-700 rounded-lg text-sm font-medium">
          {resolve.isPending ? <><Loader2 size={16} className="animate-spin" /> Resolvendo...</> : <><RefreshCw size={16} /> Resolver Pendentes</>}
        </button>
      </div>

      {/* Summary cards */}
      {!loadingSummary && summary && (
        <div className="grid grid-cols-5 gap-3">
          <div className="bg-gray-900 rounded-lg p-3 border border-gray-800">
            <div className="text-xs text-gray-500">Total</div>
            <div className="text-xl font-bold">{summary.total || 0}</div>
            <div className="text-xs text-gray-600">{summary.resolved || 0} resolvidas • {summary.pending || 0} pendentes</div>
          </div>
          <div className="bg-gray-900 rounded-lg p-3 border border-gray-800">
            <div className="text-xs text-gray-500">Accuracy 1X2</div>
            <div className={`text-xl font-bold ${(summary.accuracy_1x2 || 0) > 40 ? 'text-green-400' : 'text-red-400'}`}>
              {summary.accuracy_1x2?.toFixed(1) || 0}%
            </div>
          </div>
          <div className="bg-gray-900 rounded-lg p-3 border border-gray-800">
            <div className="text-xs text-gray-500">Accuracy O/U</div>
            <div className={`text-xl font-bold ${(summary.accuracy_over_under || 0) > 50 ? 'text-green-400' : 'text-yellow-400'}`}>
              {summary.accuracy_over_under?.toFixed(1) || 0}%
            </div>
          </div>
          <div className="bg-gray-900 rounded-lg p-3 border border-gray-800">
            <div className="text-xs text-gray-500">Brier Score</div>
            <div className={`text-xl font-bold ${(summary.avg_brier || 1) < 0.25 ? 'text-green-400' : 'text-yellow-400'}`}>
              {summary.avg_brier?.toFixed(4) || '—'}
            </div>
          </div>
          <div className="bg-gray-900 rounded-lg p-3 border border-gray-800">
            <div className="text-xs text-gray-500">Value Bets P/L</div>
            <div className={`text-xl font-bold ${(bets.reduce((s: number, b: any) => s + (b.profit || 0), 0)) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              R$ {bets.reduce((s: number, b: any) => s + (b.profit || 0), 0).toFixed(2)}
            </div>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="flex gap-4 items-end">
        <div>
          <label className="block text-xs text-gray-500 mb-1">Rodada</label>
          <select value={roundFilter} onChange={(e) => setRoundFilter(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm">
            <option value="">Todas</option>
            {rounds.map((r: number) => <option key={r} value={r}>Rodada {r}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">Status</label>
          <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm">
            <option value="">Todos</option>
            <option value="pending">Pendentes</option>
            <option value="resolved">Resolvidos</option>
          </select>
        </div>
        <div className="text-sm text-gray-500">{preds.length} previsões</div>
      </div>

      {/* Predictions table */}
      {loadingPreds ? (
        <div className="flex items-center gap-3 justify-center py-12 text-gray-500">
          <Loader2 size={20} className="animate-spin" /> Carregando...
        </div>
      ) : (
        <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="text-gray-500 border-b border-gray-800">
              <tr>
                <th className="p-3 text-left">Partida</th>
                <th className="p-3 text-center">Data</th>
                <th className="p-3 text-center">Previsão</th>
                <th className="p-3 text-center">Placar Real</th>
                <th className="p-3 text-center">1X2</th>
                <th className="p-3 text-center">O/U</th>
                <th className="p-3 text-right">Brier</th>
                <th className="p-3 text-center">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {preds.map((p: any, i: number) => {
                const isPending = p.status === 'pending';
                return (
                  <tr key={i} className="hover:bg-gray-800/50">
                    <td className="p-3">
                      <div className="font-medium">{p.home_team} vs {p.away_team}</div>
                      <div className="text-xs text-gray-500">
                        H:{(p.home_win_prob * 100).toFixed(0)}% D:{(p.draw_prob * 100).toFixed(0)}% A:{(p.away_win_prob * 100).toFixed(0)}%
                      </div>
                    </td>
                    <td className="p-3 text-center text-xs text-gray-400">
                      {p.commence_time ? new Date(p.commence_time).toLocaleDateString('pt-BR', { day: '2-digit', month: '2-digit' }) : '—'}
                    </td>
                    <td className="p-3 text-center font-mono">{p.most_likely_score || '—'}</td>
                    <td className="p-3 text-center font-mono">
                      {isPending ? <span className="text-gray-600">—</span> : `${p.actual_home_goals}x${p.actual_away_goals}`}
                    </td>
                    <td className="p-3 text-center">
                      {isPending ? <Clock size={16} className="text-gray-600 inline" /> :
                        p.correct_1x2 ? <CheckCircle size={16} className="text-green-400 inline" /> :
                        <XCircle size={16} className="text-red-400 inline" />}
                    </td>
                    <td className="p-3 text-center">
                      {isPending ? <Clock size={16} className="text-gray-600 inline" /> :
                        p.correct_over_under ? <CheckCircle size={16} className="text-green-400 inline" /> :
                        <XCircle size={16} className="text-red-400 inline" />}
                    </td>
                    <td className="p-3 text-right">{p.brier_score?.toFixed(3) || '—'}</td>
                    <td className="p-3 text-center">
                      {isPending ?
                        <span className="bg-yellow-900/30 text-yellow-400 text-xs px-2 py-0.5 rounded">Pendente</span> :
                        <span className="bg-green-900/30 text-green-400 text-xs px-2 py-0.5 rounded">Resolvido</span>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {preds.length === 0 && (
            <div className="p-12 text-center text-gray-500">
              Nenhuma previsão no histórico. Compute previsões primeiro.
            </div>
          )}
        </div>
      )}

      {/* Value Bets P/L */}
      {bets.length > 0 && (
        <div className="bg-gray-900 rounded-lg border border-gray-800">
          <div className="p-4 border-b border-gray-800">
            <h2 className="text-lg font-semibold">Value Bets — Profit/Loss</h2>
          </div>
          <table className="w-full text-sm">
            <thead className="text-gray-500 border-b border-gray-800">
              <tr>
                <th className="p-3 text-left">Partida</th>
                <th className="p-3 text-left">Aposta</th>
                <th className="p-3 text-right">Odds</th>
                <th className="p-3 text-right">Edge</th>
                <th className="p-3 text-right">Stake</th>
                <th className="p-3 text-center">Resultado</th>
                <th className="p-3 text-right">P/L</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {bets.slice(0, 20).map((b: any, i: number) => (
                <tr key={i} className="hover:bg-gray-800/50">
                  <td className="p-3">{b.home_team} vs {b.away_team}</td>
                  <td className="p-3 text-gray-400">{b.market} — {b.outcome}</td>
                  <td className="p-3 text-right font-mono">{b.best_odds?.toFixed(2)}</td>
                  <td className="p-3 text-right text-green-400">+{((b.edge || 0) * 100).toFixed(1)}%</td>
                  <td className="p-3 text-right">R$ {b.kelly_stake?.toFixed(2)}</td>
                  <td className="p-3 text-center">
                    {b.won === null ? <Clock size={16} className="text-gray-600 inline" /> :
                      b.won ? <CheckCircle size={16} className="text-green-400 inline" /> :
                      <XCircle size={16} className="text-red-400 inline" />}
                  </td>
                  <td className={`p-3 text-right font-medium ${(b.profit || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {b.profit !== null ? `R$ ${b.profit?.toFixed(2)}` : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
