import { useQuery, useMutation } from '@tanstack/react-query';
import { useState } from 'react';
import { fetchAPI, postAPI } from '../api/client';
import { CheckCircle, XCircle, Clock, RefreshCw, Loader2, Trophy, Target, TrendingUp } from 'lucide-react';

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

  const resolve = useMutation({
    mutationFn: () => postAPI<any>('/resolve'),
    onSuccess: () => { refetch(); },
  });

  const preds = predictions || [];
  const rounds = [...new Set(preds.map((p: any) => p.round).filter(Boolean))].sort((a: number, b: number) => b - a);

  // Readable outcome
  function readableOutcome(p: any) {
    if (!p.actual_outcome) return null;
    const home = p.home_team || '?';
    const away = p.away_team || '?';
    if (p.actual_outcome === 'Home') return `Vitória ${home}`;
    if (p.actual_outcome === 'Away') return `Vitória ${away}`;
    return 'Empate';
  }

  function readablePrediction(p: any) {
    const home = p.home_team || '?';
    const away = p.away_team || '?';
    const hp = p.home_win_prob || 0;
    const dp = p.draw_prob || 0;
    const ap = p.away_win_prob || 0;
    const max = Math.max(hp, dp, ap);
    if (max === hp) return `Vitória ${home} (${(hp*100).toFixed(0)}%)`;
    if (max === ap) return `Vitória ${away} (${(ap*100).toFixed(0)}%)`;
    return `Empate (${(dp*100).toFixed(0)}%)`;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Track Record</h1>
          <p className="text-gray-500">Como nosso modelo se saiu em cada jogo</p>
        </div>
        <button onClick={() => resolve.mutate()} disabled={resolve.isPending}
          className="flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-700 disabled:bg-gray-700 rounded-lg text-sm font-medium">
          {resolve.isPending ? <><Loader2 size={16} className="animate-spin" /> Atualizando...</> : <><RefreshCw size={16} /> Atualizar Resultados</>}
        </button>
      </div>

      {/* Summary — linguagem humana */}
      {!loadingSummary && summary && summary.resolved > 0 && (
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
            <div className="flex items-center gap-2 text-gray-400 text-sm mb-1">
              <Trophy size={16} /> Acertamos o vencedor?
            </div>
            <div className={`text-2xl font-bold ${(summary.accuracy_1x2 || 0) > 40 ? 'text-green-400' : 'text-red-400'}`}>
              {summary.accuracy_1x2?.toFixed(0) || 0}% das vezes
            </div>
            <div className="text-xs text-gray-600 mt-1">
              {Math.round((summary.accuracy_1x2 || 0) / 100 * (summary.resolved || 0))} de {summary.resolved} jogos
            </div>
          </div>

          <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
            <div className="flex items-center gap-2 text-gray-400 text-sm mb-1">
              <Target size={16} /> Acertamos se teria muitos gols?
            </div>
            <div className={`text-2xl font-bold ${(summary.accuracy_over_under || 0) > 50 ? 'text-green-400' : 'text-yellow-400'}`}>
              {summary.accuracy_over_under?.toFixed(0) || 0}% das vezes
            </div>
            <div className="text-xs text-gray-600 mt-1">
              Mais ou menos de 3 gols no jogo
            </div>
          </div>

          <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
            <div className="flex items-center gap-2 text-gray-400 text-sm mb-1">
              <TrendingUp size={16} /> Qualidade das probabilidades
            </div>
            <div className={`text-2xl font-bold ${(summary.avg_brier || 1) < 0.25 ? 'text-green-400' : 'text-yellow-400'}`}>
              {(summary.avg_brier || 0) < 0.25 ? 'Bom' : (summary.avg_brier || 0) < 0.50 ? 'Regular' : 'Ruim'}
            </div>
            <div className="text-xs text-gray-600 mt-1">
              Score: {summary.avg_brier?.toFixed(3)} (quanto menor, melhor)
            </div>
          </div>

          <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
            <div className="flex items-center gap-2 text-gray-400 text-sm mb-1">
              <Clock size={16} /> Status
            </div>
            <div className="text-2xl font-bold">{summary.total || 0}</div>
            <div className="text-xs text-gray-600 mt-1">
              {summary.resolved || 0} com resultado • {summary.pending || 0} aguardando
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
            <option value="pending">Aguardando resultado</option>
            <option value="resolved">Com resultado</option>
          </select>
        </div>
        <div className="text-sm text-gray-500">{preds.length} previsões</div>
      </div>

      {/* Predictions table */}
      {loadingPreds ? (
        <div className="flex items-center gap-3 justify-center py-12 text-gray-500">
          <Loader2 size={20} className="animate-spin" /> Carregando...
        </div>
      ) : preds.length === 0 ? (
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-12 text-center text-gray-500">
          Nenhuma previsão no histórico.
        </div>
      ) : (
        <div className="space-y-2">
          {preds.map((p: any, i: number) => {
            const isPending = p.status === 'pending';
            const prediction = readablePrediction(p);
            const actual = readableOutcome(p);
            const score = isPending ? null : `${p.actual_home_goals}x${p.actual_away_goals}`;

            return (
              <div key={i} className={`bg-gray-900 rounded-lg border ${isPending ? 'border-gray-800' : p.correct_1x2 ? 'border-green-900' : 'border-red-900'} p-4`}>
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3">
                      <span className="font-bold">{p.home_team}</span>
                      <span className="text-gray-600">vs</span>
                      <span className="font-bold">{p.away_team}</span>
                      {p.commence_time && (
                        <span className="text-xs text-gray-500 bg-gray-800 px-2 py-0.5 rounded">
                          {new Date(p.commence_time).toLocaleDateString('pt-BR', { day: '2-digit', month: '2-digit' })}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-4 mt-2 text-sm">
                      <div>
                        <span className="text-gray-500">Previmos: </span>
                        <span className="text-cyan-400">{prediction}</span>
                      </div>
                      {!isPending && (
                        <div>
                          <span className="text-gray-500">Aconteceu: </span>
                          <span className="text-white font-medium">{actual} ({score})</span>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-4">
                    {isPending ? (
                      <div className="flex items-center gap-2 text-yellow-500 text-sm">
                        <Clock size={18} />
                        <span>Aguardando</span>
                      </div>
                    ) : (
                      <>
                        <div className="text-center">
                          <div className="text-xs text-gray-500">Vencedor</div>
                          {p.correct_1x2 ?
                            <CheckCircle size={22} className="text-green-400 mx-auto" /> :
                            <XCircle size={22} className="text-red-400 mx-auto" />}
                        </div>
                        <div className="text-center">
                          <div className="text-xs text-gray-500">Gols</div>
                          {p.correct_over_under ?
                            <CheckCircle size={22} className="text-green-400 mx-auto" /> :
                            <XCircle size={22} className="text-red-400 mx-auto" />}
                        </div>
                      </>
                    )}
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
