import { useQuery, useMutation } from '@tanstack/react-query';
import { useState } from 'react';
import { fetchAPI, postAPI } from '../api/client';
import { CheckCircle, XCircle, Clock, RefreshCw, Loader2, Trophy, Target, TrendingUp, ChevronDown, ChevronUp } from 'lucide-react';
import { ProbabilityBar } from '../components/ProbabilityBar';
import { MatchAnalysis } from '../components/MatchAnalysis';

export function TrackRecord() {
  const [roundFilter, setRoundFilter] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [expandedMatches, setExpandedMatches] = useState<Set<number>>(new Set());

  const { data: summary, isLoading: loadingSummary } = useQuery({
    queryKey: ['trackRecord'],
    queryFn: () => fetchAPI<any>('/track-record'),
  });

  // Buscar TODAS pra popular dropdown de rodadas
  const { data: allPreds } = useQuery({
    queryKey: ['trackRecordAll'],
    queryFn: () => fetchAPI<any[]>('/track-record/predictions'),
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
  const rounds = [...new Set((allPreds || []).map((p: any) => p.round).filter(Boolean))].sort((a: number, b: number) => b - a);

  function whoWon(homeGoals: number, awayGoals: number, home: string, away: string) {
    if (homeGoals > awayGoals) return `Vitória ${home}`;
    if (awayGoals > homeGoals) return `Vitória ${away}`;
    return 'Empate';
  }

  function predictedWinner(p: any) {
    const hp = p.home_win_prob || 0;
    const dp = p.draw_prob || 0;
    const ap = p.away_win_prob || 0;
    const max = Math.max(hp, dp, ap);
    if (max === hp) return { text: `Vitória ${p.home_team}`, prob: hp };
    if (max === ap) return { text: `Vitória ${p.away_team}`, prob: ap };
    return { text: 'Empate', prob: dp };
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

      {/* Summary */}
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
              <Target size={16} /> Acertamos quantidade de gols?
            </div>
            <div className={`text-2xl font-bold ${(summary.accuracy_over_under || 0) > 50 ? 'text-green-400' : 'text-yellow-400'}`}>
              {summary.accuracy_over_under?.toFixed(0) || 0}% das vezes
            </div>
            <div className="text-xs text-gray-600 mt-1">
              Se teria 3+ gols ou menos
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

      {/* Predictions */}
      {loadingPreds ? (
        <div className="flex items-center gap-3 justify-center py-12 text-gray-500">
          <Loader2 size={20} className="animate-spin" /> Carregando...
        </div>
      ) : preds.length === 0 ? (
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-12 text-center text-gray-500">
          Nenhuma previsão no histórico.
        </div>
      ) : (
        <div className="space-y-3">
          {preds.map((p: any, i: number) => {
            const isPending = p.status === 'pending';
            const pred = predictedWinner(p);
            const hp = p.home_win_prob || 0;
            const dp = p.draw_prob || 0;
            const ap = p.away_win_prob || 0;
            const score = !isPending ? `${p.actual_home_goals} x ${p.actual_away_goals}` : null;
            const actual = !isPending ? whoWon(p.actual_home_goals, p.actual_away_goals, p.home_team, p.away_team) : null;
            const totalGoals = !isPending ? p.actual_home_goals + p.actual_away_goals : null;

            return (
              <div key={i} className={`bg-gray-900 rounded-lg border p-4 ${
                isPending ? 'border-gray-800' : p.correct_1x2 ? 'border-green-900/60' : 'border-red-900/60'
              }`}>
                {/* Header: times + data + status */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <span className="font-bold text-lg">{p.home_team}</span>
                    <span className="text-gray-600">vs</span>
                    <span className="font-bold text-lg">{p.away_team}</span>
                    {p.commence_time && (
                      <span className="text-xs text-gray-500 bg-gray-800 px-2 py-0.5 rounded">
                        {new Date(p.commence_time).toLocaleDateString('pt-BR', { weekday: 'short', day: '2-digit', month: '2-digit' })}
                      </span>
                    )}
                    {p.round && <span className="text-xs text-gray-600">R{p.round}</span>}
                  </div>
                  {isPending ? (
                    <span className="flex items-center gap-1.5 text-yellow-500 text-sm bg-yellow-900/20 px-3 py-1 rounded-full">
                      <Clock size={14} /> Aguardando resultado
                    </span>
                  ) : (
                    <div className="flex items-center gap-3">
                      <div className="flex items-center gap-1.5">
                        <span className="text-xs text-gray-500">Vencedor</span>
                        {p.correct_1x2 ?
                          <CheckCircle size={20} className="text-green-400" /> :
                          <XCircle size={20} className="text-red-400" />}
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="text-xs text-gray-500">Gols</span>
                        {p.correct_over_under ?
                          <CheckCircle size={20} className="text-green-400" /> :
                          <XCircle size={20} className="text-red-400" />}
                      </div>
                    </div>
                  )}
                </div>

                {/* Probability bar */}
                <ProbabilityBar home={hp} draw={dp} away={ap} />
                <div className="flex justify-between text-xs text-gray-600 mt-1 mb-3">
                  <span>{p.home_team} (casa)</span>
                  <span>Empate</span>
                  <span>{p.away_team} (fora)</span>
                </div>

                {/* Prediction vs Result */}
                <div className="grid grid-cols-2 gap-4">
                  {/* Nossa previsão */}
                  <div className="bg-gray-800/50 rounded-lg p-3">
                    <div className="text-xs text-gray-500 mb-1">Nossa previsão</div>
                    <div className="text-cyan-400 font-medium">{pred.text} ({(pred.prob * 100).toFixed(0)}%)</div>
                    <div className="text-xs text-gray-500 mt-1">
                      Placar previsto: <span className="text-gray-300 font-mono">{p.most_likely_score || '—'}</span>
                    </div>
                    <div className="text-xs text-gray-500">
                      xG: {p.home_xg_expected?.toFixed(2) || '?'} - {p.away_xg_expected?.toFixed(2) || '?'}
                      {' • '}
                      {(p.over_25_prob || 0) > 0.5 ? `Muitos gols (${((p.over_25_prob||0)*100).toFixed(0)}%)` : `Poucos gols (${((1-(p.over_25_prob||0))*100).toFixed(0)}%)`}
                    </div>
                  </div>

                  {/* Resultado real */}
                  <div className={`rounded-lg p-3 ${isPending ? 'bg-gray-800/30' : 'bg-gray-800/50'}`}>
                    <div className="text-xs text-gray-500 mb-1">O que aconteceu</div>
                    {isPending ? (
                      <div className="text-gray-600 italic">Jogo ainda não aconteceu</div>
                    ) : (
                      <>
                        <div className="font-medium">
                          <span className={p.correct_1x2 ? 'text-green-400' : 'text-red-400'}>{actual}</span>
                        </div>
                        <div className="text-2xl font-mono font-bold mt-1">{score}</div>
                        <div className="text-xs text-gray-500 mt-1">
                          {totalGoals !== null && (totalGoals >= 3 ? `${totalGoals} gols (muitos)` : `${totalGoals} gol${totalGoals !== 1 ? 's' : ''} (poucos)`)}
                        </div>
                      </>
                    )}
                  </div>
                </div>

                {/* Bets associadas */}
                {p.bets && p.bets.length > 0 && (
                  <div className={`mt-3 rounded-lg p-3 ${isPending ? 'bg-gray-800/30' : 'bg-gray-800/50'}`}>
                    <div className="text-xs text-gray-500 font-medium mb-2">
                      {isPending ? 'Bets recomendadas' : 'Resultado das bets'}
                    </div>
                    <div className="space-y-1.5">
                      {p.bets.map((b: any, j: number) => {
                        const label = b.outcome_label ||
                                      (b.outcome === 'Over' ? 'Mais de 2.5 gols' :
                                       b.outcome === 'Under' ? 'Menos de 2.5 gols' :
                                       b.outcome === 'Draw' ? 'Empate' :
                                       b.outcome === 'Home' ? `Vitória ${p.home_team}` :
                                       b.outcome === 'Away' ? `Vitória ${p.away_team}` :
                                       `Vitória ${b.outcome}`);
                        return (
                          <div key={j} className="flex items-center justify-between text-sm">
                            <div className="flex items-center gap-2">
                              {b.won !== null && (
                                b.won ?
                                  <CheckCircle size={14} className="text-green-400" /> :
                                  <XCircle size={14} className="text-red-400" />
                              )}
                              <span className={isPending ? 'text-cyan-400' : b.won ? 'text-green-300' : 'text-red-300'}>
                                {label}
                              </span>
                              <span className="text-gray-600">@ {b.best_odds?.toFixed(2)}</span>
                              <span className="text-green-600 text-xs">+{((b.edge||0)*100).toFixed(0)}%</span>
                            </div>
                            <div className="text-right">
                              {b.profit !== null ? (
                                <span className={`font-medium ${(b.profit||0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                  {(b.profit||0) >= 0 ? '+' : ''}R$ {b.profit?.toFixed(2)}
                                </span>
                              ) : (
                                <span className="text-gray-500">R$ {b.kelly_stake?.toFixed(2)} apostado</span>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Post-match analysis toggle — only for resolved */}
                {!isPending && (
                  <div className="mt-3">
                    <button
                      onClick={() => {
                        const next = new Set(expandedMatches);
                        if (next.has(p.match_key)) next.delete(p.match_key);
                        else next.add(p.match_key);
                        setExpandedMatches(next);
                      }}
                      className="flex items-center gap-2 text-xs text-cyan-400 hover:text-cyan-300"
                    >
                      {expandedMatches.has(p.match_key) ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                      {expandedMatches.has(p.match_key) ? 'Ocultar análise' : 'Ver análise completa do jogo'}
                    </button>
                    {expandedMatches.has(p.match_key) && (
                      <div className="mt-3">
                        <MatchAnalysis homeTeam={p.home_team} awayTeam={p.away_team} />
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
