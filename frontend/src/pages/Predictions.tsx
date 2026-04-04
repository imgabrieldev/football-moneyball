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
    refetchInterval: computing ? 5_000 : false,
  });

  const predictions = data?.predictions || [];

  useEffect(() => {
    if (computing && predictions.length > 0) setComputing(false);
  }, [predictions.length, computing]);

  async function handleRecompute() {
    setComputing(true);
    setError('');
    try {
      await fetch('/api/predictions/recompute', { method: 'POST' });
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
          <p className="text-gray-500">{predictions.length} partidas • Dixon-Coles + Poisson (10K sims)</p>
        </div>
        <button onClick={handleRecompute} disabled={computing}
          className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-700 disabled:cursor-wait rounded-lg text-sm font-medium transition-colors">
          {computing ? <><Loader2 size={16} className="animate-spin" /> Computando...</> : <><RefreshCw size={16} /> Recomputar</>}
        </button>
      </div>

      {computing && (
        <div className="bg-yellow-900/30 border border-yellow-800 rounded-lg p-4 flex items-center gap-3">
          <Loader2 size={20} className="animate-spin text-yellow-400" />
          <div>
            <p className="text-yellow-400 font-medium">Computando previsões...</p>
            <p className="text-yellow-600 text-xs">Monte Carlo em background. Atualiza automaticamente.</p>
          </div>
        </div>
      )}

      {error && <div className="bg-red-900/30 border border-red-800 rounded-lg p-3 text-sm text-red-400">{error}</div>}

      {isLoading ? (
        <div className="flex items-center gap-3 justify-center py-20 text-gray-500">
          <Loader2 size={24} className="animate-spin" /> Carregando...
        </div>
      ) : predictions.length === 0 && !computing ? (
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-12 text-center">
          <p className="text-gray-400 text-lg mb-2">Nenhuma previsão pre-computada</p>
          <p className="text-gray-600 text-sm mb-6">Clique para rodar Monte Carlo pela primeira vez.</p>
          <button onClick={handleRecompute} className="px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg font-medium">
            Computar Previsões
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          {predictions.map((pred: any, i: number) => {
            const conf = pred.confidence;
            const confColor = conf === 'alta' ? 'border-green-800' : conf === 'media' ? 'border-yellow-900' : 'border-gray-800';
            const confDot = conf === 'alta' ? 'text-green-400' : conf === 'media' ? 'text-yellow-400' : 'text-gray-600';
            return (
              <div key={i} className={`bg-gray-900 rounded-lg border ${confColor} p-4`}>
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-lg">{pred.home_team}</span>
                      <span className="text-gray-600">vs</span>
                      <span className="font-bold text-lg">{pred.away_team}</span>
                      {pred.commence_time && (
                        <span className="text-xs text-gray-500 ml-2 bg-gray-800 px-2 py-0.5 rounded">
                          {new Date(pred.commence_time).toLocaleDateString('pt-BR', { weekday: 'short', day: '2-digit', month: '2-digit' })}
                          {' '}
                          {new Date(pred.commence_time).toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' })}
                        </span>
                      )}
                    </div>
                    {pred.interpretation && (
                      <p className="text-cyan-400 text-sm mt-1">{pred.interpretation}</p>
                    )}
                    {pred.goals_hint && (
                      <p className="text-yellow-600 text-xs mt-0.5">{pred.goals_hint}</p>
                    )}
                  </div>
                  <div className="text-right">
                    <div className="font-mono text-2xl">{pred.most_likely_score}</div>
                    <div className={`text-xs ${confDot}`}>
                      {conf === 'alta' ? '●●● Alta confiança' : conf === 'media' ? '●● Média' : '● Baixa'}
                    </div>
                  </div>
                </div>

                <ProbabilityBar home={pred.home_win_prob || 0} draw={pred.draw_prob || 0} away={pred.away_win_prob || 0} />
                <div className="flex justify-between text-xs text-gray-600 mt-1">
                  <span>{pred.home_team} (casa)</span>
                  <span>Empate</span>
                  <span>{pred.away_team} (fora)</span>
                </div>

                {/* Markets — linguagem humana */}
                {pred.markets && (
                  <div className="mt-3 grid grid-cols-3 gap-3 text-xs">
                    {/* Quantos gols */}
                    <div className="bg-gray-800/50 rounded-lg p-2.5">
                      <div className="text-gray-500 font-medium mb-2">Quantos gols no jogo?</div>
                      {pred.markets.over_under?.map((ou: any) => {
                        const goals = parseFloat(ou.line);
                        const moreProb = ou.over_prob;
                        const lessProb = ou.under_prob;
                        const moreLabel = `${Math.ceil(goals)}+ gols`;
                        const winner = moreProb > lessProb ? 'more' : 'less';
                        return (
                          <div key={ou.line} className="flex justify-between py-0.5">
                            <span className={winner === 'more' ? 'text-green-400' : 'text-gray-500'}>{moreLabel}</span>
                            <span className={winner === 'more' ? 'text-green-400 font-medium' : 'text-gray-500'}>{(moreProb*100).toFixed(0)}%</span>
                          </div>
                        );
                      })}
                      <div className="border-t border-gray-700 mt-1.5 pt-1.5">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Os dois times marcam?</span>
                          <span className={(pred.markets.btts?.yes_prob||0) > 0.5 ? 'text-green-400 font-medium' : 'text-gray-400'}>
                            {(pred.markets.btts?.yes_prob||0) > 0.5 ? 'Provável' : 'Improvável'} ({((pred.markets.btts?.yes_prob||0)*100).toFixed(0)}%)
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Placares mais prováveis */}
                    <div className="bg-gray-800/50 rounded-lg p-2.5">
                      <div className="text-gray-500 font-medium mb-2">Placares mais prováveis</div>
                      {pred.markets.correct_score?.slice(0, 6).map((cs: any, idx: number) => (
                        <div key={cs.score} className="flex justify-between py-0.5">
                          <span className={`font-mono ${idx === 0 ? 'text-green-400 font-medium' : 'text-gray-300'}`}>{cs.score}</span>
                          <div className="flex items-center gap-2">
                            <div className="w-16 bg-gray-700 rounded-full h-1.5">
                              <div className="bg-green-500 h-1.5 rounded-full" style={{width: `${Math.min(cs.prob * 500, 100)}%`}} />
                            </div>
                            <span className="text-gray-400 w-10 text-right">{(cs.prob*100).toFixed(1)}%</span>
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Handicap explicado */}
                    <div className="bg-gray-800/50 rounded-lg p-2.5">
                      <div className="text-gray-500 font-medium mb-2">Vantagem necessária pra ganhar</div>
                      {pred.markets.asian_handicap?.filter((ah: any) => ah.line < 0).map((ah: any) => {
                        const needed = Math.abs(ah.line);
                        const teamName = pred.home_team;
                        return (
                          <div key={ah.line} className="flex justify-between py-0.5">
                            <span className="text-gray-400">
                              {teamName} ganha por {needed === 0.5 ? '1+' : needed === 1.5 ? '2+' : '3+'} gol{needed > 0.5 ? 's' : ''}
                            </span>
                            <span className={ah.home_prob > 0.5 ? 'text-green-400 font-medium' : 'text-gray-500'}>{(ah.home_prob*100).toFixed(0)}%</span>
                          </div>
                        );
                      })}
                      {pred.markets.asian_handicap?.filter((ah: any) => ah.line > 0).slice(0, 2).map((ah: any) => {
                        const buffer = ah.line;
                        const teamName = pred.home_team;
                        return (
                          <div key={ah.line} className="flex justify-between py-0.5">
                            <span className="text-gray-400">
                              {teamName} não perde{buffer >= 1.5 ? ` (ou perde por até ${Math.floor(buffer)})` : ''}
                            </span>
                            <span className={ah.home_prob > 0.5 ? 'text-green-400 font-medium' : 'text-gray-500'}>{(ah.home_prob*100).toFixed(0)}%</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Bets recomendadas (Betfair) */}
                {pred.recommended_bets && pred.recommended_bets.length > 0 && (
                  <div className="mt-3 bg-green-900/20 border border-green-900/40 rounded-lg p-3">
                    <div className="text-xs text-green-500 font-medium mb-2">O que apostar nesse jogo</div>
                    <div className="space-y-2">
                      {pred.recommended_bets.map((bet: any, j: number) => {
                        const hasEdge = bet.edge !== null && bet.edge !== undefined;
                        return (
                          <div key={j} className="flex items-center justify-between text-sm">
                            <div className="flex items-center gap-2">
                              {hasEdge && <span className="bg-green-600 text-white text-[10px] px-1.5 py-0.5 rounded font-medium">EDGE</span>}
                              <span className={hasEdge ? 'text-green-300 font-medium' : 'text-cyan-300'}>{bet.label}</span>
                              <span className="text-gray-500 text-xs">({((bet.model_prob || 0) * 100).toFixed(0)}% de chance)</span>
                            </div>
                            <div className="flex items-center gap-3 text-xs">
                              {hasEdge ? (
                                <>
                                  <span className="text-gray-400">Betfair <span className="text-white font-mono">{bet.odds?.toFixed(2)}</span></span>
                                  <span className="text-green-400 font-medium">+{(bet.edge * 100).toFixed(1)}% edge</span>
                                  <span className="text-white font-medium">R$ {bet.stake?.toFixed(2)}</span>
                                </>
                              ) : (
                                <span className="text-gray-500">Odds justo: <span className="text-gray-300 font-mono">{bet.odds?.toFixed(2)}</span></span>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
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
