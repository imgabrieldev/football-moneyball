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
      setError(`Error: ${e.message}`);
      setComputing(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Predictions — Monte Carlo</h1>
          <p className="text-gray-500">{predictions.length} matches • Dixon-Coles + Poisson (10K sims)</p>
        </div>
        <button onClick={handleRecompute} disabled={computing}
          className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-700 disabled:cursor-wait rounded-lg text-sm font-medium transition-colors">
          {computing ? <><Loader2 size={16} className="animate-spin" /> Computing...</> : <><RefreshCw size={16} /> Recompute</>}
        </button>
      </div>

      {computing && (
        <div className="bg-yellow-900/30 border border-yellow-800 rounded-lg p-4 flex items-center gap-3">
          <Loader2 size={20} className="animate-spin text-yellow-400" />
          <div>
            <p className="text-yellow-400 font-medium">Computing predictions...</p>
            <p className="text-yellow-600 text-xs">Monte Carlo running in the background. Updates automatically.</p>
          </div>
        </div>
      )}

      {error && <div className="bg-red-900/30 border border-red-800 rounded-lg p-3 text-sm text-red-400">{error}</div>}

      {isLoading ? (
        <div className="flex items-center gap-3 justify-center py-20 text-gray-500">
          <Loader2 size={24} className="animate-spin" /> Loading...
        </div>
      ) : predictions.length === 0 && !computing ? (
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-12 text-center">
          <p className="text-gray-400 text-lg mb-2">No pre-computed predictions</p>
          <p className="text-gray-600 text-sm mb-6">Click to run Monte Carlo for the first time.</p>
          <button onClick={handleRecompute} className="px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg font-medium">
            Compute Predictions
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          {predictions.map((pred: any, i: number) => {
            const conf = pred.confidence;
            const confColor = conf === 'high' ? 'border-green-800' : conf === 'medium' ? 'border-yellow-900' : 'border-gray-800';
            const confDot = conf === 'high' ? 'text-green-400' : conf === 'medium' ? 'text-yellow-400' : 'text-gray-600';
            return (
              <div key={i} className={`bg-gray-900 rounded-lg border ${confColor} p-4`}>
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-bold text-lg">{pred.home_team}</span>
                      <span className="text-gray-600">vs</span>
                      <span className="font-bold text-lg">{pred.away_team}</span>
                      {pred.commence_time && (
                        <span className="text-xs text-gray-500 ml-2 bg-gray-800 px-2 py-0.5 rounded">
                          {new Date(pred.commence_time).toLocaleDateString('en-US', { weekday: 'short', day: '2-digit', month: '2-digit' })}
                          {' '}
                          {new Date(pred.commence_time).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                        </span>
                      )}
                      {pred.context?.home?.coach?.coach_change_recent && (
                        <span className="text-xs bg-purple-900/40 text-purple-300 border border-purple-800 px-2 py-0.5 rounded" title={`${pred.home_team}: new coach, ${pred.context.home.coach.games_since_change} games ago`}>
                          {pred.home_team} new coach
                        </span>
                      )}
                      {pred.context?.away?.coach?.coach_change_recent && (
                        <span className="text-xs bg-purple-900/40 text-purple-300 border border-purple-800 px-2 py-0.5 rounded" title={`${pred.away_team}: new coach, ${pred.context.away.coach.games_since_change} games ago`}>
                          {pred.away_team} new coach
                        </span>
                      )}
                      {(pred.context?.home?.injuries?.key_players_out >= 2) && (
                        <span className="text-xs bg-red-900/40 text-red-300 border border-red-800 px-2 py-0.5 rounded">
                          {pred.context.home.injuries.key_players_out} starters out ({pred.home_team})
                        </span>
                      )}
                      {(pred.context?.away?.injuries?.key_players_out >= 2) && (
                        <span className="text-xs bg-red-900/40 text-red-300 border border-red-800 px-2 py-0.5 rounded">
                          {pred.context.away.injuries.key_players_out} starters out ({pred.away_team})
                        </span>
                      )}
                      {pred.context?.standing?.both_in_relegation && (
                        <span className="text-xs bg-orange-900/40 text-orange-300 border border-orange-800 px-2 py-0.5 rounded">
                          Both in relegation zone
                        </span>
                      )}
                      {pred.lineup_type === 'confirmed' && (
                        <span className="text-xs bg-green-900/40 text-green-300 border border-green-800 px-2 py-0.5 rounded">
                          Lineup confirmed
                        </span>
                      )}
                      {pred.lineup_type === 'probable-xi' && (
                        <span className="text-xs bg-blue-900/40 text-blue-300 border border-blue-800 px-2 py-0.5 rounded" title="Model uses the 11 most-used players of the last 5 games">
                          Probable lineup
                        </span>
                      )}
                      {pred.lineup_type === 'team' && (
                        <span className="text-xs bg-gray-800 text-gray-400 px-2 py-0.5 rounded" title="Model uses team average, not individual players">
                          Team model
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
                      {conf === 'high' ? '●●● High confidence' : conf === 'medium' ? '●● Medium' : '● Low'}
                    </div>
                  </div>
                </div>

                <ProbabilityBar home={pred.home_win_prob || 0} draw={pred.draw_prob || 0} away={pred.away_win_prob || 0} />
                <div className="flex justify-between text-xs text-gray-600 mt-1">
                  <span>{pred.home_team} (home)</span>
                  <span>Draw</span>
                  <span>{pred.away_team} (away)</span>
                </div>

                {/* Markets — plain language */}
                {pred.markets && (
                  <div className="mt-3 grid grid-cols-3 gap-3 text-xs">
                    {/* How many goals */}
                    <div className="bg-gray-800/50 rounded-lg p-2.5">
                      <div className="text-gray-500 font-medium mb-2">How many goals in the match?</div>
                      {pred.markets.over_under?.map((ou: any) => {
                        const goals = parseFloat(ou.line);
                        const moreProb = ou.over_prob;
                        const lessProb = ou.under_prob;
                        const moreLabel = `${Math.ceil(goals)}+ goals`;
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
                          <span className="text-gray-400">Both teams to score?</span>
                          <span className={(pred.markets.btts?.yes_prob||0) > 0.5 ? 'text-green-400 font-medium' : 'text-gray-400'}>
                            {(pred.markets.btts?.yes_prob||0) > 0.5 ? 'Likely' : 'Unlikely'} ({((pred.markets.btts?.yes_prob||0)*100).toFixed(0)}%)
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Most likely scores */}
                    <div className="bg-gray-800/50 rounded-lg p-2.5">
                      <div className="text-gray-500 font-medium mb-2">Most likely scores</div>
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

                    {/* Handicap explained */}
                    <div className="bg-gray-800/50 rounded-lg p-2.5">
                      <div className="text-gray-500 font-medium mb-2">Advantage needed to win</div>
                      {pred.markets.asian_handicap?.filter((ah: any) => ah.line < 0).map((ah: any) => {
                        const needed = Math.abs(ah.line);
                        const teamName = pred.home_team;
                        return (
                          <div key={ah.line} className="flex justify-between py-0.5">
                            <span className="text-gray-400">
                              {teamName} wins by {needed === 0.5 ? '1+' : needed === 1.5 ? '2+' : '3+'} goal{needed > 0.5 ? 's' : ''}
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
                              {teamName} does not lose{buffer >= 1.5 ? ` (or loses by up to ${Math.floor(buffer)})` : ''}
                            </span>
                            <span className={ah.home_prob > 0.5 ? 'text-green-400 font-medium' : 'text-gray-500'}>{(ah.home_prob*100).toFixed(0)}%</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* v1.2.0: Multi-markets (corners, cards, HT) */}
                {pred.multi_markets && (
                  <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                    {/* Corners */}
                    {pred.multi_markets.corners && (
                      <div className="bg-gray-800/50 rounded-lg p-2.5">
                        <div className="text-gray-500 font-medium mb-2">Corners in the match</div>
                        {pred.multi_markets.corners.slice(0, 4).map((ou: any) => {
                          const dominant = ou.over_prob > ou.under_prob ? 'over' : 'under';
                          return (
                            <div key={ou.line} className="flex justify-between py-0.5">
                              <span className={dominant === 'over' ? 'text-green-400' : 'text-gray-500'}>
                                {dominant === 'over' ? 'Over' : 'Under'} {ou.line}
                              </span>
                              <span className={dominant === 'over' ? 'text-green-400 font-medium' : 'text-gray-400'}>
                                {(Math.max(ou.over_prob, ou.under_prob) * 100).toFixed(0)}%
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    )}

                    {/* Cards */}
                    {pred.multi_markets.cards && (
                      <div className="bg-gray-800/50 rounded-lg p-2.5">
                        <div className="text-gray-500 font-medium mb-2">Cards in the match</div>
                        {pred.multi_markets.cards.slice(0, 4).map((ou: any) => {
                          const dominant = ou.over_prob > ou.under_prob ? 'over' : 'under';
                          return (
                            <div key={ou.line} className="flex justify-between py-0.5">
                              <span className={dominant === 'over' ? 'text-yellow-400' : 'text-gray-500'}>
                                {dominant === 'over' ? 'Over' : 'Under'} {ou.line}
                              </span>
                              <span className={dominant === 'over' ? 'text-yellow-400 font-medium' : 'text-gray-400'}>
                                {(Math.max(ou.over_prob, ou.under_prob) * 100).toFixed(0)}%
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    )}

                    {/* HT Result */}
                    {pred.multi_markets.ht_result && (
                      <div className="bg-gray-800/50 rounded-lg p-2.5">
                        <div className="text-gray-500 font-medium mb-2">Who wins the first half?</div>
                        <div className="flex justify-between py-0.5">
                          <span className="text-gray-400">{pred.home_team}</span>
                          <span className="text-gray-300">{(pred.multi_markets.ht_result.home_prob*100).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between py-0.5">
                          <span className="text-gray-400">HT Draw</span>
                          <span className="text-gray-300">{(pred.multi_markets.ht_result.draw_prob*100).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between py-0.5">
                          <span className="text-gray-400">{pred.away_team}</span>
                          <span className="text-gray-300">{(pred.multi_markets.ht_result.away_prob*100).toFixed(0)}%</span>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* v1.4.0: Player Props */}
                {pred.player_props && (pred.player_props.home?.length > 0 || pred.player_props.away?.length > 0) && (
                  <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                    {/* Home players */}
                    <div className="bg-gray-800/50 rounded-lg p-2.5">
                      <div className="text-gray-500 font-medium mb-2">Who scores for {pred.home_team}?</div>
                      {pred.player_props.home?.slice(0, 4).map((p: any) => (
                        <div key={p.player_id} className="flex justify-between py-0.5">
                          <span className="text-gray-300 truncate mr-2">{p.player_name}</span>
                          <span className={(p.goal_prob || 0) > 0.3 ? 'text-green-400 font-medium' : 'text-gray-400'}>
                            {((p.goal_prob || 0) * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                    </div>
                    {/* Away players */}
                    <div className="bg-gray-800/50 rounded-lg p-2.5">
                      <div className="text-gray-500 font-medium mb-2">Who scores for {pred.away_team}?</div>
                      {pred.player_props.away?.slice(0, 4).map((p: any) => (
                        <div key={p.player_id} className="flex justify-between py-0.5">
                          <span className="text-gray-300 truncate mr-2">{p.player_name}</span>
                          <span className={(p.goal_prob || 0) > 0.3 ? 'text-green-400 font-medium' : 'text-gray-400'}>
                            {((p.goal_prob || 0) * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Recommended bets (Betfair) */}
                {pred.recommended_bets && pred.recommended_bets.length > 0 && (
                  <div className="mt-3 bg-green-900/20 border border-green-900/40 rounded-lg p-3">
                    <div className="text-xs text-green-500 font-medium mb-2">What to bet on this match</div>
                    <div className="space-y-2">
                      {pred.recommended_bets.map((bet: any, j: number) => {
                        const hasEdge = bet.edge !== null && bet.edge !== undefined;
                        return (
                          <div key={j} className="flex items-center justify-between text-sm">
                            <div className="flex items-center gap-2">
                              {hasEdge && <span className="bg-green-600 text-white text-[10px] px-1.5 py-0.5 rounded font-medium">EDGE</span>}
                              <span className={hasEdge ? 'text-green-300 font-medium' : 'text-cyan-300'}>{bet.label}</span>
                              <span className="text-gray-500 text-xs">({((bet.model_prob || 0) * 100).toFixed(0)}% chance)</span>
                            </div>
                            <div className="flex items-center gap-3 text-xs">
                              {hasEdge ? (
                                <>
                                  <span className="text-gray-400">Betfair <span className="text-white font-mono">{bet.odds?.toFixed(2)}</span></span>
                                  <span className="text-green-400 font-medium">+{(bet.edge * 100).toFixed(1)}% edge</span>
                                  <span className="text-white font-medium">R$ {bet.stake?.toFixed(2)}</span>
                                </>
                              ) : (
                                <span className="text-gray-500">Fair odds: <span className="text-gray-300 font-mono">{bet.odds?.toFixed(2)}</span></span>
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
