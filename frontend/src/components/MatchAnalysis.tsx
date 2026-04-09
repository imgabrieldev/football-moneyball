import { useQuery } from '@tanstack/react-query';
import { fetchAPI } from '../api/client';
import { Loader2 } from 'lucide-react';

interface Props {
  matchId?: number;
  homeTeam?: string;
  awayTeam?: string;
}

interface StatBarProps {
  label: string;
  home: number | null | undefined;
  away: number | null | undefined;
  format?: (v: number) => string;
  suffix?: string;
  highlight?: boolean;
}

function StatBar({ label, home, away, format, suffix = '', highlight = false }: StatBarProps) {
  const h = home ?? 0;
  const a = away ?? 0;
  const total = Math.abs(h) + Math.abs(a);
  const homePct = total > 0 ? (Math.abs(h) / total) * 100 : 50;
  const fmt = format || ((v: number) => v.toString());
  const homeColor = h > a ? 'text-green-400' : 'text-gray-400';
  const awayColor = a > h ? 'text-green-400' : 'text-gray-400';
  return (
    <div className={`py-1 ${highlight ? 'bg-blue-900/20 rounded px-2' : ''}`}>
      <div className="flex justify-between items-center text-xs mb-1">
        <span className={`${homeColor} font-medium`}>{fmt(h)}{suffix}</span>
        <span className="text-gray-500 uppercase">{label}</span>
        <span className={`${awayColor} font-medium`}>{fmt(a)}{suffix}</span>
      </div>
      <div className="flex gap-px h-1">
        <div className="bg-cyan-600" style={{ width: `${homePct}%` }} />
        <div className="bg-amber-600" style={{ width: `${100 - homePct}%` }} />
      </div>
    </div>
  );
}

export function MatchAnalysis({ matchId, homeTeam, awayTeam }: Props) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['matchAnalysis', matchId, homeTeam, awayTeam],
    queryFn: () => {
      if (matchId) return fetchAPI<any>(`/match-analysis/${matchId}`);
      return fetchAPI<any>('/match-analysis/by-teams', { home_team: homeTeam!, away_team: awayTeam! });
    },
    enabled: !!(matchId || (homeTeam && awayTeam)),
  });

  if (isLoading) return <div className="flex items-center gap-2 text-gray-500"><Loader2 size={16} className="animate-spin" /> Loading analysis...</div>;
  if (error || !data || data.error) return <div className="text-gray-500 text-sm">{data?.error || 'Error loading'}</div>;

  const { match, real_stats, prediction } = data;
  if (!real_stats) return <div className="text-gray-500 text-sm">No detailed stats available</div>;

  // Compare prediction vs actual
  const predictedWinner = prediction ? (
    prediction.home_win_prob > Math.max(prediction.draw_prob, prediction.away_win_prob) ? 'home' :
    prediction.away_win_prob > prediction.draw_prob ? 'away' : 'draw'
  ) : null;
  const actualWinner = match.home_score > match.away_score ? 'home' : match.home_score < match.away_score ? 'away' : 'draw';

  return (
    <div className="bg-gray-950 border border-gray-800 rounded-lg p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-gray-800 pb-2">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-bold text-cyan-400">{match.home_team}</span>
            <span className="font-mono text-2xl">{match.home_score} - {match.away_score}</span>
            <span className="font-bold text-amber-400">{match.away_team}</span>
          </div>
          <div className="text-xs text-gray-500 mt-1">HT {real_stats.ht_score.home} - {real_stats.ht_score.away} • {real_stats.referee || 'referee N/A'}</div>
        </div>
        {prediction && (
          <div className="text-right text-xs">
            <div className="text-gray-500 mb-0.5">Model correct?</div>
            <div className={prediction.correct_1x2 ? 'text-green-400 font-bold' : 'text-red-400 font-bold'}>
              {prediction.correct_1x2 === null ? '—' : prediction.correct_1x2 ? '✓ YES' : '✗ NO'}
            </div>
            <div className="text-gray-600 mt-0.5">Brier {prediction.brier_score?.toFixed(3) || '—'}</div>
          </div>
        )}
      </div>

      {/* Prediction vs Real — 1X2 */}
      {prediction && (
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div className="bg-gray-800/50 rounded p-2">
            <div className="text-gray-500 mb-1">Model prediction</div>
            <div className="flex justify-between">
              <span className={predictedWinner === 'home' ? 'text-green-400' : 'text-gray-400'}>
                {match.home_team}: {(prediction.home_win_prob * 100).toFixed(0)}%
              </span>
              <span className={predictedWinner === 'draw' ? 'text-green-400' : 'text-gray-400'}>
                Draw: {(prediction.draw_prob * 100).toFixed(0)}%
              </span>
              <span className={predictedWinner === 'away' ? 'text-green-400' : 'text-gray-400'}>
                {match.away_team}: {(prediction.away_win_prob * 100).toFixed(0)}%
              </span>
            </div>
            <div className="mt-1 text-gray-500">
              Expected score: <span className="font-mono text-gray-300">{prediction.most_likely_score || '—'}</span>
              {' • '}xG {prediction.home_xg_expected?.toFixed(2)} - {prediction.away_xg_expected?.toFixed(2)}
            </div>
          </div>
          <div className="bg-gray-800/50 rounded p-2">
            <div className="text-gray-500 mb-1">What happened</div>
            <div className={`font-medium ${actualWinner === predictedWinner ? 'text-green-400' : 'text-red-400'}`}>
              {actualWinner === 'home' ? `${match.home_team} win` : actualWinner === 'away' ? `${match.away_team} win` : 'Draw'}
            </div>
            <div className="text-gray-500 mt-1">
              Actual score: <span className="font-mono text-gray-300">{match.home_score}-{match.away_score}</span>
              {' • '}Actual xG {real_stats.xg.home?.toFixed(2)} - {real_stats.xg.away?.toFixed(2)}
            </div>
          </div>
        </div>
      )}

      {/* Stats bars */}
      <div className="space-y-1">
        <StatBar label="Possession" home={real_stats.possession.home} away={real_stats.possession.away} suffix="%" format={(v) => v.toFixed(0)} highlight />
        <StatBar label="xG (actual)" home={real_stats.xg.home} away={real_stats.xg.away} format={(v) => v.toFixed(2)} highlight />
        <StatBar label="Shots" home={real_stats.shots.home} away={real_stats.shots.away} />
        <StatBar label="Shots on target" home={real_stats.sot.home} away={real_stats.sot.away} highlight />
        <StatBar label="Big chances" home={real_stats.big_chances.home} away={real_stats.big_chances.away} />
        <StatBar label="Goals prevented" home={real_stats.goals_prevented.home} away={real_stats.goals_prevented.away} format={(v) => v.toFixed(2)} highlight />
        <StatBar label="Corners" home={real_stats.corners.home} away={real_stats.corners.away} />
        <StatBar label="Cards" home={real_stats.cards.home} away={real_stats.cards.away} />
        <StatBar label="Fouls" home={real_stats.fouls.home} away={real_stats.fouls.away} />
        <StatBar label="Touches in box" home={real_stats.touches_box.home} away={real_stats.touches_box.away} />
        <StatBar label="Passes" home={real_stats.passes.home} away={real_stats.passes.away} />
        <StatBar label="Pass accuracy" home={real_stats.pass_accuracy.home} away={real_stats.pass_accuracy.away} suffix="%" format={(v) => v.toFixed(0)} />
        <StatBar label="Dispossessed" home={real_stats.dispossessed.home} away={real_stats.dispossessed.away} />
      </div>

      {/* Insights */}
      <div className="bg-gray-800/30 border border-gray-800 rounded p-2 text-xs text-gray-400">
        <div className="text-gray-500 font-medium mb-1">Quick analysis</div>
        {real_stats.possession.home < 45 && match.home_score > match.away_score && (
          <div><strong className="text-cyan-400">{match.home_team}</strong> won with only {real_stats.possession.home.toFixed(0)}% possession — <span className="text-gray-300">efficient counter-attack</span></div>
        )}
        {real_stats.goals_prevented.home > 0.5 && (
          <div>{match.home_team} goalkeeper saved {real_stats.goals_prevented.home.toFixed(1)} goal(s) above expected</div>
        )}
        {real_stats.goals_prevented.away > 0.5 && (
          <div>{match.away_team} goalkeeper saved {real_stats.goals_prevented.away.toFixed(1)} goal(s) above expected</div>
        )}
        {(real_stats.xg.home || 0) > match.home_score + 0.5 && (
          <div><strong className="text-red-400">{match.home_team}</strong> wasted chances: xG {real_stats.xg.home?.toFixed(1)} but only {match.home_score} goal(s)</div>
        )}
        {(real_stats.xg.away || 0) > match.away_score + 0.5 && (
          <div><strong className="text-red-400">{match.away_team}</strong> wasted chances: xG {real_stats.xg.away?.toFixed(1)} but only {match.away_score} goal(s)</div>
        )}
      </div>
    </div>
  );
}
