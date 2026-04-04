export function ProbabilityBar({ home, draw, away }: { home: number; draw: number; away: number }) {
  return (
    <div className="flex h-6 rounded overflow-hidden text-xs font-medium">
      <div
        className="bg-green-600 flex items-center justify-center text-white"
        style={{ width: `${home * 100}%` }}
      >
        {home > 0.15 && `${(home * 100).toFixed(0)}%`}
      </div>
      <div
        className="bg-gray-600 flex items-center justify-center text-white"
        style={{ width: `${draw * 100}%` }}
      >
        {draw > 0.15 && `${(draw * 100).toFixed(0)}%`}
      </div>
      <div
        className="bg-red-600 flex items-center justify-center text-white"
        style={{ width: `${away * 100}%` }}
      >
        {away > 0.15 && `${(away * 100).toFixed(0)}%`}
      </div>
    </div>
  );
}
