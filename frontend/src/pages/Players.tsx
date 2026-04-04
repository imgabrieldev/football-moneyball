import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { api } from '../api/client';

export function Players() {
  const [team, setTeam] = useState('');
  const [sortBy, setSortBy] = useState('xg');

  const { data: players, isLoading } = useQuery({
    queryKey: ['players', team],
    queryFn: () => api.players(team || undefined),
  });

  const sorted = [...(players || [])].sort((a, b) => (b[sortBy] || 0) - (a[sortBy] || 0));

  const teams = [...new Set((players || []).map((p: any) => p.team))].sort();

  const columns = [
    { key: 'player_name', label: 'Jogador', align: 'left' },
    { key: 'team', label: 'Time', align: 'left' },
    { key: 'matches', label: 'J', align: 'right' },
    { key: 'minutes', label: 'Min', align: 'right' },
    { key: 'goals', label: 'Gols', align: 'right' },
    { key: 'xg', label: 'xG', align: 'right' },
    { key: 'assists', label: 'Ast', align: 'right' },
    { key: 'xa', label: 'xA', align: 'right' },
    { key: 'shots', label: 'Chut', align: 'right' },
    { key: 'passes', label: 'Pass', align: 'right' },
    { key: 'tackles', label: 'Tckl', align: 'right' },
  ];

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Jogadores</h1>

      <div className="flex gap-4 items-end">
        <div>
          <label className="block text-xs text-gray-500 mb-1">Filtrar por time</label>
          <select
            value={team}
            onChange={(e) => setTeam(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm"
          >
            <option value="">Todos</option>
            {teams.map((t: string) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>
        <div className="text-sm text-gray-500">{sorted.length} jogadores</div>
      </div>

      {isLoading ? (
        <div className="text-gray-500">Carregando jogadores...</div>
      ) : (
        <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-auto">
          <table className="w-full text-sm">
            <thead className="text-gray-500 border-b border-gray-800 sticky top-0 bg-gray-900">
              <tr>
                {columns.map((col) => (
                  <th
                    key={col.key}
                    className={`p-3 cursor-pointer hover:text-gray-300 ${col.align === 'right' ? 'text-right' : 'text-left'} ${sortBy === col.key ? 'text-green-400' : ''}`}
                    onClick={() => setSortBy(col.key)}
                  >
                    {col.label} {sortBy === col.key && '↓'}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {sorted.map((player: any, i: number) => {
                const overperf = (player.goals || 0) - (player.xg || 0);
                return (
                  <tr key={i} className="hover:bg-gray-800/50">
                    <td className="p-3 font-medium">{player.player_name}</td>
                    <td className="p-3 text-gray-400">{player.team}</td>
                    <td className="p-3 text-right">{player.matches}</td>
                    <td className="p-3 text-right">{player.minutes}</td>
                    <td className="p-3 text-right font-medium">{player.goals}</td>
                    <td className="p-3 text-right">
                      {player.xg?.toFixed(1)}
                      {Math.abs(overperf) > 1 && (
                        <span className={`ml-1 text-xs ${overperf > 0 ? 'text-green-400' : 'text-red-400'}`}>
                          ({overperf > 0 ? '+' : ''}{overperf.toFixed(1)})
                        </span>
                      )}
                    </td>
                    <td className="p-3 text-right">{player.assists}</td>
                    <td className="p-3 text-right">{player.xa?.toFixed(1)}</td>
                    <td className="p-3 text-right">{player.shots}</td>
                    <td className="p-3 text-right">{player.passes}</td>
                    <td className="p-3 text-right">{player.tackles}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
