const API_BASE = '/api';

export async function fetchAPI<T>(path: string, params?: Record<string, string>): Promise<T> {
  const url = new URL(`${API_BASE}${path}`, window.location.origin);
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
  }
  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export const api = {
  health: () => fetchAPI<{ status: string; version: string }>('/health'),
  matches: (comp?: string, season?: string) =>
    fetchAPI<any[]>('/matches', { competition: comp || 'Brasileirão Série A', season: season || '2026' }),
  predictions: (comp?: string, season?: string) =>
    fetchAPI<{ predictions: any[]; total: number }>('/predictions', { competition: comp || 'Brasileirão Série A', season: season || '2026' }),
  prediction: (home: string, away: string) =>
    fetchAPI<any>(`/predictions/${encodeURIComponent(home)}/${encodeURIComponent(away)}`),
  valueBets: (bankroll?: number, minEdge?: number) =>
    fetchAPI<{ value_bets: any[]; total_matches: number; matches_with_value: number }>('/value-bets', {
      bankroll: String(bankroll || 1000),
      min_edge: String(minEdge || 0.03),
    }),
  players: (team?: string) => {
    const params: Record<string, string> = {};
    if (team) params.team = team;
    return fetchAPI<any[]>('/players', params);
  },
  backtest: (bankroll?: number) =>
    fetchAPI<any>('/backtest', { bankroll: String(bankroll || 1000) }),
  verify: () => fetchAPI<any>('/verify'),
};
