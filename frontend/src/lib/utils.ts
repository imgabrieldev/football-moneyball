export function formatPct(value: number): string {
  return `${(value * 100).toFixed(0)}%`;
}

export function formatOdds(value: number): string {
  return value.toFixed(2);
}

export function formatMoney(value: number): string {
  return `R$ ${value.toFixed(2)}`;
}

export function edgeColor(edge: number): string {
  if (edge >= 0.10) return 'text-green-400';
  if (edge >= 0.05) return 'text-green-300';
  if (edge >= 0.03) return 'text-yellow-300';
  return 'text-gray-400';
}

export function resultColor(won: boolean): string {
  return won ? 'text-green-400' : 'text-red-400';
}

export function brierColor(brier: number): string {
  if (brier < 0.20) return 'text-green-400';
  if (brier < 0.25) return 'text-yellow-300';
  return 'text-red-400';
}
