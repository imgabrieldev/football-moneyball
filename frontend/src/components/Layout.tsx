import { Link, useLocation } from 'react-router-dom';
import { BarChart3, TrendingUp, Users, Target, CheckCircle, Home, History } from 'lucide-react';

const navItems = [
  { path: '/', label: 'Dashboard', icon: Home },
  { path: '/predictions', label: 'Previsões', icon: BarChart3 },
  { path: '/value-bets', label: 'Value Bets', icon: TrendingUp },
  { path: '/track-record', label: 'Track Record', icon: History },
  { path: '/players', label: 'Jogadores', icon: Users },
  { path: '/backtest', label: 'Backtest', icon: Target },
  { path: '/verify', label: 'Verificação', icon: CheckCircle },
];

export function Layout({ children }: { children: React.ReactNode }) {
  const location = useLocation();

  return (
    <div className="flex h-screen bg-gray-950 text-gray-100">
      {/* Sidebar */}
      <nav className="w-56 bg-gray-900 border-r border-gray-800 flex flex-col">
        <div className="p-4 border-b border-gray-800">
          <h1 className="text-lg font-bold text-green-400">⚽ Moneyball</h1>
          <p className="text-xs text-gray-500">Football Analytics</p>
        </div>
        <div className="flex-1 py-2">
          {navItems.map(({ path, label, icon: Icon }) => {
            const active = location.pathname === path;
            return (
              <Link
                key={path}
                to={path}
                className={`flex items-center gap-3 px-4 py-2.5 text-sm transition-colors ${
                  active
                    ? 'bg-gray-800 text-green-400 border-r-2 border-green-400'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
                }`}
              >
                <Icon size={18} />
                {label}
              </Link>
            );
          })}
        </div>
        <div className="p-4 border-t border-gray-800 text-xs text-gray-600">
          Brasileirão 2026 • v0.9.0
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-1 overflow-auto p-6">
        {children}
      </main>
    </div>
  );
}
