import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { Predictions } from './pages/Predictions';
import { ValueBets } from './pages/ValueBets';
import { Players } from './pages/Players';
import { Backtest } from './pages/Backtest';
import { Verify } from './pages/Verify';
import { TrackRecord } from './pages/TrackRecord';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 60_000, retry: 1 },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/value-bets" element={<ValueBets />} />
            <Route path="/players" element={<Players />} />
            <Route path="/backtest" element={<Backtest />} />
            <Route path="/verify" element={<Verify />} />
            <Route path="/track-record" element={<TrackRecord />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
