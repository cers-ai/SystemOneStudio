import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import { HealthPanel } from '@/features/system/HealthPanel';
import { Wizard } from '@/features/wizard/Wizard';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Long-running training progress arrives over WebSocket, so polling the
      // query cache aggressively only wastes requests.
      staleTime: 30_000,
      retry: 1,
    },
  },
});

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <main>
        <h1>SystemOneStudio</h1>
        <p>一站式可视化决策模型训练平台</p>
        <HealthPanel />
        <hr />
        <Wizard />
      </main>
    </QueryClientProvider>
  );
}
