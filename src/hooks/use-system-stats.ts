import { useState, useEffect, useCallback } from 'react';
import { fetchSystemStats, SystemStats } from '@/lib/api';

interface UseSystemStatsOptions {
  pollingInterval?: number;
  enabled?: boolean;
}

interface UseSystemStatsResult {
  stats: SystemStats | null;
  memoryHistory: number[];
  gpuHistory: number[];
  isLoading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

const MAX_HISTORY_LENGTH = 20;

export function useSystemStats(options: UseSystemStatsOptions = {}): UseSystemStatsResult {
  const { pollingInterval = 3000, enabled = true } = options;
  
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [memoryHistory, setMemoryHistory] = useState<number[]>([]);
  const [gpuHistory, setGpuHistory] = useState<number[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const refetch = useCallback(async () => {
    try {
      const data = await fetchSystemStats();
      setStats(data);
      setError(null);
      
      // Update history
      setMemoryHistory(prev => {
        const newHistory = [...prev, data.memory_used];
        return newHistory.slice(-MAX_HISTORY_LENGTH);
      });
      
      setGpuHistory(prev => {
        const newHistory = [...prev, data.gpu_utilization];
        return newHistory.slice(-MAX_HISTORY_LENGTH);
      });
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to fetch stats'));
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;

    // Initial fetch
    refetch();

    // Set up polling
    const interval = setInterval(refetch, pollingInterval);
    return () => clearInterval(interval);
  }, [enabled, pollingInterval, refetch]);

  return {
    stats,
    memoryHistory,
    gpuHistory,
    isLoading,
    error,
    refetch,
  };
}
