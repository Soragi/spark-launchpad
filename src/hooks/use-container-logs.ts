import { useState, useEffect, useCallback, useRef } from 'react';
import { getLogsWebSocketUrl } from '@/lib/api';

interface LogMessage {
  type: 'log' | 'error';
  message: string;
  timestamp?: Date;
}

interface UseContainerLogsOptions {
  containerId: string;
  enabled?: boolean;
  maxLogs?: number;
}

interface UseContainerLogsResult {
  logs: LogMessage[];
  isConnected: boolean;
  error: string | null;
  connect: () => void;
  disconnect: () => void;
  clearLogs: () => void;
}

export function useContainerLogs(options: UseContainerLogsOptions): UseContainerLogsResult {
  const { containerId, enabled = true, maxLogs = 500 } = options;
  
  const [logs, setLogs] = useState<LogMessage[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const clearLogs = useCallback(() => {
    setLogs([]);
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    disconnect();

    try {
      const wsUrl = getLogsWebSocketUrl(containerId);
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        setError(null);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const newLog: LogMessage = {
            type: data.type,
            message: data.message,
            timestamp: new Date(),
          };

          setLogs((prev) => {
            const updated = [...prev, newLog];
            // Keep only the last maxLogs entries
            return updated.slice(-maxLogs);
          });

          if (data.type === 'error') {
            setError(data.message);
          }
        } catch {
          // Handle non-JSON messages
          setLogs((prev) => [...prev.slice(-maxLogs + 1), {
            type: 'log',
            message: event.data,
            timestamp: new Date(),
          }]);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        wsRef.current = null;

        // Attempt to reconnect after 3 seconds if still enabled
        if (enabled) {
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, 3000);
        }
      };

      ws.onerror = () => {
        setError('WebSocket connection error');
        setIsConnected(false);
      };
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to connect');
      setIsConnected(false);
    }
  }, [containerId, enabled, maxLogs, disconnect]);

  useEffect(() => {
    if (enabled && containerId) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [enabled, containerId, connect, disconnect]);

  return {
    logs,
    isConnected,
    error,
    connect,
    disconnect,
    clearLogs,
  };
}
