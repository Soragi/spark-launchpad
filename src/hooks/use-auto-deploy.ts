import { useState, useEffect, useCallback, useRef } from 'react';
import { startAutoDeploy, getDeploymentJob, getJobLogsWebSocketUrl, DeploymentJob } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';

interface UseAutoDeployOptions {
  onComplete?: (job: DeploymentJob) => void;
  onError?: (error: string) => void;
}

interface UseAutoDeployResult {
  deploy: (launchableId: string, dryRun?: boolean) => Promise<DeploymentJob | null>;
  job: DeploymentJob | null;
  logs: string[];
  isDeploying: boolean;
  error: string | null;
  reset: () => void;
}

export function useAutoDeploy(options: UseAutoDeployOptions = {}): UseAutoDeployResult {
  const [job, setJob] = useState<DeploymentJob | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [isDeploying, setIsDeploying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const pollIntervalRef = useRef<number | null>(null);
  const { toast } = useToast();

  // Cleanup function
  const cleanup = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }, []);

  // Reset state
  const reset = useCallback(() => {
    cleanup();
    setJob(null);
    setLogs([]);
    setIsDeploying(false);
    setError(null);
  }, [cleanup]);

  // Poll for job status as fallback
  const startPolling = useCallback((jobId: string) => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
    }
    
    pollIntervalRef.current = window.setInterval(async () => {
      try {
        const updatedJob = await getDeploymentJob(jobId);
        setJob(updatedJob);
        setLogs(updatedJob.logs);
        
        if (updatedJob.status === 'completed' || updatedJob.status === 'failed') {
          cleanup();
          setIsDeploying(false);
          
          if (updatedJob.status === 'completed') {
            options.onComplete?.(updatedJob);
          } else if (updatedJob.error) {
            options.onError?.(updatedJob.error);
          }
        }
      } catch (err) {
        console.error('Failed to poll job status:', err);
      }
    }, 2000);
  }, [cleanup, options]);

  // Connect to WebSocket for live logs
  const connectWebSocket = useCallback((jobId: string) => {
    const wsUrl = getJobLogsWebSocketUrl(jobId);
    
    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      
      ws.onopen = () => {
        console.log('Connected to job logs WebSocket');
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'log') {
            setLogs(prev => [...prev, data.message]);
            setJob(prev => prev ? {
              ...prev,
              status: data.status || prev.status,
              current_step: data.current_step || prev.current_step,
              total_steps: data.total_steps || prev.total_steps,
            } : null);
          } else if (data.type === 'complete') {
            setJob(prev => prev ? { ...prev, status: data.status } : null);
            setIsDeploying(false);
            cleanup();
            
            if (data.status === 'completed') {
              options.onComplete?.(job!);
            } else if (data.error) {
              setError(data.error);
              options.onError?.(data.error);
            }
          } else if (data.type === 'error') {
            setError(data.message);
            options.onError?.(data.message);
          }
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };
      
      ws.onerror = () => {
        console.warn('WebSocket error, falling back to polling');
        ws.close();
        startPolling(jobId);
      };
      
      ws.onclose = () => {
        console.log('WebSocket closed');
      };
      
    } catch (err) {
      console.warn('Failed to connect WebSocket, using polling');
      startPolling(jobId);
    }
  }, [cleanup, job, options, startPolling]);

  // Start deployment
  const deploy = useCallback(async (launchableId: string, dryRun = false): Promise<DeploymentJob | null> => {
    reset();
    setIsDeploying(true);
    setError(null);
    
    // Get API keys from localStorage
    const ngcApiKey = localStorage.getItem('ngc_api_key') || undefined;
    const hfApiKey = localStorage.getItem('hf_api_key') || undefined;
    
    try {
      const newJob = await startAutoDeploy({
        launchable_id: launchableId,
        ngc_api_key: ngcApiKey,
        hf_api_key: hfApiKey,
        dry_run: dryRun,
      });
      
      setJob(newJob);
      setLogs(newJob.logs);
      
      if (dryRun) {
        setIsDeploying(false);
        return newJob;
      }
      
      toast({
        title: "Deployment Started",
        description: `Automated deployment of ${launchableId} has begun.`,
      });
      
      // Connect to WebSocket for live updates
      connectWebSocket(newJob.id);
      
      return newJob;
      
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Deployment failed';
      setError(message);
      setIsDeploying(false);
      
      toast({
        title: "Deployment Failed",
        description: message,
        variant: "destructive",
      });
      
      options.onError?.(message);
      return null;
    }
  }, [reset, toast, connectWebSocket, options]);

  // Cleanup on unmount
  useEffect(() => {
    return cleanup;
  }, [cleanup]);

  return {
    deploy,
    job,
    logs,
    isDeploying,
    error,
    reset,
  };
}
