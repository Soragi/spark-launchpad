import { useState, useEffect, useCallback } from 'react';
import { listDeployments, deployLaunchable, stopDeployment, DeploymentStatus } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';

interface UseDeploymentsResult {
  deployments: DeploymentStatus[];
  isLoading: boolean;
  error: Error | null;
  deploy: (launchableId: string, requiresApiKey?: boolean) => Promise<void>;
  stop: (launchableId: string) => Promise<void>;
  getDeploymentStatus: (launchableId: string) => DeploymentStatus | undefined;
  refetch: () => Promise<void>;
}

export function useDeployments(): UseDeploymentsResult {
  const [deployments, setDeployments] = useState<DeploymentStatus[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const { toast } = useToast();

  const refetch = useCallback(async () => {
    try {
      const data = await listDeployments();
      setDeployments(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to fetch deployments'));
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    refetch();
    // Poll every 10 seconds
    const interval = setInterval(refetch, 10000);
    return () => clearInterval(interval);
  }, [refetch]);

  const deploy = useCallback(async (launchableId: string, requiresApiKey = false) => {
    try {
      let ngcKey: string | undefined;
      let hfKey: string | undefined;

      if (requiresApiKey) {
        // Get API keys from localStorage
        ngcKey = localStorage.getItem('ngc_api_key') || undefined;
        hfKey = localStorage.getItem('hf_api_key') || undefined;

        if (!hfKey) {
          toast({
            title: "API Key Required",
            description: "Please configure your HuggingFace API key in Settings first.",
            variant: "destructive",
          });
          return;
        }
      }

      toast({
        title: "Deploying...",
        description: `Starting deployment of ${launchableId}. This may take a few minutes.`,
      });

      await deployLaunchable(launchableId, ngcKey, hfKey);
      await refetch();

      toast({
        title: "Deployment Successful",
        description: `${launchableId} is now running.`,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Deployment failed';
      toast({
        title: "Deployment Failed",
        description: message,
        variant: "destructive",
      });
      throw err;
    }
  }, [refetch, toast]);

  const stop = useCallback(async (launchableId: string) => {
    try {
      await stopDeployment(launchableId);
      await refetch();

      toast({
        title: "Stopped",
        description: `${launchableId} has been stopped.`,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to stop deployment';
      toast({
        title: "Error",
        description: message,
        variant: "destructive",
      });
      throw err;
    }
  }, [refetch, toast]);

  const getDeploymentStatus = useCallback((launchableId: string) => {
    return deployments.find(d => d.id === launchableId);
  }, [deployments]);

  return {
    deployments,
    isLoading,
    error,
    deploy,
    stop,
    getDeploymentStatus,
    refetch,
  };
}
