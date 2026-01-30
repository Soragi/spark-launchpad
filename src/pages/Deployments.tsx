import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import Layout from "@/components/layout/Layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { useDeployments } from "@/hooks/use-deployments";
import { useContainerLogs } from "@/hooks/use-container-logs";
import { useSavedLaunchables } from "@/hooks/use-saved-launchables";
import { useAutoDeploy } from "@/hooks/use-auto-deploy";
import { getContainerLogs } from "@/lib/api";
import {
  Container,
  Play,
  Square,
  ExternalLink,
  RefreshCw,
  Terminal,
  Clock,
  HardDrive,
  Wifi,
  WifiOff,
  X,
  Loader2,
  Trash2,
  Rocket,
  AlertCircle,
  CheckCircle,
  XCircle,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useToast } from "@/hooks/use-toast";

interface LogViewerProps {
  containerId: string;
  containerName: string;
  onClose: () => void;
}

const LogViewer = ({ containerId, containerName, onClose }: LogViewerProps) => {
  const { logs, isConnected, error, clearLogs } = useContainerLogs({
    containerId,
    enabled: true,
  });
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <DialogContent className="max-w-4xl h-[600px] flex flex-col bg-card border-border">
      <DialogHeader className="flex-shrink-0">
        <div className="flex items-center justify-between">
          <DialogTitle className="flex items-center gap-2">
            <Terminal className="h-5 w-5" />
            Logs: {containerName}
          </DialogTitle>
          <div className="flex items-center gap-2">
            {isConnected ? (
              <Badge variant="outline" className="bg-status-running/20 text-status-running border-status-running/30">
                <Wifi className="h-3 w-3 mr-1" />
                Live
              </Badge>
            ) : (
              <Badge variant="outline" className="bg-muted text-muted-foreground">
                <WifiOff className="h-3 w-3 mr-1" />
                Disconnected
              </Badge>
            )}
            <Button variant="ghost" size="sm" onClick={clearLogs}>
              Clear
            </Button>
          </div>
        </div>
      </DialogHeader>

      {error && (
        <div className="px-4 py-2 bg-destructive/10 text-destructive text-sm rounded">
          {error}
        </div>
      )}

      <ScrollArea className="flex-1 bg-secondary rounded-lg p-4">
        <div ref={scrollRef} className="font-mono text-xs space-y-1">
          {logs.length === 0 ? (
            <div className="text-muted-foreground">Waiting for logs...</div>
          ) : (
            logs.map((log, index) => (
              <div
                key={index}
                className={`${
                  log.type === "error" ? "text-destructive" : "text-foreground"
                }`}
              >
                {log.message}
              </div>
            ))
          )}
        </div>
      </ScrollArea>
    </DialogContent>
  );
};

interface DeploymentLogViewerProps {
  logs: string[];
  status: string;
  currentStep: number;
  totalSteps: number;
  error?: string | null;
  onClose: () => void;
}

const DeploymentLogViewer = ({
  logs,
  status,
  currentStep,
  totalSteps,
  error,
  onClose,
}: DeploymentLogViewerProps) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  const progress = totalSteps > 0 ? (currentStep / totalSteps) * 100 : 0;

  const statusIcon = {
    pending: <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />,
    running: <Loader2 className="h-4 w-4 animate-spin text-primary" />,
    completed: <CheckCircle className="h-4 w-4 text-status-running" />,
    failed: <XCircle className="h-4 w-4 text-destructive" />,
    dry_run: <AlertCircle className="h-4 w-4 text-yellow-500" />,
  }[status] || <AlertCircle className="h-4 w-4" />;

  return (
    <DialogContent className="max-w-4xl h-[600px] flex flex-col bg-card border-border">
      <DialogHeader className="flex-shrink-0">
        <div className="flex items-center justify-between">
          <DialogTitle className="flex items-center gap-2">
            <Rocket className="h-5 w-5 text-primary" />
            Deployment Progress
          </DialogTitle>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="flex items-center gap-1">
              {statusIcon}
              {status}
            </Badge>
          </div>
        </div>
      </DialogHeader>

      {/* Progress bar */}
      <div className="space-y-2">
        <div className="flex justify-between text-sm text-muted-foreground">
          <span>Step {currentStep} of {totalSteps}</span>
          <span>{Math.round(progress)}%</span>
        </div>
        <Progress value={progress} className="h-2" />
      </div>

      {error && (
        <div className="px-4 py-2 bg-destructive/10 text-destructive text-sm rounded border border-destructive/30">
          <strong>Error:</strong> {error}
        </div>
      )}

      <ScrollArea className="flex-1 bg-secondary rounded-lg p-4">
        <div ref={scrollRef} className="font-mono text-xs space-y-1">
          {logs.length === 0 ? (
            <div className="text-muted-foreground">Waiting for deployment to start...</div>
          ) : (
            logs.map((log, index) => (
              <div
                key={index}
                className={`${
                  log.startsWith("ERROR") ? "text-destructive" : 
                  log.startsWith("$") ? "text-primary" :
                  log.startsWith("---") ? "text-muted-foreground font-bold mt-2" :
                  log.startsWith("✓") ? "text-status-running" :
                  "text-foreground"
                }`}
              >
                {log}
              </div>
            ))
          )}
        </div>
      </ScrollArea>
    </DialogContent>
  );
};

const Deployments = () => {
  const navigate = useNavigate();
  const { deployments, isLoading, stop, refetch } = useDeployments();
  const { savedLaunchables, removeLaunchable } = useSavedLaunchables();
  const { toast } = useToast();
  const [selectedContainer, setSelectedContainer] = useState<{
    id: string;
    name: string;
  } | null>(null);
  const [stoppingId, setStoppingId] = useState<string | null>(null);
  const [showDeploymentLogs, setShowDeploymentLogs] = useState(false);
  const [deployingId, setDeployingId] = useState<string | null>(null);

  const {
    deploy,
    job,
    logs: deployLogs,
    isDeploying,
    error: deployError,
    reset: resetDeploy,
  } = useAutoDeploy({
    onComplete: (completedJob) => {
      toast({
        title: "Deployment Complete",
        description: `${completedJob.launchable_id} has been deployed successfully!`,
      });
      refetch();
    },
    onError: (error) => {
      toast({
        title: "Deployment Failed",
        description: error,
        variant: "destructive",
      });
    },
  });

  const handleStop = async (id: string) => {
    setStoppingId(id);
    try {
      await stop(id);
    } finally {
      setStoppingId(null);
    }
  };

  const handleDeploy = async (launchableId: string, requiresApiKey?: boolean) => {
    // Check for API keys if required
    if (requiresApiKey) {
      const hfKey = localStorage.getItem('hf_api_key');
      if (!hfKey) {
        toast({
          title: "API Key Required",
          description: "Please configure your HuggingFace API key in Settings first.",
          variant: "destructive",
        });
        navigate('/settings');
        return;
      }
    }

    setDeployingId(launchableId);
    setShowDeploymentLogs(true);
    await deploy(launchableId);
    setDeployingId(null);
  };

  const getPortUrl = (ports: Record<string, string> | undefined) => {
    if (!ports) return null;
    const firstPort = Object.values(ports)[0];
    if (firstPort) {
      return `http://localhost:${firstPort}`;
    }
    return null;
  };

  const runningDeployments = deployments.filter((d) => d.status === "running");
  const stoppedDeployments = deployments.filter((d) => d.status !== "running");

  return (
    <Layout>
      <div className="space-y-8 animate-fade-in">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold flex items-center gap-3">
              <Container className="h-8 w-8 text-primary" />
              Deployments
            </h1>
            <p className="text-muted-foreground mt-2">
              Monitor and manage your running containers
            </p>
          </div>
          <Button variant="outline" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>

        {/* Saved Launchables Section */}
        {savedLaunchables.length > 0 && (
          <div>
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Rocket className="h-5 w-5 text-primary" />
              Saved Launchables ({savedLaunchables.length})
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {savedLaunchables.map((launchable) => (
                <Card
                  key={launchable.id}
                  className="bg-card border-border hover:border-primary/50 transition-colors"
                >
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg font-medium">
                        {launchable.title}
                      </CardTitle>
                      <Badge variant="secondary" className="bg-primary/20 text-primary border-0">
                        <Clock className="h-3 w-3 mr-1" />
                        {launchable.duration}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <p className="text-sm text-muted-foreground line-clamp-2">
                      {launchable.description}
                    </p>
                    <div className="flex gap-2 pt-2 border-t border-border">
                      <Button
                        size="sm"
                        className="flex-1 nvidia-gradient nvidia-glow"
                        onClick={() => handleDeploy(launchable.id, launchable.requiresApiKey)}
                        disabled={isDeploying && deployingId === launchable.id}
                      >
                        {isDeploying && deployingId === launchable.id ? (
                          <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                        ) : (
                          <Rocket className="h-4 w-4 mr-1" />
                        )}
                        Deploy
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        asChild
                      >
                        <a href={launchable.url} target="_blank" rel="noopener noreferrer">
                          <ExternalLink className="h-4 w-4" />
                        </a>
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => removeLaunchable(launchable.id)}
                        title="Remove from deployments"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}

        {/* Running Containers */}
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        ) : deployments.length === 0 && savedLaunchables.length === 0 ? (
          <Card className="bg-card border-border">
            <CardContent className="py-12 text-center">
              <Container className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium mb-2">No Deployments</h3>
              <p className="text-muted-foreground">
                Add a launchable from the Launchables page to see it here.
              </p>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-6">
            {/* Running Containers */}
            {runningDeployments.length > 0 && (
              <div>
                <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                  <span className="inline-block w-2 h-2 rounded-full bg-status-running animate-pulse" />
                  Running ({runningDeployments.length})
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {runningDeployments.map((deployment) => (
                    <Card
                      key={deployment.id}
                      className="bg-card border-border hover:border-primary/50 transition-colors"
                    >
                      <CardHeader className="pb-2">
                        <div className="flex items-center justify-between">
                          <CardTitle className="text-lg font-medium capitalize">
                            {deployment.id.replace(/-/g, " ")}
                          </CardTitle>
                          <Badge className="bg-status-running/20 text-status-running border-0">
                            <span className="inline-block w-1.5 h-1.5 rounded-full bg-status-running animate-pulse mr-1" />
                            Running
                          </Badge>
                        </div>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        {/* Container Info */}
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div className="flex items-center gap-1 text-muted-foreground">
                            <HardDrive className="h-3 w-3" />
                            <span>ID:</span>
                          </div>
                          <div className="font-mono text-xs truncate">
                            {deployment.container_id}
                          </div>

                          {deployment.uptime && (
                            <>
                              <div className="flex items-center gap-1 text-muted-foreground">
                                <Clock className="h-3 w-3" />
                                <span>Uptime:</span>
                              </div>
                              <div>{deployment.uptime}</div>
                            </>
                          )}

                          {deployment.ports && Object.keys(deployment.ports).length > 0 && (
                            <>
                              <div className="text-muted-foreground">Ports:</div>
                              <div className="font-mono text-xs">
                                {Object.entries(deployment.ports).map(([port, hostPort]) => (
                                  <div key={port}>
                                    {hostPort} → {port}
                                  </div>
                                ))}
                              </div>
                            </>
                          )}

                          {deployment.image && (
                            <>
                              <div className="text-muted-foreground">Image:</div>
                              <div className="font-mono text-xs truncate" title={deployment.image}>
                                {deployment.image.split("/").pop()?.split(":")[0]}
                              </div>
                            </>
                          )}
                        </div>

                        {/* Actions */}
                        <div className="flex gap-2 pt-2 border-t border-border">
                          <Button
                            variant="outline"
                            size="sm"
                            className="flex-1"
                            onClick={() =>
                              setSelectedContainer({
                                id: deployment.id,
                                name: deployment.name,
                              })
                            }
                          >
                            <Terminal className="h-4 w-4 mr-1" />
                            Logs
                          </Button>
                          {getPortUrl(deployment.ports) && (
                            <Button
                              variant="outline"
                              size="sm"
                              className="flex-1"
                              onClick={() =>
                                window.open(getPortUrl(deployment.ports), "_blank")
                              }
                            >
                              <ExternalLink className="h-4 w-4 mr-1" />
                              Open
                            </Button>
                          )}
                          <Button
                            variant="destructive"
                            size="sm"
                            onClick={() => handleStop(deployment.id)}
                            disabled={stoppingId === deployment.id}
                          >
                            {stoppingId === deployment.id ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <Square className="h-4 w-4" />
                            )}
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            )}

            {/* Stopped Containers */}
            {stoppedDeployments.length > 0 && (
              <div>
                <h2 className="text-xl font-semibold mb-4 text-muted-foreground">
                  Stopped ({stoppedDeployments.length})
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {stoppedDeployments.map((deployment) => (
                    <Card
                      key={deployment.id}
                      className="bg-card border-border opacity-60"
                    >
                      <CardHeader className="pb-2">
                        <div className="flex items-center justify-between">
                          <CardTitle className="text-lg font-medium capitalize">
                            {deployment.id.replace(/-/g, " ")}
                          </CardTitle>
                          <Badge variant="outline" className="text-muted-foreground">
                            {deployment.status}
                          </Badge>
                        </div>
                      </CardHeader>
                      <CardContent>
                        <div className="text-sm text-muted-foreground">
                          Container ID: {deployment.container_id}
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Container Log Viewer Dialog */}
      <Dialog
        open={!!selectedContainer}
        onOpenChange={() => setSelectedContainer(null)}
      >
        {selectedContainer && (
          <LogViewer
            containerId={selectedContainer.id}
            containerName={selectedContainer.name}
            onClose={() => setSelectedContainer(null)}
          />
        )}
      </Dialog>

      {/* Deployment Log Viewer Dialog */}
      <Dialog
        open={showDeploymentLogs}
        onOpenChange={(open) => {
          if (!open && !isDeploying) {
            setShowDeploymentLogs(false);
            resetDeploy();
          }
        }}
      >
        <DeploymentLogViewer
          logs={deployLogs}
          status={job?.status || 'pending'}
          currentStep={job?.current_step || 0}
          totalSteps={job?.total_steps || 0}
          error={deployError}
          onClose={() => {
            setShowDeploymentLogs(false);
            resetDeploy();
          }}
        />
      </Dialog>
    </Layout>
  );
};

export default Deployments;
