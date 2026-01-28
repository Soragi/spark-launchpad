import { useState } from "react";
import Layout from "@/components/layout/Layout";
import SystemStats from "@/components/dashboard/SystemStats";
import ServiceCard from "@/components/dashboard/ServiceCard";
import { Cpu, MonitorPlay, Rocket, AlertCircle } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { useSystemStats } from "@/hooks/use-system-stats";
import { manageJupyterLab } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import { Alert, AlertDescription } from "@/components/ui/alert";

const Index = () => {
  const { stats, memoryHistory, gpuHistory, isLoading, error } = useSystemStats();
  const [jupyterStatus, setJupyterStatus] = useState<"running" | "stopped">("stopped");
  const [jupyterLoading, setJupyterLoading] = useState(false);
  const { toast } = useToast();

  const handleStartJupyter = async () => {
    setJupyterLoading(true);
    try {
      await manageJupyterLab("start");
      setJupyterStatus("running");
      toast({
        title: "JupyterLab Started",
        description: "JupyterLab is now running at http://localhost:8888",
      });
    } catch (err) {
      toast({
        title: "Failed to start JupyterLab",
        description: err instanceof Error ? err.message : "Unknown error",
        variant: "destructive",
      });
    } finally {
      setJupyterLoading(false);
    }
  };

  const handleStopJupyter = async () => {
    setJupyterLoading(true);
    try {
      await manageJupyterLab("stop");
      setJupyterStatus("stopped");
      toast({
        title: "JupyterLab Stopped",
        description: "JupyterLab has been stopped.",
      });
    } catch (err) {
      toast({
        title: "Failed to stop JupyterLab",
        description: err instanceof Error ? err.message : "Unknown error",
        variant: "destructive",
      });
    } finally {
      setJupyterLoading(false);
    }
  };

  const JupyterIcon = (
    <div className="text-orange-500 font-bold text-lg italic">
      jupyter<span className="text-orange-400">lab</span>
    </div>
  );

  return (
    <Layout>
      <div className="space-y-8 animate-fade-in">
        {/* Welcome header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-2">
              <span className="text-2xl">👋</span> Welcome
            </h1>
            <h2 className="text-3xl font-bold mt-2">
              Your <span className="text-primary nvidia-text-glow">DGX</span> Dashboard
            </h2>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Cpu className="h-4 w-4 text-primary" />
              <span>{stats?.gpu_name || "GB10 Grace Blackwell Superchip"}</span>
            </div>
          </div>
        </div>

        {/* API Connection Error */}
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Unable to connect to backend API. System stats are unavailable. 
              Make sure the backend is running on port 8000.
            </AlertDescription>
          </Alert>
        )}

        {/* System stats */}
        <SystemStats
          memoryUsed={stats?.memory_used ?? 0}
          memoryTotal={stats?.memory_total ?? 128}
          gpuUtilization={stats?.gpu_utilization ?? 0}
          memoryHistory={memoryHistory}
          gpuHistory={gpuHistory}
          gpuTemperature={stats?.gpu_temperature}
          driverVersion={stats?.driver_version}
          isLoading={isLoading}
        />

        {/* Main content grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* JupyterLab Service */}
          <ServiceCard
            name="JupyterLab"
            status={jupyterStatus}
            icon={JupyterIcon}
            logs={jupyterStatus === "running" ? ["[I] Server running at http://localhost:8888/"] : []}
            onStart={handleStartJupyter}
            onStop={handleStopJupyter}
            onOpenInBrowser={() => window.open("http://localhost:8888", "_blank")}
            isLoading={jupyterLoading}
          />

          {/* Quick Links */}
          <Card className="bg-card border-border">
            <CardContent className="p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Rocket className="h-5 w-5 text-primary" />
                Quick Actions
              </h3>
              <div className="space-y-3">
                <Link to="/launchables">
                  <Button className="w-full justify-start nvidia-gradient nvidia-glow">
                    <MonitorPlay className="h-4 w-4 mr-2" />
                    Browse Launchables
                  </Button>
                </Link>
                <a
                  href="https://build.nvidia.com/spark"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Button variant="outline" className="w-full justify-start">
                    Explore NVIDIA Spark Playbooks
                  </Button>
                </a>
                <Link to="/settings">
                  <Button variant="outline" className="w-full justify-start">
                    Configure API Keys
                  </Button>
                </Link>
              </div>

              <div className="mt-6 p-4 bg-secondary rounded-lg">
                <h4 className="text-sm font-medium mb-2">System Info</h4>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="text-muted-foreground">GPU Memory</div>
                  <div className="font-mono">128 GB</div>
                  <div className="text-muted-foreground">CPU Cores</div>
                  <div className="font-mono">20</div>
                  <div className="text-muted-foreground">Storage</div>
                  <div className="font-mono">4 TB NVMe</div>
                  <div className="text-muted-foreground">OS</div>
                  <div className="font-mono">DGX OS</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </Layout>
  );
};

export default Index;
