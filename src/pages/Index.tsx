import { useState, useEffect } from "react";
import Layout from "@/components/layout/Layout";
import SystemStats from "@/components/dashboard/SystemStats";
import ServiceCard from "@/components/dashboard/ServiceCard";
import { Cpu, MonitorPlay, Rocket } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

const Index = () => {
  // Simulated system stats - in production these would come from system APIs
  const [memoryUsed, setMemoryUsed] = useState(9.58);
  const [gpuUtilization, setGpuUtilization] = useState(0);
  const [memoryHistory, setMemoryHistory] = useState<number[]>([8, 9, 8.5, 9.2, 9.58, 10, 9.5, 9.58]);
  const [gpuHistory, setGpuHistory] = useState<number[]>([0, 5, 2, 0, 3, 0, 0, 0]);
  const [jupyterStatus, setJupyterStatus] = useState<"running" | "stopped">("stopped");

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      const newMemory = memoryUsed + (Math.random() - 0.5) * 0.5;
      const clampedMemory = Math.max(5, Math.min(20, newMemory));
      setMemoryUsed(clampedMemory);
      setMemoryHistory((prev) => [...prev.slice(-7), clampedMemory]);

      const newGpu = jupyterStatus === "running" ? Math.random() * 30 : Math.random() * 5;
      setGpuUtilization(newGpu);
      setGpuHistory((prev) => [...prev.slice(-7), newGpu]);
    }, 3000);

    return () => clearInterval(interval);
  }, [memoryUsed, jupyterStatus]);

  const handleStartJupyter = () => {
    setJupyterStatus("running");
  };

  const handleStopJupyter = () => {
    setJupyterStatus("stopped");
  };

  const JupyterIcon = () => (
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
              <span>GB10 Grace Blackwell Superchip</span>
            </div>
          </div>
        </div>

        {/* System stats */}
        <SystemStats
          memoryUsed={memoryUsed}
          memoryTotal={128}
          gpuUtilization={gpuUtilization}
          memoryHistory={memoryHistory}
          gpuHistory={gpuHistory}
        />

        {/* Main content grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* JupyterLab Service */}
          <ServiceCard
            name="JupyterLab"
            status={jupyterStatus}
            icon={<JupyterIcon />}
            workingDirectory="/home/igaros/jupyterlab"
            logs={jupyterStatus === "running" ? ["[I] Server running at http://localhost:8888/"] : []}
            onStart={handleStartJupyter}
            onStop={handleStopJupyter}
            onOpenInBrowser={() => window.open("http://localhost:8888", "_blank")}
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
