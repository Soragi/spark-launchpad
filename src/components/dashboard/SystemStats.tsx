import { Card, CardContent } from "@/components/ui/card";
import GaugeChart from "./GaugeChart";
import UsageChart from "./UsageChart";
import { Skeleton } from "@/components/ui/skeleton";
import { Thermometer, Cpu, Activity } from "lucide-react";

interface SystemStatsProps {
  memoryUsed: number;
  memoryTotal: number;
  gpuUtilization: number;
  memoryHistory: number[];
  gpuHistory: number[];
  gpuTemperature?: number;
  driverVersion?: string;
  isLoading?: boolean;
}

const SystemStats = ({
  memoryUsed,
  memoryTotal,
  gpuUtilization,
  memoryHistory,
  gpuHistory,
  gpuTemperature,
  driverVersion,
  isLoading,
}: SystemStatsProps) => {
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card className="bg-card border-border">
          <CardContent className="p-6">
            <div className="flex flex-col lg:flex-row items-center justify-around gap-6">
              <Skeleton className="h-28 w-40 rounded-xl" />
              <Skeleton className="w-full lg:w-48 h-24 rounded-xl" />
            </div>
          </CardContent>
        </Card>
        <Card className="bg-card border-border">
          <CardContent className="p-6">
            <div className="flex flex-col lg:flex-row items-center justify-around gap-6">
              <Skeleton className="h-28 w-40 rounded-xl" />
              <Skeleton className="w-full lg:w-48 h-24 rounded-xl" />
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* System Memory Card */}
      <Card className="bg-card border-border overflow-hidden">
        <CardContent className="p-6">
          <div className="flex flex-col lg:flex-row items-center justify-around gap-6">
            <GaugeChart
              value={memoryUsed}
              max={memoryTotal}
              label="System Memory"
              unit="GB"
            />
            <div className="w-full lg:w-52 space-y-2">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span className="flex items-center gap-1">
                  <Activity className="h-3 w-3" />
                  Live Usage
                </span>
                <span className="font-mono">{memoryTotal.toFixed(0)} GB</span>
              </div>
              <div className="h-20 bg-secondary/30 rounded-lg p-2">
                <UsageChart 
                  data={memoryHistory} 
                  max={memoryTotal} 
                  color="hsl(var(--gauge-green))"
                  gradientId="memory-gradient"
                />
              </div>
              <div className="flex justify-between text-[10px] text-muted-foreground font-mono">
                <span>-1m</span>
                <span>now</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* GPU Utilization Card */}
      <Card className="bg-card border-border overflow-hidden">
        <CardContent className="p-6">
          <div className="flex flex-col lg:flex-row items-center justify-around gap-6">
            <GaugeChart
              value={gpuUtilization}
              max={100}
              label="GPU Utilization"
              showPercentage
            />
            <div className="w-full lg:w-52 space-y-2">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span className="flex items-center gap-1">
                  <Activity className="h-3 w-3" />
                  Live Usage
                </span>
                <span className="font-mono">100%</span>
              </div>
              <div className="h-20 bg-secondary/30 rounded-lg p-2">
                <UsageChart 
                  data={gpuHistory} 
                  max={100} 
                  color="hsl(var(--primary))"
                  gradientId="gpu-gradient"
                />
              </div>
              <div className="flex justify-between text-[10px] text-muted-foreground font-mono">
                <span>-1m</span>
                <span>now</span>
              </div>
              
              {/* GPU Info badges */}
              {(gpuTemperature !== undefined || driverVersion) && (
                <div className="flex items-center gap-3 pt-2">
                  {gpuTemperature !== undefined && (
                    <div className="flex items-center gap-1.5 px-2 py-1 bg-secondary/50 rounded-md text-xs">
                      <Thermometer className="h-3 w-3 text-orange-400" />
                      <span className="font-mono">{gpuTemperature}°C</span>
                    </div>
                  )}
                  {driverVersion && (
                    <div className="flex items-center gap-1.5 px-2 py-1 bg-secondary/50 rounded-md text-xs">
                      <Cpu className="h-3 w-3 text-primary" />
                      <span className="font-mono">v{driverVersion}</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default SystemStats;
