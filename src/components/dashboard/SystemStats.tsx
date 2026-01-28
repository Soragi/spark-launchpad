import { Card, CardContent } from "@/components/ui/card";
import GaugeChart from "./GaugeChart";
import UsageChart from "./UsageChart";
import { Skeleton } from "@/components/ui/skeleton";
import { Thermometer, Cpu } from "lucide-react";

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
              <Skeleton className="h-32 w-32 rounded-full" />
              <Skeleton className="w-full lg:w-48 h-32" />
            </div>
          </CardContent>
        </Card>
        <Card className="bg-card border-border">
          <CardContent className="p-6">
            <div className="flex flex-col lg:flex-row items-center justify-around gap-6">
              <Skeleton className="h-32 w-32 rounded-full" />
              <Skeleton className="w-full lg:w-48 h-32" />
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <Card className="bg-card border-border">
        <CardContent className="p-6">
          <div className="flex flex-col lg:flex-row items-center justify-around gap-6">
            <GaugeChart
              value={memoryUsed}
              max={memoryTotal}
              label="System Memory"
              unit="GB"
            />
            <div className="w-full lg:w-48 h-32">
              <div className="text-xs text-muted-foreground mb-2 text-right">
                {memoryTotal.toFixed(0)}GB
              </div>
              <UsageChart data={memoryHistory} max={memoryTotal} color="hsl(var(--gauge-green))" />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="bg-card border-border">
        <CardContent className="p-6">
          <div className="flex flex-col lg:flex-row items-center justify-around gap-6">
            <GaugeChart
              value={gpuUtilization}
              max={100}
              label="GPU Utilization"
              showPercentage
            />
            <div className="w-full lg:w-48 h-32">
              <div className="text-xs text-muted-foreground mb-2 text-right">
                100%
              </div>
              <UsageChart data={gpuHistory} max={100} color="hsl(var(--gauge-green))" />
              {/* GPU Temperature and Driver info */}
              {(gpuTemperature !== undefined || driverVersion) && (
                <div className="mt-2 flex items-center gap-4 text-xs text-muted-foreground">
                  {gpuTemperature !== undefined && (
                    <div className="flex items-center gap-1">
                      <Thermometer className="h-3 w-3" />
                      <span>{gpuTemperature}°C</span>
                    </div>
                  )}
                  {driverVersion && (
                    <div className="flex items-center gap-1">
                      <Cpu className="h-3 w-3" />
                      <span>v{driverVersion}</span>
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
