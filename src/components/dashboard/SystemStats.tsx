import { Card, CardContent } from "@/components/ui/card";
import GaugeChart from "./GaugeChart";
import UsageChart from "./UsageChart";

interface SystemStatsProps {
  memoryUsed: number;
  memoryTotal: number;
  gpuUtilization: number;
  memoryHistory: number[];
  gpuHistory: number[];
}

const SystemStats = ({
  memoryUsed,
  memoryTotal,
  gpuUtilization,
  memoryHistory,
  gpuHistory,
}: SystemStatsProps) => {
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
                {memoryTotal.toFixed(2)}GB
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
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default SystemStats;
