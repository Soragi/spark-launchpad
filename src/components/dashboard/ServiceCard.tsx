import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Play, Square, ExternalLink, ChevronDown, Copy, Loader2 } from "lucide-react";

interface ServiceCardProps {
  name: string;
  status: "running" | "stopped";
  icon?: React.ReactNode;
  logs?: string[];
  onStart?: () => void;
  onStop?: () => void;
  onOpenInBrowser?: () => void;
  isLoading?: boolean;
}

const ServiceCard = ({
  name,
  status,
  icon,
  logs = [],
  onStart,
  onStop,
  onOpenInBrowser,
  isLoading = false,
}: ServiceCardProps) => {
  const [showLogs, setShowLogs] = useState(false);

  return (
    <Card className="bg-card border-border">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <div className="flex items-center gap-3">
          {icon && <div className="text-primary">{icon}</div>}
          <CardTitle className="text-lg font-medium">{name}</CardTitle>
          <span
            className={`px-2 py-0.5 text-xs font-medium rounded-full ${
              status === "running"
                ? "bg-status-running/20 text-status-running"
                : "bg-muted text-muted-foreground"
            }`}
          >
            <span className={`inline-block w-1.5 h-1.5 rounded-full mr-1.5 ${
              status === "running" ? "bg-status-running animate-pulse-glow" : "bg-muted-foreground"
            }`} />
            {status.toUpperCase()}
          </span>
        </div>
        <Button
          variant={status === "running" ? "destructive" : "default"}
          size="sm"
          onClick={status === "running" ? onStop : onStart}
          className={status === "running" ? "" : "nvidia-gradient"}
          disabled={isLoading}
        >
          {isLoading ? (
            <>
              <Loader2 className="h-3 w-3 mr-1 animate-spin" /> Loading...
            </>
          ) : status === "running" ? (
            <>
              <Square className="h-3 w-3 mr-1" /> Stop
            </>
          ) : (
            <>
              <Play className="h-3 w-3 mr-1" /> Start
            </>
          )}
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <div
            className="flex items-center justify-between cursor-pointer"
            onClick={() => setShowLogs(!showLogs)}
          >
            <label className="text-sm text-muted-foreground">Output Log</label>
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="icon" className="h-6 w-6">
                <Copy className="h-3 w-3" />
              </Button>
              <ChevronDown
                className={`h-4 w-4 text-muted-foreground transition-transform ${
                  showLogs ? "rotate-180" : ""
                }`}
              />
            </div>
          </div>
          <div className={`bg-secondary rounded-md p-3 font-mono text-xs min-h-[60px] ${showLogs ? "" : "max-h-[60px] overflow-hidden"}`}>
            {logs.length > 0 ? (
              logs.map((log, i) => (
                <div key={i} className="text-muted-foreground">
                  {log}
                </div>
              ))
            ) : (
              <span className="text-muted-foreground">No logs yet</span>
            )}
          </div>
        </div>

        <Button
          variant="outline"
          className="w-full"
          disabled={status !== "running"}
          onClick={onOpenInBrowser}
        >
          <ExternalLink className="h-4 w-4 mr-2" />
          Open in Browser
        </Button>
      </CardContent>
    </Card>
  );
};

export default ServiceCard;
