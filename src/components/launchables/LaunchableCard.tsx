import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Clock, Rocket, ExternalLink, Square, CheckCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { DeploymentStatus } from "@/lib/api";

export interface Launchable {
  id: string;
  title: string;
  description: string;
  duration: string;
  url: string;
  category?: "quickstart" | "new" | "featured" | "all";
  requiresApiKey?: boolean;
}

interface LaunchableCardProps {
  launchable: Launchable;
  onDeploy: (launchable: Launchable) => void;
  deploymentStatus?: DeploymentStatus;
}

const LaunchableCard = ({ launchable, onDeploy, deploymentStatus }: LaunchableCardProps) => {
  const isRunning = deploymentStatus?.status === "running";
  return (
    <Card className="bg-card border-border card-hover group overflow-hidden">
      <CardContent className="p-5 h-full flex flex-col">
        <div className="flex items-start justify-between mb-3">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              {isRunning && (
                <Badge variant="secondary" className="bg-status-running/20 text-status-running border-0">
                  <span className="inline-block w-1.5 h-1.5 rounded-full bg-status-running animate-pulse mr-1" />
                  Running
                </Badge>
              )}
              {launchable.category === "new" && !isRunning && (
                <Badge variant="secondary" className="bg-primary/20 text-primary border-0">
                  New
                </Badge>
              )}
              {launchable.category === "quickstart" && !isRunning && (
                <Badge variant="secondary" className="bg-blue-500/20 text-blue-400 border-0">
                  Quickstart
                </Badge>
              )}
            </div>
            <h3 className="font-semibold text-foreground group-hover:text-primary transition-colors line-clamp-2">
              {launchable.title}
            </h3>
          </div>
        </div>

        <p className="text-sm text-muted-foreground mb-4 flex-1 line-clamp-3">
          {launchable.description}
        </p>

        <div className="flex items-center justify-between pt-3 border-t border-border">
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <Clock className="h-3 w-3" />
            <span>{launchable.duration}</span>
          </div>
          <div className="flex gap-2">
            <Button
              variant="ghost"
              size="sm"
              asChild
              className="text-muted-foreground hover:text-foreground"
            >
              <a href={launchable.url} target="_blank" rel="noopener noreferrer">
                <ExternalLink className="h-4 w-4" />
              </a>
            </Button>
            <Button
              size="sm"
              onClick={() => onDeploy(launchable)}
              variant={isRunning ? "destructive" : "default"}
              className={isRunning ? "" : "nvidia-gradient nvidia-glow"}
            >
              {isRunning ? (
                <>
                  <Square className="h-4 w-4 mr-1" />
                  Stop
                </>
              ) : (
                <>
                  <Rocket className="h-4 w-4 mr-1" />
                  Deploy
                </>
              )}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default LaunchableCard;
