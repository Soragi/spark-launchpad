import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Clock, ExternalLink, Plus, Check, Rocket, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

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
  onAddToDeployments?: (launchable: Launchable) => void;
  onDeploy?: (launchable: Launchable) => void;
  isSaved?: boolean;
  isDeploying?: boolean;
}

const LaunchableCard = ({
  launchable,
  onAddToDeployments,
  onDeploy,
  isSaved = false,
  isDeploying = false,
}: LaunchableCardProps) => {
  const [showDeployDialog, setShowDeployDialog] = useState(false);

  const handleAddClick = () => {
    if (onDeploy) {
      setShowDeployDialog(true);
    } else if (onAddToDeployments) {
      onAddToDeployments(launchable);
    }
  };

  const handleSaveOnly = () => {
    onAddToDeployments?.(launchable);
    setShowDeployDialog(false);
  };

  const handleDeployNow = () => {
    onDeploy?.(launchable);
    setShowDeployDialog(false);
  };

  const hasApiKeys = () => {
    if (!launchable.requiresApiKey) return true;
    const hfKey = localStorage.getItem('hf_api_key');
    return !!hfKey;
  };

  return (
    <>
      <Card className="bg-card border-border card-hover group overflow-hidden">
        <CardContent className="p-5 h-full flex flex-col">
          <div className="flex items-start justify-between mb-3">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                {launchable.category === "new" && (
                  <Badge variant="secondary" className="bg-primary/20 text-primary border-0">
                    New
                  </Badge>
                )}
                {launchable.category === "quickstart" && (
                  <Badge variant="secondary" className="bg-blue-500/20 text-blue-400 border-0">
                    Quickstart
                  </Badge>
                )}
                {launchable.requiresApiKey && (
                  <Badge variant="outline" className="text-xs">
                    API Key Required
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
            <div className="flex items-center gap-2">
              {onAddToDeployments && (
                <Button
                  size="sm"
                  variant={isSaved ? "secondary" : "outline"}
                  onClick={handleAddClick}
                  disabled={isSaved || isDeploying}
                  title={isSaved ? "Already added to deployments" : "Add to deployments"}
                >
                  {isDeploying ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : isSaved ? (
                    <Check className="h-4 w-4" />
                  ) : (
                    <Plus className="h-4 w-4" />
                  )}
                </Button>
              )}
              <Button
                size="sm"
                asChild
                className="nvidia-gradient nvidia-glow"
              >
                <a href={launchable.url} target="_blank" rel="noopener noreferrer">
                  <ExternalLink className="h-4 w-4 mr-1" />
                  View Blueprint
                </a>
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Deploy Dialog */}
      <Dialog open={showDeployDialog} onOpenChange={setShowDeployDialog}>
        <DialogContent className="bg-card border-border">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Rocket className="h-5 w-5 text-primary" />
              Deploy {launchable.title}
            </DialogTitle>
            <DialogDescription>
              Choose how you want to add this launchable to your deployments.
            </DialogDescription>
          </DialogHeader>

          {launchable.requiresApiKey && !hasApiKeys() && (
            <div className="bg-destructive/10 border border-destructive/30 rounded-lg p-3">
              <p className="text-sm text-destructive">
                <strong>Warning:</strong> This launchable requires API keys. Please configure your HuggingFace API key in Settings before deploying.
              </p>
            </div>
          )}

          <div className="space-y-3">
            <div className="p-4 rounded-lg border border-border bg-secondary/50">
              <h4 className="font-medium mb-1">Save Only</h4>
              <p className="text-sm text-muted-foreground">
                Bookmark this launchable for later. You can deploy it manually from the Deployments page.
              </p>
            </div>
            <div className="p-4 rounded-lg border border-primary/50 bg-primary/5">
              <h4 className="font-medium mb-1 flex items-center gap-2">
                <Rocket className="h-4 w-4 text-primary" />
                Deploy Now
              </h4>
              <p className="text-sm text-muted-foreground">
                Automatically fetch instructions from NVIDIA, inject your API keys, and execute the deployment commands.
              </p>
            </div>
          </div>

          <DialogFooter className="gap-2 sm:gap-0">
            <Button variant="outline" onClick={handleSaveOnly}>
              Save Only
            </Button>
            <Button
              onClick={handleDeployNow}
              className="nvidia-gradient nvidia-glow"
              disabled={launchable.requiresApiKey && !hasApiKeys()}
            >
              <Rocket className="h-4 w-4 mr-2" />
              Deploy Now
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default LaunchableCard;
