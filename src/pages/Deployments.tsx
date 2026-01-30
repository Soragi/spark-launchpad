import { useNavigate } from "react-router-dom";
import Layout from "@/components/layout/Layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useSavedLaunchables } from "@/hooks/use-saved-launchables";
import {
  Bookmark,
  Clock,
  ExternalLink,
  Trash2,
  Rocket,
} from "lucide-react";

const Deployments = () => {
  const navigate = useNavigate();
  const { savedLaunchables, removeLaunchable } = useSavedLaunchables();

  return (
    <Layout>
      <div className="space-y-8 animate-fade-in">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Bookmark className="h-8 w-8 text-primary" />
            Saved Launchables
          </h1>
          <p className="text-muted-foreground mt-2">
            Your saved launchables for quick access. Click Deploy to view instructions.
          </p>
        </div>

        {/* Saved Launchables */}
        {savedLaunchables.length === 0 ? (
          <Card className="bg-card border-border">
            <CardContent className="py-12 text-center">
              <Bookmark className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium mb-2">No Saved Launchables</h3>
              <p className="text-muted-foreground mb-4">
                Save launchables by clicking the + button on the Launchables page.
              </p>
              <Button onClick={() => navigate("/launchables")}>
                <Rocket className="h-4 w-4 mr-2" />
                Browse Launchables
              </Button>
            </CardContent>
          </Card>
        ) : (
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
                  {launchable.requiresApiKey && (
                    <Badge variant="outline" className="text-xs">
                      API Key Required
                    </Badge>
                  )}
                  <div className="flex gap-2 pt-2 border-t border-border">
                    <Button
                      size="sm"
                      className="flex-1 nvidia-gradient nvidia-glow"
                      onClick={() => navigate(`/launchables/${launchable.id}`)}
                    >
                      <Rocket className="h-4 w-4 mr-1" />
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
                      title="Remove from saved"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </Layout>
  );
};

export default Deployments;
