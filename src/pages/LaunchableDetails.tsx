import { useState, useEffect } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import Layout from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import CodeBlock from "@/components/code/CodeBlock";
import { 
  ArrowLeft, 
  Clock, 
  ExternalLink, 
  Copy, 
  Check,
  Key,
  AlertTriangle,
  Rocket
} from "lucide-react";
import { launchables } from "@/data/launchables";
import { fetchInstructions, type InstructionsResponse } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

const LaunchableDetails = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { toast } = useToast();
  
  const [instructions, setInstructions] = useState<InstructionsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const launchable = launchables.find(l => l.id === id);

  useEffect(() => {
    if (!id) return;

    const loadInstructions = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        const data = await fetchInstructions(id);
        setInstructions(data);
      } catch (err) {
        console.error("Failed to fetch instructions:", err);
        setError(err instanceof Error ? err.message : "Failed to fetch instructions");
      } finally {
        setIsLoading(false);
      }
    };

    loadInstructions();
  }, [id]);

  const copyCommand = async (command: string, index: number) => {
    try {
      await navigator.clipboard.writeText(command);
      setCopiedIndex(index);
      toast({
        title: "Copied!",
        description: "Command copied to clipboard",
      });
      setTimeout(() => setCopiedIndex(null), 2000);
    } catch (err) {
      toast({
        title: "Failed to copy",
        description: "Could not copy to clipboard",
        variant: "destructive",
      });
    }
  };

  const copyAllCommands = async () => {
    if (!instructions?.commands.length) return;
    
    try {
      const allCommands = instructions.commands.join('\n\n');
      await navigator.clipboard.writeText(allCommands);
      toast({
        title: "All commands copied!",
        description: `${instructions.commands.length} commands copied to clipboard`,
      });
    } catch (err) {
      toast({
        title: "Failed to copy",
        description: "Could not copy to clipboard",
        variant: "destructive",
      });
    }
  };

  if (!launchable) {
    return (
      <Layout>
        <div className="flex flex-col items-center justify-center min-h-[50vh] gap-4">
          <h1 className="text-2xl font-bold">Launchable Not Found</h1>
          <p className="text-muted-foreground">The requested launchable could not be found.</p>
          <Button onClick={() => navigate("/launchables")}>
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Launchables
          </Button>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center gap-4">
          <Button 
            variant="ghost" 
            size="icon"
            onClick={() => navigate(-1)}
          >
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
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
                  <Key className="h-3 w-3 mr-1" />
                  API Key Required
                </Badge>
              )}
            </div>
            <h1 className="text-2xl font-bold">{launchable.title}</h1>
          </div>
          <Button asChild className="nvidia-gradient nvidia-glow">
            <a href={launchable.url} target="_blank" rel="noopener noreferrer">
              <ExternalLink className="h-4 w-4 mr-2" />
              View on NVIDIA
            </a>
          </Button>
        </div>

        {/* Overview Card */}
        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Rocket className="h-5 w-5 text-primary" />
              Overview
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-muted-foreground">{launchable.description}</p>
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-1 text-muted-foreground">
                <Clock className="h-4 w-4" />
                <span>Estimated time: {launchable.duration}</span>
              </div>
            </div>
            
            {launchable.requiresApiKey && (
              <div className="flex items-start gap-3 p-4 rounded-lg bg-amber-500/10 border border-amber-500/30">
                <AlertTriangle className="h-5 w-5 text-amber-500 mt-0.5" />
                <div>
                  <p className="font-medium text-amber-500">API Key Required</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    This launchable requires API keys (HuggingFace or NGC). 
                    Make sure to configure them in{" "}
                    <Link to="/settings" className="text-primary hover:underline">
                      Settings
                    </Link>{" "}
                    before running the commands.
                  </p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Instructions Card */}
        <Card className="bg-card border-border">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              Deployment Instructions
              {instructions?.commands && (
                <Badge variant="secondary" className="ml-2">
                  {instructions.commands.length} commands
                </Badge>
              )}
            </CardTitle>
            {instructions?.commands && instructions.commands.length > 0 && (
              <Button variant="outline" size="sm" onClick={copyAllCommands}>
                <Copy className="h-4 w-4 mr-2" />
                Copy All
              </Button>
            )}
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="space-y-4">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="space-y-2">
                    <Skeleton className="h-4 w-24" />
                    <Skeleton className="h-16 w-full" />
                  </div>
                ))}
              </div>
            ) : error ? (
              <div className="flex flex-col items-center justify-center py-8 gap-4">
                <AlertTriangle className="h-12 w-12 text-destructive" />
                <div className="text-center">
                  <p className="font-medium text-destructive">Failed to load instructions</p>
                  <p className="text-sm text-muted-foreground mt-1">{error}</p>
                </div>
                <Button 
                  variant="outline" 
                  onClick={() => window.location.reload()}
                >
                  Try Again
                </Button>
              </div>
            ) : instructions?.commands && instructions.commands.length > 0 ? (
              <ScrollArea className="h-[500px] pr-4">
                <div className="space-y-4">
                  {instructions.commands.map((command, index) => (
                    <div key={index} className="group">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs text-muted-foreground font-mono">
                          Step {index + 1}
                        </span>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="opacity-0 group-hover:opacity-100 transition-opacity h-7"
                          onClick={() => copyCommand(command, index)}
                        >
                          {copiedIndex === index ? (
                            <Check className="h-3 w-3 text-green-500" />
                          ) : (
                            <Copy className="h-3 w-3" />
                          )}
                        </Button>
                      </div>
                      <div className="border border-border rounded-lg overflow-hidden">
                        <CodeBlock code={command} language="bash" />
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            ) : (
              <div className="flex flex-col items-center justify-center py-8 gap-4">
                <p className="text-muted-foreground">No commands found for this launchable.</p>
                <Button asChild variant="outline">
                  <a href={launchable.url} target="_blank" rel="noopener noreferrer">
                    <ExternalLink className="h-4 w-4 mr-2" />
                    View instructions on NVIDIA
                  </a>
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </Layout>
  );
};

export default LaunchableDetails;
