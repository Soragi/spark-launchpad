import { useState, useMemo, useEffect } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import Layout from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import CodeBlock from "@/components/code/CodeBlock";
import { 
  ArrowLeft, 
  Clock, 
  ExternalLink, 
  Copy, 
  Check,
  Key,
  AlertTriangle,
  Rocket,
  Terminal,
  Info,
  Lightbulb,
  BookOpen
} from "lucide-react";
import { launchables } from "@/data/launchables";
import { getInstructions, type LaunchableInstructions } from "@/data/launchableInstructions";
import { copyToClipboard } from "@/lib/clipboard";
import { useToast } from "@/hooks/use-toast";

interface ApiKeys {
  ngcApiKey: string;
  hfToken: string;
}

const LaunchableDetails = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { toast } = useToast();
  
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  
  const [apiKeys, setApiKeys] = useState<ApiKeys>({ ngcApiKey: "", hfToken: "" });

  // Load saved API keys from localStorage
  useEffect(() => {
    const savedKeys = localStorage.getItem("dgx-spark-api-keys");
    if (savedKeys) {
      try {
        const parsed = JSON.parse(savedKeys);
        setApiKeys(parsed);
      } catch (e) {
        console.error("Failed to parse saved keys");
      }
    }
  }, []);

  const launchable = launchables.find(l => l.id === id);
  
  // Get static instructions from pre-scraped data
  const instructions: LaunchableInstructions | null = useMemo(() => {
    if (!id) return null;
    return getInstructions(id);
  }, [id]);

  // Substitute API keys into commands
  const substituteApiKeys = (command: string): string => {
    let result = command;
    if (apiKeys.ngcApiKey) {
      result = result.replace(/\$NGC_API_KEY/g, apiKeys.ngcApiKey);
      result = result.replace(/\$\{NGC_API_KEY\}/g, apiKeys.ngcApiKey);
    }
    if (apiKeys.hfToken) {
      result = result.replace(/\$HF_TOKEN/g, apiKeys.hfToken);
      result = result.replace(/\$\{HF_TOKEN\}/g, apiKeys.hfToken);
      result = result.replace(/<your_huggingface_token>/g, apiKeys.hfToken);
    }
    return result;
  };

  // Get all code blocks from steps
  const allCodeBlocks = useMemo(() => {
    if (!instructions?.steps) return [];
    return instructions.steps
      .filter(step => step.code)
      .map(step => step.code as string);
  }, [instructions]);

  const copyCommand = async (command: string, index: number) => {
    const substitutedCommand = substituteApiKeys(command);
    const success = await copyToClipboard(substitutedCommand);
    
    if (success) {
      setCopiedIndex(index);
      toast({
        title: "Copied!",
        description: apiKeys.ngcApiKey || apiKeys.hfToken 
          ? "Command copied with your API keys" 
          : "Command copied to clipboard",
      });
      setTimeout(() => setCopiedIndex(null), 2000);
    } else {
      toast({
        title: "Failed to copy",
        description: "Could not copy to clipboard. Try selecting the text manually.",
        variant: "destructive",
      });
    }
  };

  const copyAllCommands = async () => {
    if (!allCodeBlocks.length) return;
    
    const allCommands = allCodeBlocks.map(substituteApiKeys).join('\n\n');
    const success = await copyToClipboard(allCommands);
    
    if (success) {
      toast({
        title: "All commands copied!",
        description: apiKeys.ngcApiKey || apiKeys.hfToken 
          ? `${allCodeBlocks.length} commands copied with your API keys` 
          : `${allCodeBlocks.length} commands copied to clipboard`,
      });
    } else {
      toast({
        title: "Failed to copy",
        description: "Could not copy to clipboard. Try selecting the text manually.",
        variant: "destructive",
      });
    }
  };

  const handleOpenTerminal = async () => {
    // Copy commands and show instructions for opening terminal manually
    if (allCodeBlocks.length > 0) {
      const allCommands = allCodeBlocks.map(substituteApiKeys).join('\n\n');
      const success = await copyToClipboard(allCommands);
      
      if (success) {
        toast({
          title: "Commands copied to clipboard!",
          description: "Open a terminal on your DGX Spark and paste the commands to run them.",
        });
      } else {
        toast({
          title: "Open Terminal on DGX Spark",
          description: "Use SSH or the desktop environment to open a terminal, then copy the commands from this page.",
        });
      }
    } else {
      toast({
        title: "Open Terminal on DGX Spark",
        description: "Use SSH or the desktop environment to open a terminal on your DGX Spark system.",
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
          <div className="flex items-center gap-2">
            <Button 
              variant="outline" 
              onClick={handleOpenTerminal}
            >
              <Terminal className="h-4 w-4 mr-2" />
              Copy & Open Terminal
            </Button>
            <Button asChild className="nvidia-gradient nvidia-glow">
              <a href={launchable.url} target="_blank" rel="noopener noreferrer">
                <ExternalLink className="h-4 w-4 mr-2" />
                View on NVIDIA
              </a>
            </Button>
          </div>
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
            <p className="text-muted-foreground">
              {instructions?.overview || launchable.description}
            </p>
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-1 text-muted-foreground">
                <Clock className="h-4 w-4" />
                <span>Estimated time: {launchable.duration}</span>
              </div>
              {instructions?.steps && (
                <div className="flex items-center gap-1 text-muted-foreground">
                  <BookOpen className="h-4 w-4" />
                  <span>{instructions.steps.length} steps</span>
                </div>
              )}
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
              {instructions?.steps && (
                <Badge variant="secondary" className="ml-2">
                  {instructions.steps.length} steps
                </Badge>
              )}
            </CardTitle>
            {allCodeBlocks.length > 0 && (
              <Button variant="outline" size="sm" onClick={copyAllCommands}>
                <Copy className="h-4 w-4 mr-2" />
                Copy All
              </Button>
            )}
          </CardHeader>
          <CardContent>
            {instructions?.steps && instructions.steps.length > 0 ? (
              <div className="space-y-6">
                {/* Prerequisites */}
                {instructions.prerequisites && instructions.prerequisites.length > 0 && (
                  <div className="space-y-2 p-4 rounded-lg bg-muted/50 border border-border">
                    <h4 className="text-sm font-medium flex items-center gap-2">
                      <Info className="h-4 w-4 text-primary" />
                      Prerequisites
                    </h4>
                    <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1 ml-6">
                      {instructions.prerequisites.map((prereq, i) => (
                        <li key={i}>{prereq}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {/* Steps */}
                <ScrollArea className="h-[600px] pr-4">
                  <div className="space-y-8">
                    {instructions.steps.map((step, index) => (
                      <div key={index} className="group">
                        {/* Step header */}
                        <div className="flex items-start gap-3 mb-3">
                          <div className="flex items-center justify-center w-7 h-7 rounded-full bg-primary/20 text-primary text-sm font-medium shrink-0">
                            {index + 1}
                          </div>
                          <div className="flex-1">
                            <h3 className="font-medium text-foreground">{step.title}</h3>
                            <p className="text-sm text-muted-foreground mt-1">{step.description}</p>
                          </div>
                        </div>
                        
                        {/* Code block */}
                        {step.code && (
                          <div className="ml-10">
                            <div className="relative">
                              <Button
                                variant="ghost"
                                size="sm"
                                className="absolute top-2 right-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity h-7"
                                onClick={() => copyCommand(step.code as string, index)}
                              >
                                {copiedIndex === index ? (
                                  <Check className="h-3 w-3 text-green-500" />
                                ) : (
                                  <Copy className="h-3 w-3" />
                                )}
                              </Button>
                              <div className="border border-border rounded-lg overflow-hidden">
                                <CodeBlock code={step.code} language="bash" />
                              </div>
                            </div>
                            
                            {/* Note */}
                            {step.note && (
                              <div className="flex items-start gap-2 mt-2 text-sm text-muted-foreground">
                                <Lightbulb className="h-4 w-4 text-amber-500 mt-0.5 shrink-0" />
                                <span>{step.note}</span>
                              </div>
                            )}
                            
                            {/* Warning */}
                            {step.warning && (
                              <div className="flex items-start gap-2 mt-2 p-3 rounded-lg bg-red-500/10 border border-red-500/30">
                                <AlertTriangle className="h-4 w-4 text-red-500 mt-0.5 shrink-0" />
                                <span className="text-sm text-red-400">{step.warning}</span>
                              </div>
                            )}
                          </div>
                        )}
                        
                        {/* Note without code */}
                        {!step.code && step.note && (
                          <div className="ml-10 flex items-start gap-2 text-sm text-muted-foreground">
                            <Lightbulb className="h-4 w-4 text-amber-500 mt-0.5 shrink-0" />
                            <span>{step.note}</span>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </ScrollArea>
                
                {/* Next Steps */}
                {instructions.nextSteps && instructions.nextSteps.length > 0 && (
                  <div className="space-y-2 pt-4 border-t border-border">
                    <h4 className="text-sm font-medium text-muted-foreground">Next Steps</h4>
                    <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1">
                      {instructions.nextSteps.map((nextStep, i) => (
                        <li key={i}>{nextStep}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {/* Resources */}
                {instructions.resources && instructions.resources.length > 0 && (
                  <div className="space-y-2 pt-4 border-t border-border">
                    <h4 className="text-sm font-medium text-muted-foreground">Resources</h4>
                    <ul className="space-y-1">
                      {instructions.resources.map((resource, i) => (
                        <li key={i}>
                          <a 
                            href={resource.url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="text-sm text-primary hover:underline flex items-center gap-1"
                          >
                            <ExternalLink className="h-3 w-3" />
                            {resource.title}
                          </a>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-8 gap-4">
                <p className="text-muted-foreground">Instructions coming soon for this launchable.</p>
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
