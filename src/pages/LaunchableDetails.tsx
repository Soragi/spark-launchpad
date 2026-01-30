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
  FileCode,
  Info
} from "lucide-react";
import { launchables } from "@/data/launchables";
import { getInstructions, type LaunchableInstructions } from "@/data/launchableInstructions";
import { openTerminal } from "@/lib/api";
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
  const [isOpeningTerminal, setIsOpeningTerminal] = useState(false);
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
    }
    return result;
  };

  const copyCommand = async (command: string, index: number) => {
    try {
      const substitutedCommand = substituteApiKeys(command);
      await navigator.clipboard.writeText(substitutedCommand);
      setCopiedIndex(index);
      toast({
        title: "Copied!",
        description: apiKeys.ngcApiKey || apiKeys.hfToken 
          ? "Command copied with your API keys" 
          : "Command copied to clipboard",
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
      const allCommands = instructions.commands.map(substituteApiKeys).join('\n\n');
      await navigator.clipboard.writeText(allCommands);
      toast({
        title: "All commands copied!",
        description: apiKeys.ngcApiKey || apiKeys.hfToken 
          ? `${instructions.commands.length} commands copied with your API keys` 
          : `${instructions.commands.length} commands copied to clipboard`,
      });
    } catch (err) {
      toast({
        title: "Failed to copy",
        description: "Could not copy to clipboard",
        variant: "destructive",
      });
    }
  };

  const copyAsScript = async () => {
    if (!instructions?.commands.length || !launchable) return;
    
    const substitutedCommands = instructions.commands.map(substituteApiKeys);
    
    const scriptContent = `#!/usr/bin/env bash

# =============================================================================
# ${launchable.title}
# =============================================================================
# Generated from DGX Spark Launchpad
# ${launchable.description}
# =============================================================================

set -euo pipefail

echo "Starting deployment: ${launchable.title}"
echo "============================================="

${substitutedCommands.map((cmd, i) => `# Step ${i + 1}\necho "Running step ${i + 1}..."\n${cmd}`).join('\n\n')}

echo ""
echo "============================================="
echo "Deployment complete!"
`;
    
    try {
      await navigator.clipboard.writeText(scriptContent);
      toast({
        title: "Script copied!",
        description: apiKeys.ngcApiKey || apiKeys.hfToken 
          ? "Shell script copied with your API keys. Save as .sh file and run with bash." 
          : "Shell script copied to clipboard. Save as .sh file and run with bash.",
      });
    } catch (err) {
      toast({
        title: "Failed to copy",
        description: "Could not copy to clipboard",
        variant: "destructive",
      });
    }
  };

  const handleOpenTerminal = async () => {
    setIsOpeningTerminal(true);
    try {
      await openTerminal();
      toast({
        title: "Terminal Opened",
        description: "A terminal window has been opened on your DGX Spark.",
      });
    } catch (err) {
      toast({
        title: "Failed to open terminal",
        description: err instanceof Error ? err.message : "Could not open terminal",
        variant: "destructive",
      });
    } finally {
      setIsOpeningTerminal(false);
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
              disabled={isOpeningTerminal}
            >
              <Terminal className="h-4 w-4 mr-2" />
              {isOpeningTerminal ? "Opening..." : "Open Terminal"}
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
                  {instructions.commands.length} steps
                </Badge>
              )}
            </CardTitle>
            {instructions?.commands && instructions.commands.length > 0 && (
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={copyAsScript}>
                  <FileCode className="h-4 w-4 mr-2" />
                  Copy as Script
                </Button>
                <Button variant="outline" size="sm" onClick={copyAllCommands}>
                  <Copy className="h-4 w-4 mr-2" />
                  Copy All
                </Button>
              </div>
            )}
          </CardHeader>
          <CardContent>
            {instructions?.commands && instructions.commands.length > 0 ? (
              <div className="space-y-6">
                {/* Overview from instructions */}
                {instructions.overview && (
                  <div className="flex items-start gap-3 p-4 rounded-lg bg-muted/50 border border-border">
                    <Info className="h-5 w-5 text-primary mt-0.5 shrink-0" />
                    <p className="text-sm text-muted-foreground">{instructions.overview}</p>
                  </div>
                )}
                
                {/* Prerequisites */}
                {instructions.prerequisites && instructions.prerequisites.length > 0 && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium text-muted-foreground">Prerequisites</h4>
                    <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1">
                      {instructions.prerequisites.map((prereq, i) => (
                        <li key={i}>{prereq}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {/* Commands */}
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
                
                {/* Notes */}
                {instructions.notes && instructions.notes.length > 0 && (
                  <div className="space-y-2 pt-4 border-t border-border">
                    <h4 className="text-sm font-medium text-muted-foreground">Notes</h4>
                    <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1">
                      {instructions.notes.map((note, i) => (
                        <li key={i}>{note}</li>
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
