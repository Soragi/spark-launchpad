import { useState, useEffect } from "react";
import Layout from "@/components/layout/Layout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Eye, EyeOff, Key, Save, Shield, Check } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface ApiKeys {
  ngcApiKey: string;
  hfToken: string;
}

const Settings = () => {
  const [apiKeys, setApiKeys] = useState<ApiKeys>({
    ngcApiKey: "",
    hfToken: "",
  });
  const [showNgcKey, setShowNgcKey] = useState(false);
  const [showHfToken, setShowHfToken] = useState(false);
  const [autoUpdate, setAutoUpdate] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const { toast } = useToast();

  // Load saved keys from localStorage on mount
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

  const handleSave = async () => {
    setIsSaving(true);

    // Save to localStorage
    localStorage.setItem("dgx-spark-api-keys", JSON.stringify(apiKeys));

    // Simulate saving
    await new Promise((resolve) => setTimeout(resolve, 1000));

    toast({
      title: "Settings Saved",
      description: "Your API keys have been securely saved.",
    });

    setIsSaving(false);
  };

  const maskKey = (key: string) => {
    if (!key) return "";
    if (key.length <= 8) return "•".repeat(key.length);
    return key.slice(0, 4) + "•".repeat(key.length - 8) + key.slice(-4);
  };

  return (
    <Layout>
      <div className="max-w-2xl mx-auto space-y-8 animate-fade-in">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Key className="h-8 w-8 text-primary" />
            Settings
          </h1>
          <p className="text-muted-foreground mt-2">
            Configure your API keys and preferences for DGX Spark deployments.
          </p>
        </div>

        {/* API Keys Card */}
        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-primary" />
              API Keys
            </CardTitle>
            <CardDescription>
              These keys are stored locally and used for deployments that require authentication.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* NGC API Key */}
            <div className="space-y-2">
              <Label htmlFor="ngc-key">NVIDIA NGC API Key</Label>
              <p className="text-xs text-muted-foreground">
                Required for accessing NGC containers and models.{" "}
                <a
                  href="https://ngc.nvidia.com/setup/api-key"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  Get your key →
                </a>
              </p>
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <Input
                    id="ngc-key"
                    type={showNgcKey ? "text" : "password"}
                    value={apiKeys.ngcApiKey}
                    onChange={(e) =>
                      setApiKeys({ ...apiKeys, ngcApiKey: e.target.value })
                    }
                    placeholder="nvapi-..."
                    className="font-mono bg-secondary border-border pr-10"
                  />
                  <button
                    type="button"
                    onClick={() => setShowNgcKey(!showNgcKey)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  >
                    {showNgcKey ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </button>
                </div>
                {apiKeys.ngcApiKey && (
                  <div className="flex items-center justify-center w-10 h-10 rounded-md bg-status-running/20">
                    <Check className="h-4 w-4 text-status-running" />
                  </div>
                )}
              </div>
            </div>

            {/* HuggingFace Token */}
            <div className="space-y-2">
              <Label htmlFor="hf-token">HuggingFace Token</Label>
              <p className="text-xs text-muted-foreground">
                Required for downloading models from HuggingFace Hub.{" "}
                <a
                  href="https://huggingface.co/settings/tokens"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  Get your token →
                </a>
              </p>
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <Input
                    id="hf-token"
                    type={showHfToken ? "text" : "password"}
                    value={apiKeys.hfToken}
                    onChange={(e) =>
                      setApiKeys({ ...apiKeys, hfToken: e.target.value })
                    }
                    placeholder="hf_..."
                    className="font-mono bg-secondary border-border pr-10"
                  />
                  <button
                    type="button"
                    onClick={() => setShowHfToken(!showHfToken)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  >
                    {showHfToken ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </button>
                </div>
                {apiKeys.hfToken && (
                  <div className="flex items-center justify-center w-10 h-10 rounded-md bg-status-running/20">
                    <Check className="h-4 w-4 text-status-running" />
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Preferences Card */}
        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle>Preferences</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label>Auto-update Containers</Label>
                <p className="text-xs text-muted-foreground">
                  Automatically pull latest container images before deployment
                </p>
              </div>
              <Switch checked={autoUpdate} onCheckedChange={setAutoUpdate} />
            </div>
          </CardContent>
        </Card>

        {/* Save Button */}
        <Button
          onClick={handleSave}
          disabled={isSaving}
          className="w-full nvidia-gradient nvidia-glow"
          size="lg"
        >
          {isSaving ? (
            "Saving..."
          ) : (
            <>
              <Save className="h-4 w-4 mr-2" />
              Save Settings
            </>
          )}
        </Button>

        {/* Security Note */}
        <p className="text-xs text-center text-muted-foreground">
          API keys are stored locally in your browser and are never sent to external servers.
          They are only used for direct communication with NVIDIA and HuggingFace APIs.
        </p>
      </div>
    </Layout>
  );
};

export default Settings;
