import { useState } from "react";
import Layout from "@/components/layout/Layout";
import LaunchableCard, { Launchable } from "@/components/launchables/LaunchableCard";
import { launchables } from "@/data/launchables";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Search, Rocket, AlertCircle, Loader2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { Link } from "react-router-dom";
import { useDeployments } from "@/hooks/use-deployments";

const Launchables = () => {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedTab, setSelectedTab] = useState("all");
  const [deployDialog, setDeployDialog] = useState<Launchable | null>(null);
  const [isDeploying, setIsDeploying] = useState(false);
  const { toast } = useToast();
  const { deploy, stop, getDeploymentStatus } = useDeployments();

  const filteredLaunchables = launchables.filter((l) => {
    const matchesSearch =
      l.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      l.description.toLowerCase().includes(searchQuery.toLowerCase());

    if (selectedTab === "all") return matchesSearch;
    if (selectedTab === "quickstart") return matchesSearch && l.category === "quickstart";
    if (selectedTab === "new") return matchesSearch && l.category === "new";
    return matchesSearch;
  });

  const handleDeploy = (launchable: Launchable) => {
    const status = getDeploymentStatus(launchable.id);
    if (status?.status === "running") {
      // If running, stop it
      stop(launchable.id);
    } else {
      // If not running, show deploy dialog
      setDeployDialog(launchable);
    }
  };

  const confirmDeploy = async () => {
    if (!deployDialog) return;

    // Check if API key is required but not configured
    if (deployDialog.requiresApiKey) {
      const hfKey = localStorage.getItem("hf_api_key");
      if (!hfKey) {
        toast({
          title: "API Key Required",
          description: "Please configure your HuggingFace API key in Settings first.",
          variant: "destructive",
        });
        setDeployDialog(null);
        return;
      }
    }

    setIsDeploying(true);

    try {
      await deploy(deployDialog.id, deployDialog.requiresApiKey);
      setDeployDialog(null);
    } catch (error) {
      // Error already handled in hook
    } finally {
      setIsDeploying(false);
    }
  };

  const quickstarts = launchables.filter((l) => l.category === "quickstart");
  const newPlaybooks = launchables.filter((l) => l.category === "new");

  return (
    <Layout>
      <div className="space-y-8 animate-fade-in">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Rocket className="h-8 w-8 text-primary" />
            Launchables
          </h1>
          <p className="text-muted-foreground mt-2">
            Deploy pre-configured AI workflows with one click. Find instructions and examples to run AI workloads on your DGX Spark.
          </p>
        </div>

        {/* Search */}
        <div className="relative max-w-xl">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search by name, description, or category..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 bg-secondary border-border"
          />
        </div>

        {/* Tabs */}
        <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-6">
          <TabsList className="bg-secondary">
            <TabsTrigger value="all">All Playbooks</TabsTrigger>
            <TabsTrigger value="quickstart">Quickstarts</TabsTrigger>
            <TabsTrigger value="new">What's New</TabsTrigger>
          </TabsList>

          <TabsContent value="all" className="space-y-8">
            {searchQuery ? (
              <div>
                <h2 className="text-lg font-semibold mb-4">
                  Search Results ({filteredLaunchables.length})
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                  {filteredLaunchables.map((launchable) => (
                    <LaunchableCard
                      key={launchable.id}
                      launchable={launchable}
                      onDeploy={handleDeploy}
                      deploymentStatus={getDeploymentStatus(launchable.id)}
                    />
                  ))}
                </div>
              </div>
            ) : (
              <>
                {/* Quickstarts Section */}
                <div>
                  <h2 className="text-xl font-semibold mb-4">Developer Quickstarts</h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    {quickstarts.map((launchable) => (
                      <LaunchableCard
                        key={launchable.id}
                        launchable={launchable}
                        onDeploy={handleDeploy}
                        deploymentStatus={getDeploymentStatus(launchable.id)}
                      />
                    ))}
                  </div>
                </div>

                {/* What's New Section */}
                <div>
                  <h2 className="text-xl font-semibold mb-4">What's New</h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                    {newPlaybooks.map((launchable) => (
                      <LaunchableCard
                        key={launchable.id}
                        launchable={launchable}
                        onDeploy={handleDeploy}
                        deploymentStatus={getDeploymentStatus(launchable.id)}
                      />
                    ))}
                  </div>
                </div>

                {/* All Playbooks */}
                <div>
                  <h2 className="text-xl font-semibold mb-4">All Playbooks</h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                    {launchables.map((launchable) => (
                      <LaunchableCard
                        key={launchable.id}
                        launchable={launchable}
                        onDeploy={handleDeploy}
                        deploymentStatus={getDeploymentStatus(launchable.id)}
                      />
                    ))}
                  </div>
                </div>
              </>
            )}
          </TabsContent>

          <TabsContent value="quickstart">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {filteredLaunchables.map((launchable) => (
                <LaunchableCard
                  key={launchable.id}
                  launchable={launchable}
                  onDeploy={handleDeploy}
                  deploymentStatus={getDeploymentStatus(launchable.id)}
                />
              ))}
            </div>
          </TabsContent>

          <TabsContent value="new">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {filteredLaunchables.map((launchable) => (
                <LaunchableCard
                  key={launchable.id}
                  launchable={launchable}
                  onDeploy={handleDeploy}
                  deploymentStatus={getDeploymentStatus(launchable.id)}
                />
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </div>

      {/* Deploy Dialog */}
      <Dialog open={!!deployDialog} onOpenChange={() => setDeployDialog(null)}>
        <DialogContent className="bg-card border-border">
          <DialogHeader>
            <DialogTitle>Deploy {deployDialog?.title}</DialogTitle>
            <DialogDescription>
              {deployDialog?.description}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="flex items-center gap-2 text-sm">
              <span className="text-muted-foreground">Estimated time:</span>
              <span className="font-medium">{deployDialog?.duration}</span>
            </div>

            {deployDialog?.requiresApiKey && (
              <div className="flex items-start gap-2 p-3 bg-destructive/10 rounded-lg border border-destructive/20">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div className="text-sm">
                  <p className="font-medium text-destructive">API Key Required</p>
                  <p className="text-muted-foreground mt-1">
                    This deployment requires an NGC or HuggingFace API key.{" "}
                    <Link to="/settings" className="text-primary hover:underline">
                      Configure your keys
                    </Link>
                  </p>
                </div>
              </div>
            )}

            <div className="text-sm text-muted-foreground">
              This will execute the deployment script from the NVIDIA Spark playbook.
              You can view the full instructions at{" "}
              <a
                href={deployDialog?.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline"
              >
                build.nvidia.com
              </a>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setDeployDialog(null)}>
              Cancel
            </Button>
            <Button
              onClick={confirmDeploy}
              disabled={isDeploying}
              className="nvidia-gradient"
            >
              {isDeploying ? "Deploying..." : "Confirm Deploy"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Layout>
  );
};

export default Launchables;
