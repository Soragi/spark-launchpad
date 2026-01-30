import { useState } from "react";
import Layout from "@/components/layout/Layout";
import LaunchableCard, { Launchable } from "@/components/launchables/LaunchableCard";
import { launchables } from "@/data/launchables";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Search, Rocket } from "lucide-react";
import { useSavedLaunchables } from "@/hooks/use-saved-launchables";
import { useToast } from "@/hooks/use-toast";

const Launchables = () => {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedTab, setSelectedTab] = useState("all");
  const { addLaunchable, isLaunchableSaved } = useSavedLaunchables();
  const { toast } = useToast();

  const handleAddToDeployments = (launchable: Launchable) => {
    addLaunchable(launchable);
    toast({
      title: "Saved",
      description: `${launchable.title} has been saved to your list.`,
    });
  };

  const filteredLaunchables = launchables.filter((l) => {
    const matchesSearch =
      l.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      l.description.toLowerCase().includes(searchQuery.toLowerCase());

    if (selectedTab === "all") return matchesSearch;
    if (selectedTab === "quickstart") return matchesSearch && l.category === "quickstart";
    if (selectedTab === "new") return matchesSearch && l.category === "new";
    return matchesSearch;
  });

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
                      onAddToDeployments={handleAddToDeployments}
                      isSaved={isLaunchableSaved(launchable.id)}
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
                        onAddToDeployments={handleAddToDeployments}
                        isSaved={isLaunchableSaved(launchable.id)}
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
                        onAddToDeployments={handleAddToDeployments}
                        isSaved={isLaunchableSaved(launchable.id)}
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
                        onAddToDeployments={handleAddToDeployments}
                        isSaved={isLaunchableSaved(launchable.id)}
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
                  onAddToDeployments={handleAddToDeployments}
                  isSaved={isLaunchableSaved(launchable.id)}
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
                  onAddToDeployments={handleAddToDeployments}
                  isSaved={isLaunchableSaved(launchable.id)}
                />
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </Layout>
  );
};

export default Launchables;
