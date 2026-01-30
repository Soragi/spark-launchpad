import { useState, useEffect, useRef } from "react";
import { AlertTriangle, X, ExternalLink, Loader2 } from "lucide-react";
import { checkApiHealth } from "@/lib/api";
import { Button } from "@/components/ui/button";

const DevModeBanner = () => {
  const [isBackendAvailable, setIsBackendAvailable] = useState<boolean | null>(null);
  const [isDismissed, setIsDismissed] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const maxRetries = 10;
  const initialCheckDone = useRef(false);

  useEffect(() => {
    let timeoutId: NodeJS.Timeout;
    let intervalId: NodeJS.Timeout;

    const checkBackend = async () => {
      const healthy = await checkApiHealth();
      setIsBackendAvailable(healthy);
      
      if (!healthy && !initialCheckDone.current && retryCount < maxRetries) {
        // Retry quickly on startup (every 2 seconds for first 10 attempts)
        setRetryCount(prev => prev + 1);
        timeoutId = setTimeout(checkBackend, 2000);
      } else {
        initialCheckDone.current = true;
      }
    };

    checkBackend();
    
    // After initial checks, poll every 30 seconds
    const startPolling = setTimeout(() => {
      intervalId = setInterval(async () => {
        const healthy = await checkApiHealth();
        setIsBackendAvailable(healthy);
      }, 30000);
    }, maxRetries * 2000);

    return () => {
      clearTimeout(timeoutId);
      clearTimeout(startPolling);
      clearInterval(intervalId);
    };
  }, []);

  // Don't show if backend is available or if dismissed
  if (isBackendAvailable || isDismissed) {
    return null;
  }

  // Show loading state during initial connection attempts
  const isConnecting = isBackendAvailable === null || (retryCount > 0 && retryCount < maxRetries);

  return (
    <div className="bg-gradient-to-r from-amber-500/10 via-amber-500/5 to-amber-500/10 border-b border-amber-500/20 px-4 py-3">
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-8 h-8 rounded-full bg-amber-500/20">
            {isConnecting ? (
              <Loader2 className="h-4 w-4 text-amber-500 animate-spin" />
            ) : (
              <AlertTriangle className="h-4 w-4 text-amber-500" />
            )}
          </div>
          <div className="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-3">
            <span className="font-semibold text-amber-500 text-sm">
              {isConnecting ? "Connecting to Backend..." : "Development Mode"}
            </span>
            <span className="text-sm text-muted-foreground">
              {isConnecting 
                ? `Attempting to connect (${retryCount}/${maxRetries})...`
                : "Backend not connected. Deploy on DGX Spark for full functionality."
              }
            </span>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <a
            href="https://build.nvidia.com/spark"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden sm:flex"
          >
            <Button variant="outline" size="sm" className="text-xs border-amber-500/30 hover:bg-amber-500/10">
              <ExternalLink className="h-3 w-3 mr-1.5" />
              Learn More
            </Button>
          </a>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-muted-foreground hover:text-foreground"
            onClick={() => setIsDismissed(true)}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
};

export default DevModeBanner;
