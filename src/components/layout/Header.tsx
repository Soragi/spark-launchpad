import { NavLink } from "react-router-dom";
import { ExternalLink, Cpu } from "lucide-react";

const Header = () => {
  return (
    <header className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between px-4 md:px-8">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Cpu className="h-8 w-8 text-primary" />
            <span className="text-xl font-bold">
              <span className="text-primary">DGX</span> Spark
            </span>
          </div>
        </div>

        <nav className="flex items-center gap-1 md:gap-2">
          <NavLink
            to="/"
            className={({ isActive }) =>
              `px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                isActive
                  ? "text-foreground bg-secondary"
                  : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
              }`
            }
          >
            Home
          </NavLink>
          <NavLink
            to="/launchables"
            className={({ isActive }) =>
              `px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                isActive
                  ? "text-foreground bg-secondary"
                  : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
              }`
            }
          >
            Launchables
          </NavLink>
          <NavLink
            to="/settings"
            className={({ isActive }) =>
              `px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                isActive
                  ? "text-foreground bg-secondary"
                  : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
              }`
            }
          >
            Settings
          </NavLink>
          <a
            href="https://www.nvidia.com/en-us/support/dgx-spark/"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 px-3 py-2 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
          >
            Docs
            <ExternalLink className="h-3 w-3" />
          </a>
          <a
            href="https://forums.developer.nvidia.com/"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 px-3 py-2 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
          >
            Forums
            <ExternalLink className="h-3 w-3" />
          </a>
        </nav>
      </div>
    </header>
  );
};

export default Header;
