
# DGX Spark WebUI - Custom Launch Script

## Overview

This plan creates a standalone bash script that can be added to DGX Spark's custom scripts launcher. The script will handle:
- Checking for Docker availability
- Cloning or updating the WebUI repository
- Building and starting the Docker containers
- Providing status feedback to the user

## Script Behavior

The script follows the same pattern as the Live VLM WebUI example:
- Uses `set -euo pipefail` for safety
- Configurable via environment variables (PORT)
- Handles cleanup on interrupt signals
- Runs in foreground for visibility
- Checks prerequisites before starting

## Files to Create

### 1. `launch-dgx-spark-webui.sh`

A self-contained bash script with the following sections:

```text
+------------------------------------------+
|  Configuration Variables                  |
|  - PORT (default: 8080)                   |
|  - REPO_URL                               |
|  - INSTALL_DIR                            |
+------------------------------------------+
           |
           v
+------------------------------------------+
|  Cleanup Handler                          |
|  - Stop containers on exit                |
|  - Clean shutdown                         |
+------------------------------------------+
           |
           v
+------------------------------------------+
|  Prerequisite Checks                      |
|  - Ensure Docker is installed             |
|  - Ensure Docker Compose is available     |
|  - Ensure Docker daemon is running        |
+------------------------------------------+
           |
           v
+------------------------------------------+
|  Clone/Update Repository                  |
|  - Clone if not exists                    |
|  - Git pull if exists                     |
+------------------------------------------+
           |
           v
+------------------------------------------+
|  Build and Start                          |
|  - docker compose build                   |
|  - docker compose up (foreground)         |
+------------------------------------------+
```

## Script Contents

The script will include:

1. **Shebang and strict mode**
   - `#!/usr/bin/env bash`
   - `set -euo pipefail`

2. **Configuration**
   - `PORT="${PORT:-8080}"` - configurable frontend port
   - `REPO_URL` - GitHub repository URL
   - `INSTALL_DIR` - where to clone the repo (~/.dgx-spark-webui)

3. **Cleanup trap**
   - Catches INT, TERM, HUP, QUIT signals
   - Runs `docker compose down` on exit

4. **Docker checks**
   - Verify `docker` command exists
   - Verify Docker daemon is running
   - Verify `docker compose` is available

5. **Repository management**
   - Clone if directory doesn't exist
   - Pull latest changes if it does

6. **Port configuration**
   - Modify docker-compose.yml to use custom port if specified

7. **Build and run**
   - `docker compose build` to build images
   - `docker compose up` (without -d) to run in foreground

## Technical Details

### Port Handling
The script will use `sed` to dynamically update the port mapping in docker-compose.yml if a custom port is specified:
```bash
sed -i "s/8080:80/${PORT}:80/" docker-compose.yml
```

### Repository URL
Will use the GitHub repository URL (to be published). For now, a placeholder that can be updated.

### Foreground Execution
Like the example script, uses `exec docker compose up` to replace the shell process, ensuring proper signal handling.

### Dependencies
- Docker (with Docker Compose v2 plugin)
- Git (for cloning/updating)
- curl or wget (optional, for Docker install)

## README Update

The README.md will be updated to include instructions for using the launch script:

1. Download the script
2. Make it executable
3. Add to DGX Spark custom scripts
4. Run from the launcher

## Summary of Changes

| File | Action | Description |
|------|--------|-------------|
| `launch-dgx-spark-webui.sh` | Create | Main launcher script for DGX Spark custom scripts |
| `README.md` | Update | Add section about using the launcher script |
