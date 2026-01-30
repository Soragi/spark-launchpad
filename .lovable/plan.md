

# Automated Launchable Deployment System

## Overview
This plan adds an automated deployment feature that fetches installation instructions from NVIDIA blueprint pages, extracts the shell commands, injects stored API keys, and executes them on the DGX Spark system when a user adds a launchable to deployments.

## Architecture

```text
+------------------+     +------------------+     +------------------+
|    Frontend      |     |    Backend       |     |   DGX Spark      |
|                  |     |    (FastAPI)     |     |   Host Shell     |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        | 1. User clicks         |                        |
        |    "Add to Deploy"     |                        |
        |----------------------->|                        |
        |    + API keys from     |                        |
        |    localStorage        |                        |
        |                        |                        |
        |                        | 2. Fetch instructions  |
        |                        |    from NVIDIA         |
        |                        |    build.nvidia.com    |
        |                        |                        |
        |                        | 3. Parse & extract     |
        |                        |    shell commands      |
        |                        |                        |
        |                        | 4. Inject API keys     |
        |                        |    (HF_TOKEN, NGC_KEY) |
        |                        |                        |
        |                        | 5. Execute commands--->|
        |                        |    via subprocess      |
        |                        |                        |
        | 6. Stream status/<-----|                        |
        |    logs via WebSocket  |                        |
        |                        |                        |
+------------------+     +------------------+     +------------------+
```

## Implementation Plan

### Phase 1: Backend - Instruction Fetching & Parsing

**File: `backend/app/main.py`**

Add new functionality to:

1. **Fetch Instructions Endpoint**: Create a new endpoint `/api/launchables/{id}/instructions` that:
   - Fetches the instructions page from `build.nvidia.com/spark/{id}/instructions`
   - Parses HTML to extract code blocks (shell commands)
   - Returns structured command list

2. **Automated Deploy Endpoint**: Create `/api/launchables/{id}/auto-deploy` that:
   - Accepts API keys (NGC, HuggingFace) in the request body
   - Fetches and parses instructions
   - Injects environment variables for API keys into commands
   - Executes commands sequentially via subprocess
   - Returns execution status

3. **Dependencies**: Add `beautifulsoup4` and `httpx` to `backend/requirements.txt` for HTML parsing and async HTTP requests

**Command Extraction Logic**:
```python
# Extract code blocks from NVIDIA instruction pages
# Pattern: Look for ```bash or ```shell code blocks
# Also handle inline Bash blocks with "CopiedCopy" markers
```

**API Key Injection**:
- Commands containing `docker run` will have `-e HF_TOKEN=xxx -e NGC_API_KEY=xxx` added
- Support for `export HF_TOKEN=...` style commands

### Phase 2: Frontend - Deploy Button Enhancement

**File: `src/components/launchables/LaunchableCard.tsx`**

Modify the "Add to Deployments" button to:
1. Show a deployment dialog with options:
   - "Save Only" - current behavior, just bookmarks
   - "Deploy Now" - triggers automated deployment
2. Check if required API keys are configured
3. Show warning if `requiresApiKey` is true but keys are missing

**File: `src/pages/Deployments.tsx`**

Add to the saved launchables cards:
- "Deploy" button that triggers automated deployment
- Deployment status indicator (pending/running/completed/failed)
- Log viewer for deployment output

### Phase 3: New Hook - useAutoDeploy

**File: `src/hooks/use-auto-deploy.ts`**

Create a hook that:
1. Calls the backend auto-deploy endpoint
2. Manages deployment state (idle/deploying/success/error)
3. Retrieves API keys from localStorage and passes them securely
4. Provides real-time feedback via polling or WebSocket

### Phase 4: Backend - Execution Engine

**File: `backend/app/main.py`**

Add robust command execution:

1. **Command Queue**: Execute commands sequentially with proper error handling
2. **Environment Setup**: Set `HF_TOKEN` and `NGC_API_KEY` environment variables
3. **Output Streaming**: Store stdout/stderr for frontend display
4. **Safety Checks**:
   - Only allow commands from trusted NVIDIA sources
   - Validate command patterns (docker, git, wget, curl, pip)
   - Timeout protection for long-running commands

### Phase 5: Deployment Status Tracking

**File: `backend/app/main.py`**

Add deployment job tracking:
- Store deployment jobs in memory (or SQLite for persistence)
- Track status: `pending` -> `running` -> `completed` | `failed`
- Store execution logs per job
- New endpoint: `GET /api/deployments/{id}/job-status`

## Data Models

### DeploymentJob
```python
class DeploymentJob(BaseModel):
    id: str
    launchable_id: str
    status: str  # pending, running, completed, failed
    commands: List[str]
    current_step: int
    total_steps: int
    logs: List[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error: Optional[str]
```

### AutoDeployRequest
```python
class AutoDeployRequest(BaseModel):
    launchable_id: str
    ngc_api_key: Optional[str]
    hf_api_key: Optional[str]
    dry_run: bool = False  # Preview commands without executing
```

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `backend/requirements.txt` | Modify | Add `beautifulsoup4`, `httpx` |
| `backend/app/main.py` | Modify | Add instruction fetching, parsing, auto-deploy endpoints |
| `src/hooks/use-auto-deploy.ts` | Create | New hook for automated deployment |
| `src/components/launchables/LaunchableCard.tsx` | Modify | Add deploy option to button |
| `src/pages/Deployments.tsx` | Modify | Add deploy button and status to saved cards |
| `src/lib/api.ts` | Modify | Add auto-deploy API functions |

## Security Considerations

1. **API Key Handling**: Keys are passed in request body (HTTPS in production), never logged
2. **Command Validation**: Only execute recognized safe commands (docker, git, pip, etc.)
3. **Source Validation**: Only fetch instructions from `build.nvidia.com` domain
4. **Timeout Protection**: Commands have execution timeout (configurable, default 30 minutes)
5. **User Confirmation**: Require explicit user action to trigger deployment

## User Flow

1. User navigates to Launchables page
2. User clicks "Add to Deployments" on a launchable
3. Dialog appears with options:
   - **Save Only**: Bookmark for later (current behavior)
   - **Deploy Now**: Start automated deployment
4. If "Deploy Now" and API keys required but missing → redirect to Settings
5. Deployment starts, user redirected to Deployments page
6. Deployment status shows progress with live log streaming
7. On completion, container appears in "Running" section

## Technical Details

### Instruction Page URL Pattern
```
https://build.nvidia.com/spark/{launchable_id}/instructions
```

### Command Extraction Regex
```python
# Match code blocks in markdown/HTML
code_block_pattern = r"```(?:bash|shell|sh)?\n([\s\S]*?)```"
```

### Environment Variable Injection
For `docker run` commands, insert before the image name:
```bash
-e HF_TOKEN=${HF_TOKEN} -e NGC_API_KEY=${NGC_API_KEY}
```

