from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import subprocess
import json
import docker
import os
import asyncio
import random
import re
import uuid
import httpx
from datetime import datetime
from bs4 import BeautifulSoup

app = FastAPI(title="DGX Spark API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Docker client - lazy initialization for resilience
_docker_client = None

# In-memory job storage for deployment tracking
deployment_jobs = {}


def get_docker_client():
    """Get Docker client with lazy initialization and error handling"""
    global _docker_client
    if _docker_client is None:
        try:
            _docker_client = docker.from_env()
            # Test connection
            _docker_client.ping()
        except docker.errors.DockerException as e:
            raise HTTPException(
                status_code=503,
                detail=f"Docker is not available: {str(e)}. Ensure the Docker socket is mounted."
            )
    return _docker_client


class SystemStats(BaseModel):
    memory_used: float
    memory_total: float
    gpu_utilization: float
    gpu_memory_used: float
    gpu_memory_total: float
    gpu_name: str
    gpu_temperature: float
    cuda_version: str
    driver_version: str


class DeploymentRequest(BaseModel):
    launchable_id: str
    ngc_api_key: Optional[str] = None
    hf_api_key: Optional[str] = None


class DeploymentStatus(BaseModel):
    id: str
    name: str
    status: str
    container_id: Optional[str] = None
    ports: Optional[dict] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    uptime: Optional[str] = None
    image: Optional[str] = None


class ServiceAction(BaseModel):
    action: str  # start, stop, restart


class AutoDeployRequest(BaseModel):
    launchable_id: str
    ngc_api_key: Optional[str] = None
    hf_api_key: Optional[str] = None
    dry_run: bool = False


class DeploymentJob(BaseModel):
    id: str
    launchable_id: str
    status: str  # pending, running, completed, failed
    commands: List[str] = []
    current_step: int = 0
    total_steps: int = 0
    logs: List[str] = []
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class InstructionsResponse(BaseModel):
    launchable_id: str
    commands: List[str]
    raw_html: Optional[str] = None


# Launchable deployment configurations
LAUNCHABLE_CONFIGS = {
    "vscode": {
        "image": "codercom/code-server:latest",
        "name": "dgx-spark-vscode",
        "ports": {"8443/tcp": 8443},
        "environment": {},
        "volumes": {"/home": {"bind": "/home/coder/project", "mode": "rw"}},
    },
    "open-webui": {
        "image": "ghcr.io/open-webui/open-webui:main",
        "name": "dgx-spark-open-webui",
        "ports": {"8080/tcp": 3000},
        "environment": {"OLLAMA_BASE_URL": "http://host.docker.internal:11434"},
        "volumes": {"open-webui": {"bind": "/app/backend/data", "mode": "rw"}},
    },
    "comfy-ui": {
        "image": "ghcr.io/ai-dock/comfyui:latest",
        "name": "dgx-spark-comfyui",
        "ports": {"8188/tcp": 8188},
        "environment": {},
        "volumes": {"comfyui-data": {"bind": "/workspace", "mode": "rw"}},
        "runtime": "nvidia",
    },
    "jupyterlab": {
        "image": "jupyter/base-notebook:latest",
        "name": "dgx-spark-jupyter",
        "ports": {"8888/tcp": 8888},
        "environment": {"JUPYTER_ENABLE_LAB": "yes"},
        "volumes": {"/home": {"bind": "/home/jovyan/work", "mode": "rw"}},
    },
    "ollama": {
        "image": "ollama/ollama:latest",
        "name": "dgx-spark-ollama",
        "ports": {"11434/tcp": 11434},
        "environment": {},
        "volumes": {"ollama": {"bind": "/root/.ollama", "mode": "rw"}},
        "runtime": "nvidia",
    },
    "vllm": {
        "image": "vllm/vllm-openai:latest",
        "name": "dgx-spark-vllm",
        "ports": {"8000/tcp": 8000},
        "environment": {},
        "volumes": {"vllm-cache": {"bind": "/root/.cache", "mode": "rw"}},
        "runtime": "nvidia",
        "requires_api_key": True,
    },
    "sglang": {
        "image": "lmsysorg/sglang:latest",
        "name": "dgx-spark-sglang",
        "ports": {"30000/tcp": 30000},
        "environment": {},
        "volumes": {"sglang-cache": {"bind": "/root/.cache", "mode": "rw"}},
        "runtime": "nvidia",
        "requires_api_key": True,
    },
}

# Allowed command patterns for security
ALLOWED_COMMAND_PATTERNS = [
    r'^docker\s+',
    r'^git\s+',
    r'^wget\s+',
    r'^curl\s+',
    r'^pip\s+',
    r'^pip3\s+',
    r'^python\s+',
    r'^python3\s+',
    r'^chmod\s+',
    r'^mkdir\s+',
    r'^cd\s+',
    r'^export\s+',
    r'^echo\s+',
    r'^cat\s+',
    r'^sudo\s+docker\s+',
]


def is_command_allowed(command: str) -> bool:
    """Check if a command matches allowed patterns"""
    command = command.strip()
    for pattern in ALLOWED_COMMAND_PATTERNS:
        if re.match(pattern, command, re.IGNORECASE):
            return True
    return False


def extract_commands_from_html(html_content: str) -> List[str]:
    """Extract shell commands from HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    commands = []
    
    # Find code blocks - common patterns in documentation pages
    # Pattern 1: <pre><code> blocks
    for pre in soup.find_all('pre'):
        code = pre.get_text(strip=True)
        if code:
            # Split by newlines and filter empty lines
            for line in code.split('\n'):
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#') and is_command_allowed(line):
                    commands.append(line)
    
    # Pattern 2: code elements with specific classes
    for code in soup.find_all('code', class_=re.compile(r'(bash|shell|sh|language-bash|language-shell)')):
        text = code.get_text(strip=True)
        if text and is_command_allowed(text):
            commands.append(text)
    
    # Pattern 3: Look for divs with copy button indicators
    for div in soup.find_all(['div', 'span'], class_=re.compile(r'(copy|code|command)')):
        code_elem = div.find('code') or div.find('pre')
        if code_elem:
            text = code_elem.get_text(strip=True)
            if text and is_command_allowed(text):
                commands.append(text)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_commands = []
    for cmd in commands:
        if cmd not in seen:
            seen.add(cmd)
            unique_commands.append(cmd)
    
    return unique_commands


def inject_api_keys(command: str, ngc_api_key: Optional[str], hf_api_key: Optional[str]) -> str:
    """Inject API keys into docker run commands"""
    if not command.strip().startswith('docker run'):
        return command
    
    env_vars = []
    if hf_api_key:
        env_vars.append(f'-e HF_TOKEN={hf_api_key}')
        env_vars.append(f'-e HUGGING_FACE_HUB_TOKEN={hf_api_key}')
    if ngc_api_key:
        env_vars.append(f'-e NGC_API_KEY={ngc_api_key}')
    
    if not env_vars:
        return command
    
    # Find position to insert env vars (after 'docker run' and any existing flags before image name)
    # Insert after 'docker run'
    parts = command.split('docker run', 1)
    if len(parts) == 2:
        return f"docker run {' '.join(env_vars)} {parts[1].strip()}"
    
    return command


async def fetch_instructions(launchable_id: str) -> tuple[List[str], str]:
    """Fetch and parse instructions from NVIDIA build page"""
    url = f"https://build.nvidia.com/spark/{launchable_id}/instructions"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            html_content = response.text
            commands = extract_commands_from_html(html_content)
            return commands, html_content
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Failed to fetch instructions: {str(e)}"
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to connect to NVIDIA: {str(e)}"
            )


async def execute_command(command: str, job_id: str, env: dict) -> tuple[bool, str]:
    """Execute a shell command and return success status and output"""
    try:
        # Log the command being executed (without sensitive data)
        safe_cmd = re.sub(r'(HF_TOKEN|NGC_API_KEY|HUGGING_FACE_HUB_TOKEN)=[^\s]+', r'\1=***', command)
        deployment_jobs[job_id]["logs"].append(f"$ {safe_cmd}")
        
        # Run command with timeout
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **env}
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=1800  # 30 minute timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            return False, "Command timed out after 30 minutes"
        
        stdout_text = stdout.decode('utf-8', errors='replace')
        stderr_text = stderr.decode('utf-8', errors='replace')
        
        # Log output
        if stdout_text:
            for line in stdout_text.split('\n')[:50]:  # Limit log lines
                if line.strip():
                    deployment_jobs[job_id]["logs"].append(line)
        
        if process.returncode != 0:
            if stderr_text:
                for line in stderr_text.split('\n')[:20]:
                    if line.strip():
                        deployment_jobs[job_id]["logs"].append(f"ERROR: {line}")
            return False, stderr_text or f"Command failed with exit code {process.returncode}"
        
        return True, stdout_text
        
    except Exception as e:
        return False, str(e)


async def run_auto_deploy(job_id: str, commands: List[str], ngc_api_key: Optional[str], hf_api_key: Optional[str]):
    """Background task to run automated deployment"""
    job = deployment_jobs.get(job_id)
    if not job:
        return
    
    job["status"] = "running"
    job["started_at"] = datetime.now().isoformat()
    job["total_steps"] = len(commands)
    
    # Build environment
    env = {}
    if hf_api_key:
        env["HF_TOKEN"] = hf_api_key
        env["HUGGING_FACE_HUB_TOKEN"] = hf_api_key
    if ngc_api_key:
        env["NGC_API_KEY"] = ngc_api_key
    
    try:
        for i, command in enumerate(commands):
            job["current_step"] = i + 1
            job["logs"].append(f"\n--- Step {i + 1}/{len(commands)} ---")
            
            # Inject API keys into docker commands
            modified_command = inject_api_keys(command, ngc_api_key, hf_api_key)
            
            success, output = await execute_command(modified_command, job_id, env)
            
            if not success:
                job["status"] = "failed"
                job["error"] = output
                job["completed_at"] = datetime.now().isoformat()
                return
        
        job["status"] = "completed"
        job["completed_at"] = datetime.now().isoformat()
        job["logs"].append("\n✓ Deployment completed successfully!")
        
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = datetime.now().isoformat()


def run_nvidia_smi() -> dict:
    """Run nvidia-smi and parse JSON output"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise Exception(f"nvidia-smi failed: {result.stderr}")

        lines = result.stdout.strip().split("\n")
        if lines:
            parts = [p.strip() for p in lines[0].split(",")]
            return {
                "gpu_name": parts[0],
                "gpu_memory_used": float(parts[1]),
                "gpu_memory_total": float(parts[2]),
                "gpu_utilization": float(parts[3]),
                "gpu_temperature": float(parts[4]),
                "driver_version": parts[5],
            }
    except FileNotFoundError:
        # nvidia-smi not available, return simulated data for development
        # Add variation to make it clear this is simulated
        base_gpu_util = random.uniform(0, 5)  # Idle GPU shows near 0%
        base_mem_used = random.uniform(0.5, 2.0)  # Minimal memory usage
        base_temp = random.uniform(35, 42)  # Cool idle temperature
        
        return {
            "gpu_name": "Simulated GPU (Development Mode)",
            "gpu_memory_used": round(base_mem_used, 2),
            "gpu_memory_total": 128.0,
            "gpu_utilization": round(base_gpu_util, 1),
            "gpu_temperature": round(base_temp, 0),
            "driver_version": "dev-mode",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_system_memory() -> tuple:
    """Get system memory usage"""
    try:
        with open("/proc/meminfo", "r") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(":")] = int(parts[1])

            total_kb = meminfo.get("MemTotal", 0)
            available_kb = meminfo.get("MemAvailable", 0)
            used_kb = total_kb - available_kb

            # Convert to GB
            total_gb = total_kb / 1024 / 1024
            used_gb = used_kb / 1024 / 1024
            return used_gb, total_gb
    except Exception:
        # Return mock data for development
        return 24.5, 128.0


def get_cuda_version() -> str:
    """Get CUDA version from nvidia-smi"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Try to get CUDA version from nvcc
        nvcc_result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if "release" in nvcc_result.stdout:
            for line in nvcc_result.stdout.split("\n"):
                if "release" in line:
                    return line.split("release")[-1].split(",")[0].strip()
        return "12.4"
    except Exception:
        return "12.4"


@app.get("/api/system/stats", response_model=SystemStats)
async def get_system_stats():
    """Get real-time system statistics"""
    gpu_info = run_nvidia_smi()
    memory_used, memory_total = get_system_memory()
    cuda_version = get_cuda_version()

    return SystemStats(
        memory_used=memory_used,
        memory_total=memory_total,
        gpu_utilization=gpu_info["gpu_utilization"],
        gpu_memory_used=gpu_info["gpu_memory_used"],
        gpu_memory_total=gpu_info["gpu_memory_total"],
        gpu_name=gpu_info["gpu_name"],
        gpu_temperature=gpu_info["gpu_temperature"],
        cuda_version=cuda_version,
        driver_version=gpu_info["driver_version"],
    )


def calculate_uptime(started_at: str) -> str:
    """Calculate human-readable uptime from container start time"""
    try:
        # Parse ISO format datetime
        start_time = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        now = datetime.now(start_time.tzinfo)
        delta = now - start_time
        
        days = delta.days
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    except Exception:
        return "unknown"


@app.get("/api/deployments", response_model=List[DeploymentStatus])
async def list_deployments():
    """List all DGX Spark deployments"""
    deployments = []
    try:
        docker_client = get_docker_client()
        containers = docker_client.containers.list(all=True)
        for container in containers:
            if container.name.startswith("dgx-spark-"):
                ports = {}
                if container.attrs.get("NetworkSettings", {}).get("Ports"):
                    for port, bindings in container.attrs["NetworkSettings"]["Ports"].items():
                        if bindings:
                            ports[port] = bindings[0].get("HostPort")

                # Get container details
                state = container.attrs.get("State", {})
                started_at = state.get("StartedAt", "")
                uptime = calculate_uptime(started_at) if container.status == "running" else None
                
                deployments.append(
                    DeploymentStatus(
                        id=container.name.replace("dgx-spark-", ""),
                        name=container.name,
                        status=container.status,
                        container_id=container.short_id,
                        ports=ports,
                        created_at=started_at,
                        uptime=uptime,
                        image=container.image.tags[0] if container.image.tags else container.image.short_id,
                    )
                )
    except docker.errors.DockerException as e:
        raise HTTPException(status_code=500, detail=f"Docker error: {str(e)}")

    return deployments


@app.post("/api/deployments/{launchable_id}", response_model=DeploymentStatus)
async def deploy_launchable(launchable_id: str, request: DeploymentRequest):
    """Deploy a launchable container"""
    if launchable_id not in LAUNCHABLE_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Launchable '{launchable_id}' not found")

    config = LAUNCHABLE_CONFIGS[launchable_id]

    # Check if API key is required
    if config.get("requires_api_key") and not request.hf_api_key:
        raise HTTPException(
            status_code=400, detail="HuggingFace API key required for this launchable"
        )

    try:
        docker_client = get_docker_client()
        # Check if container already exists
        try:
            existing = docker_client.containers.get(config["name"])
            existing.remove(force=True)
        except docker.errors.NotFound:
            pass

        # Build environment variables
        env = dict(config.get("environment", {}))
        if request.ngc_api_key:
            env["NGC_API_KEY"] = request.ngc_api_key
        if request.hf_api_key:
            env["HF_TOKEN"] = request.hf_api_key
            env["HUGGING_FACE_HUB_TOKEN"] = request.hf_api_key

        # Container run kwargs
        run_kwargs = {
            "image": config["image"],
            "name": config["name"],
            "ports": config["ports"],
            "environment": env,
            "detach": True,
            "restart_policy": {"Name": "unless-stopped"},
        }

        # Add GPU runtime if specified
        if config.get("runtime") == "nvidia":
            run_kwargs["runtime"] = "nvidia"
            run_kwargs["device_requests"] = [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]

        # Add volumes
        if config.get("volumes"):
            run_kwargs["volumes"] = config["volumes"]

        # Pull image first
        docker_client.images.pull(config["image"])

        # Run container
        container = docker_client.containers.run(**run_kwargs)

        return DeploymentStatus(
            id=launchable_id,
            name=config["name"],
            status="running",
            container_id=container.short_id,
            ports=config["ports"],
        )

    except docker.errors.ImageNotFound:
        raise HTTPException(status_code=404, detail=f"Docker image not found: {config['image']}")
    except docker.errors.DockerException as e:
        raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")


@app.delete("/api/deployments/{launchable_id}")
async def stop_deployment(launchable_id: str):
    """Stop and remove a deployment"""
    container_name = f"dgx-spark-{launchable_id}"
    try:
        docker_client = get_docker_client()
        container = docker_client.containers.get(container_name)
        container.stop(timeout=10)
        container.remove()
        return {"status": "stopped", "id": launchable_id}
    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail=f"Deployment '{launchable_id}' not found")
    except docker.errors.DockerException as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop deployment: {str(e)}")


@app.post("/api/services/jupyterlab", response_model=DeploymentStatus)
async def manage_jupyterlab(action: ServiceAction):
    """Manage JupyterLab service"""
    container_name = "dgx-spark-jupyter"

    try:
        if action.action == "start":
            return await deploy_launchable("jupyterlab", DeploymentRequest(launchable_id="jupyterlab"))
        elif action.action == "stop":
            docker_client = get_docker_client()
            container = docker_client.containers.get(container_name)
            container.stop(timeout=10)
            return DeploymentStatus(
                id="jupyterlab",
                name=container_name,
                status="stopped",
            )
        elif action.action == "restart":
            docker_client = get_docker_client()
            container = docker_client.containers.get(container_name)
            container.restart(timeout=10)
            return DeploymentStatus(
                id="jupyterlab",
                name=container_name,
                status="running",
                container_id=container.short_id,
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action.action}")
    except docker.errors.NotFound:
        if action.action == "stop":
            return DeploymentStatus(id="jupyterlab", name=container_name, status="stopped")
        raise HTTPException(status_code=404, detail="JupyterLab container not found")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "dgx-spark-api"}


@app.get("/api/deployments/{launchable_id}/logs")
async def get_container_logs(launchable_id: str, tail: int = 100):
    """Get recent logs from a container"""
    container_name = f"dgx-spark-{launchable_id}"
    try:
        docker_client = get_docker_client()
        container = docker_client.containers.get(container_name)
        logs = container.logs(tail=tail, timestamps=True).decode("utf-8")
        return {"logs": logs.split("\n")}
    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail=f"Container '{launchable_id}' not found")
    except docker.errors.DockerException as e:
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")


# ============ Auto-Deploy Endpoints ============

@app.get("/api/launchables/{launchable_id}/instructions", response_model=InstructionsResponse)
async def get_launchable_instructions(launchable_id: str):
    """Fetch and parse installation instructions from NVIDIA"""
    commands, html_content = await fetch_instructions(launchable_id)
    
    return InstructionsResponse(
        launchable_id=launchable_id,
        commands=commands,
        raw_html=html_content[:5000] if html_content else None  # Limit raw HTML size
    )


@app.post("/api/launchables/{launchable_id}/auto-deploy", response_model=DeploymentJob)
async def auto_deploy_launchable(
    launchable_id: str,
    request: AutoDeployRequest,
    background_tasks: BackgroundTasks
):
    """Start automated deployment from NVIDIA instructions"""
    # Fetch instructions
    commands, _ = await fetch_instructions(launchable_id)
    
    if not commands:
        raise HTTPException(
            status_code=400,
            detail="No executable commands found in instructions"
        )
    
    # Create job
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "launchable_id": launchable_id,
        "status": "pending",
        "commands": commands,
        "current_step": 0,
        "total_steps": len(commands),
        "logs": [f"Starting automated deployment for {launchable_id}..."],
        "started_at": None,
        "completed_at": None,
        "error": None,
    }
    
    deployment_jobs[job_id] = job
    
    if request.dry_run:
        job["status"] = "dry_run"
        job["logs"].append("Dry run mode - commands will not be executed")
        job["logs"].extend([f"Would execute: {cmd}" for cmd in commands])
        return DeploymentJob(**job)
    
    # Start background deployment
    background_tasks.add_task(
        run_auto_deploy,
        job_id,
        commands,
        request.ngc_api_key,
        request.hf_api_key
    )
    
    return DeploymentJob(**job)


@app.get("/api/deployments/jobs/{job_id}", response_model=DeploymentJob)
async def get_deployment_job(job_id: str):
    """Get status of a deployment job"""
    job = deployment_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return DeploymentJob(**job)


@app.get("/api/deployments/jobs", response_model=List[DeploymentJob])
async def list_deployment_jobs():
    """List all deployment jobs"""
    return [DeploymentJob(**job) for job in deployment_jobs.values()]


@app.websocket("/ws/logs/{launchable_id}")
async def websocket_logs(websocket: WebSocket, launchable_id: str):
    """Stream container logs via WebSocket"""
    await websocket.accept()
    
    container_name = f"dgx-spark-{launchable_id}"
    
    try:
        docker_client = get_docker_client()
        container = docker_client.containers.get(container_name)
        
        # Send initial logs
        initial_logs = container.logs(tail=50, timestamps=True).decode("utf-8")
        for line in initial_logs.split("\n"):
            if line.strip():
                await websocket.send_json({"type": "log", "message": line})
        
        # Stream new logs
        log_stream = container.logs(stream=True, follow=True, timestamps=True, since=datetime.now())
        
        for log_line in log_stream:
            try:
                line = log_line.decode("utf-8").strip()
                if line:
                    await websocket.send_json({"type": "log", "message": line})
            except WebSocketDisconnect:
                break
            except Exception:
                continue
                
    except docker.errors.NotFound:
        await websocket.send_json({"type": "error", "message": f"Container '{launchable_id}' not found"})
        await websocket.close()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close()


@app.websocket("/ws/jobs/{job_id}")
async def websocket_job_logs(websocket: WebSocket, job_id: str):
    """Stream deployment job logs via WebSocket"""
    await websocket.accept()
    
    job = deployment_jobs.get(job_id)
    if not job:
        await websocket.send_json({"type": "error", "message": f"Job '{job_id}' not found"})
        await websocket.close()
        return
    
    last_log_index = 0
    
    try:
        while True:
            job = deployment_jobs.get(job_id)
            if not job:
                break
            
            # Send new logs
            current_logs = job.get("logs", [])
            if len(current_logs) > last_log_index:
                for log in current_logs[last_log_index:]:
                    await websocket.send_json({
                        "type": "log",
                        "message": log,
                        "status": job["status"],
                        "current_step": job["current_step"],
                        "total_steps": job["total_steps"]
                    })
                last_log_index = len(current_logs)
            
            # Check if job is complete
            if job["status"] in ["completed", "failed", "dry_run"]:
                await websocket.send_json({
                    "type": "complete",
                    "status": job["status"],
                    "error": job.get("error")
                })
                break
            
            await asyncio.sleep(0.5)
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close()
