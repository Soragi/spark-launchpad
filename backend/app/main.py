from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import subprocess
import json
import docker
import os
import asyncio
from datetime import datetime

app = FastAPI(title="DGX Spark API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Docker client - connects to host Docker via socket
docker_client = docker.from_env()


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
        # nvidia-smi not available, return mock data for development
        return {
            "gpu_name": "NVIDIA GB10 (Development Mode)",
            "gpu_memory_used": 12.5,
            "gpu_memory_total": 128.0,
            "gpu_utilization": 15.0,
            "gpu_temperature": 45.0,
            "driver_version": "550.54.14",
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
            container = docker_client.containers.get(container_name)
            container.stop(timeout=10)
            return DeploymentStatus(
                id="jupyterlab",
                name=container_name,
                status="stopped",
            )
        elif action.action == "restart":
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
        container = docker_client.containers.get(container_name)
        logs = container.logs(tail=tail, timestamps=True).decode("utf-8")
        return {"logs": logs.split("\n")}
    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail=f"Container '{launchable_id}' not found")
    except docker.errors.DockerException as e:
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")


@app.websocket("/ws/logs/{launchable_id}")
async def websocket_logs(websocket: WebSocket, launchable_id: str):
    """Stream container logs via WebSocket"""
    await websocket.accept()
    
    container_name = f"dgx-spark-{launchable_id}"
    
    try:
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
