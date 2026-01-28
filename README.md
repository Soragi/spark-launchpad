# DGX Spark WebUI

A modern web-based dashboard for managing your NVIDIA DGX Spark system. Monitor system resources in real-time with `nvidia-smi`, deploy AI workloads with one click, and manage configurations.

![DGX Spark Dashboard](https://build.nvidia.com/_next/image?url=https%3A%2F%2Fassets.ngc.nvidia.com%2Fproducts%2Fapi-catalog%2Fspark%2Fdgx-spark-hero.jpg&w=1920&q=75)

## Features

- **📊 Real-time Monitoring**: GPU utilization, memory usage, and temperature from `nvidia-smi`
- **🚀 Launchables**: One-click deployments for 30+ NVIDIA Spark playbooks
- **🐳 Docker Deployments**: Deploy containers as siblings on your DGX Spark
- **🔑 API Key Management**: Securely store NGC and HuggingFace tokens
- **🎛️ JupyterLab Control**: Start/stop JupyterLab from the dashboard

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA Container Toolkit (for GPU-accelerated deployments)
- Access to Docker socket (for managing sibling containers)

### Option 1: Docker Compose (Recommended)

```bash
# Clone or download the project
cd dgx-spark-ui

# Build and run all services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

The UI will be available at `http://localhost:8080`

### Option 2: Build Images Separately

```bash
# Build frontend
docker build -t dgx-spark-ui .

# Build backend
docker build -t dgx-spark-api ./backend

# Run backend (needs Docker socket access)
docker run -d \
  --name dgx-spark-api \
  -p 8000:8000 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --group-add $(getent group docker | cut -d: -f3) \
  dgx-spark-api

# Run frontend
docker run -d \
  --name dgx-spark-ui \
  -p 8080:80 \
  dgx-spark-ui
```

### Option 3: Development Mode

```bash
# Frontend
npm install
npm run dev

# Backend (in another terminal)
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DGX Spark Host                       │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │  Frontend   │  │   Backend   │  │    Deployed     │ │
│  │   (Nginx)   │──│  (FastAPI)  │──│   Containers    │ │
│  │   :8080     │  │   :8000     │  │  (JupyterLab,   │ │
│  └─────────────┘  └──────┬──────┘  │   Ollama, etc)  │ │
│                          │         └─────────────────┘ │
│                    Docker Socket                        │
│                   /var/run/docker.sock                  │
└─────────────────────────────────────────────────────────┘
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/system/stats` | GET | Real-time system statistics from nvidia-smi |
| `/api/deployments` | GET | List all running deployments |
| `/api/deployments/{id}` | POST | Deploy a launchable |
| `/api/deployments/{id}` | DELETE | Stop a deployment |
| `/api/services/jupyterlab` | POST | Start/stop/restart JupyterLab |
| `/api/health` | GET | Health check |

## Configuration

### API Keys

Configure your API keys in the Settings page (`/settings`):

1. **NGC API Key**: Required for NVIDIA NGC container registry
   - Get one at [ngc.nvidia.com](https://ngc.nvidia.com/setup/api-key)
2. **HuggingFace Token**: Required for gated models
   - Get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

Keys are stored in browser localStorage and sent to the backend when deploying launchables that require authentication.

## Supported Launchables

### Quickstarts
- **VS Code** - Browser-based development environment
- **JupyterLab** - Interactive notebook environment
- **Open WebUI** - Chat interface with Ollama
- **ComfyUI** - Stable Diffusion workflow editor

### AI/ML Workloads
- **Ollama** - Local LLM inference
- **vLLM** - High-throughput LLM serving
- **SGLang** - Fast inference framework
- **TensorRT-LLM** - Optimized inference
- **NeMo** - Model training and fine-tuning
- And 20+ more from [build.nvidia.com/spark](https://build.nvidia.com/spark)

## Troubleshooting

### Docker Socket Permission Denied

```bash
# Add your user to the docker group
sudo usermod -aG docker $USER

# Or run with the docker group
docker run --group-add $(getent group docker | cut -d: -f3) ...
```

### nvidia-smi Not Found

The backend gracefully falls back to mock data when `nvidia-smi` is not available (useful for development).

### GPU Container Runtime

Ensure NVIDIA Container Toolkit is installed:

```bash
nvidia-container-cli info
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

## Tech Stack

- **Frontend**: React 18 + TypeScript + Tailwind CSS + Shadcn/ui
- **Backend**: Python FastAPI + Docker SDK
- **Serving**: Nginx (frontend) + Uvicorn (backend)

## License

MIT License - See LICENSE file for details.

## Support

- [NVIDIA DGX Spark Support](https://www.nvidia.com/en-us/support/dgx-spark/)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
- [DGX Spark Playbooks](https://build.nvidia.com/spark)
