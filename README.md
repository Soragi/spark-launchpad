# Spark Launchpad

A dashboard for managing your NVIDIA DGX Spark system. Monitor GPU resources, deploy AI workloads, and access 30+ NVIDIA playbooks.

## Quick Start

```bash
git clone <repository-url>
cd spark-launchpad
docker compose up -d
```

Open `http://localhost:8080` in your browser.

## Requirements

- Docker and Docker Compose
- NVIDIA Container Toolkit (for GPU features)

## Commands

```bash
# Start
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

## Development

```bash
# Frontend
npm install && npm run dev

# Backend
cd backend && pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API Keys

Configure in Settings (`/settings`):
- **NGC API Key**: [ngc.nvidia.com](https://ngc.nvidia.com/setup/api-key)
- **HuggingFace Token**: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

## Links

- [DGX Spark Playbooks](https://build.nvidia.com/spark)
- [NVIDIA Support](https://www.nvidia.com/en-us/support/dgx-spark/)
