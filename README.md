# DGX Spark WebUI

A web-based dashboard for managing your DGX Spark system. Monitor system resources, deploy AI workloads with one click, and manage your API keys.

![DGX Spark Dashboard](https://build.nvidia.com/_next/image?url=https%3A%2F%2Fassets.ngc.nvidia.com%2Fproducts%2Fapi-catalog%2Fspark%2Fdgx-spark-hero.jpg&w=1920&q=75)

## Features

- **📊 System Dashboard**: Monitor memory usage and GPU utilization in real-time
- **🚀 Launchables**: One-click deployments for 30+ NVIDIA Spark playbooks
- **🔑 API Key Management**: Securely store NGC and HuggingFace tokens
- **🐳 Docker Ready**: Easy deployment with Docker

## Quick Start

### Option 1: Docker Run (Easiest)

```bash
# Build the image
docker build -t dgx-spark-ui .

# Run the container
docker run -d -p 8080:80 --name dgx-spark-ui dgx-spark-ui
```

Access the UI at: **http://localhost:8080**

### Option 2: Docker Compose

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

Access the UI at: **http://localhost:8080**

### Option 3: Development Mode

```bash
# Install dependencies
npm install

# Start dev server
npm run dev
```

Access the UI at: **http://localhost:5173**

## Configuration

### API Keys

The UI stores API keys locally in your browser's localStorage. To configure:

1. Navigate to **Settings** in the UI
2. Enter your **NVIDIA NGC API Key** (get one at [ngc.nvidia.com](https://ngc.nvidia.com/setup/api-key))
3. Enter your **HuggingFace Token** (get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))
4. Click **Save Settings**

### Changing the Port

To run on a different port, modify the docker run command:

```bash
# Run on port 3000 instead
docker run -d -p 3000:80 --name dgx-spark-ui dgx-spark-ui
```

Or update `docker-compose.yml`:

```yaml
ports:
  - "3000:80"
```

## Launchables

The UI includes all official NVIDIA Spark playbooks from [build.nvidia.com/spark](https://build.nvidia.com/spark):

### Quickstarts
- VS Code
- DGX Dashboard
- Open WebUI with Ollama
- Comfy UI

### AI/ML Workloads
- vLLM for Inference
- SGLang for Inference
- TensorRT-LLM
- NeMo Fine-tuning
- LLaMA Factory
- And 20+ more...

## Architecture

```
dgx-spark-ui/
├── src/
│   ├── components/
│   │   ├── dashboard/     # Dashboard components (gauges, stats)
│   │   ├── launchables/   # Launchable cards and grid
│   │   ├── layout/        # Header, Layout components
│   │   └── ui/            # Shadcn UI components
│   ├── data/
│   │   └── launchables.ts # All playbook definitions
│   ├── pages/
│   │   ├── Index.tsx      # Dashboard page
│   │   ├── Launchables.tsx # Launchables browser
│   │   └── Settings.tsx   # API key management
│   └── App.tsx
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
└── README.md
```

## Tech Stack

- **React 18** + **TypeScript**
- **Tailwind CSS** with custom NVIDIA theme
- **Shadcn/ui** components
- **Vite** for development and building
- **Nginx** for production serving

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is for personal use with your DGX Spark system.

## Support

- [NVIDIA DGX Spark Support](https://www.nvidia.com/en-us/support/dgx-spark/)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
- [DGX Spark Playbooks](https://build.nvidia.com/spark)
