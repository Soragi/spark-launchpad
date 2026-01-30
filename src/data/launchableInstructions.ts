// Pre-scraped installation instructions for all launchables
// This eliminates the need for live API calls to fetch instructions

export interface LaunchableInstructions {
  launchable_id: string;
  title: string;
  commands: string[];
  overview: string;
  prerequisites: string[];
  notes?: string[];
}

export const launchableInstructions: Record<string, LaunchableInstructions> = {
  "vscode": {
    launchable_id: "vscode",
    title: "VS Code",
    overview: "Install and use Visual Studio Code on your DGX Spark device with access to the system's ARM64 architecture and GPU resources.",
    prerequisites: [
      "DGX Spark device is set up",
      "Administrative privileges",
      "Active internet connection"
    ],
    commands: [
      `# Verify system requirements
# Verify ARM64 architecture
uname -m
# Expected output: aarch64

# Check available disk space (VS Code requires ~200MB)
df -h /

# Verify desktop environment is running
ps aux | grep -E "(gnome|kde|xfce)"

# Verify GUI desktop environment is available
echo $DISPLAY
# Should return display information like :0 or :10.0`,

      `# Download VS Code ARM64 installer
wget https://code.visualstudio.com/sha/download?build=stable\\&os=linux-deb-arm64 -O vscode-arm64.deb`,

      `# Install VS Code package
sudo dpkg -i vscode-arm64.deb

# Fix any dependency issues if they occur
sudo apt-get install -f`,

      `# Verify installation
# Check if VS Code is installed
which code

# Verify version
code --version

# Test launch (will open VS Code GUI)
code &`,

      `# Configure for Spark development
# Launch VS Code if not already running
code

# Or create a new project directory and open it
mkdir ~/spark-dev-workspace
cd ~/spark-dev-workspace
code .`,

      `# Validate setup and test functionality
# Create test directory and file
mkdir ~/vscode-test
cd ~/vscode-test
echo 'print("Hello from DGX Spark!")' > test.py
code test.py`
    ],
    notes: [
      "From within VS Code: Open File > Preferences > Settings",
      "Search for 'terminal integrated shell' to configure default terminal",
      "Install recommended extensions via Extensions tab (left sidebar)"
    ]
  },

  "dgx-dashboard": {
    launchable_id: "dgx-dashboard",
    title: "DGX Dashboard",
    overview: "The DGX Dashboard is a web application that runs locally on DGX Spark devices, providing a graphical interface for system updates, resource monitoring, and an integrated JupyterLab environment.",
    prerequisites: [
      "NVIDIA Grace Blackwell GB10 Superchip System",
      "NVIDIA DGX OS",
      "NVIDIA Sync installed (for remote access) or SSH client configured"
    ],
    commands: [
      `# Access the DGX Dashboard locally
# Click the DGX Dashboard icon in your application launcher
# Or open a browser and navigate to:
# http://localhost:8888`,

      `# For remote access via SSH tunneling
ssh -L 8888:localhost:8888 user@<SPARK_IP>

# Then open in your local browser:
# http://localhost:8888`,

      `# Launch JupyterLab from the Dashboard
# 1. Open the DGX Dashboard in your browser
# 2. Click "Launch JupyterLab" button
# 3. Select your preferred Python environment
# 4. Wait for JupyterLab to start`,

      `# Test GPU functionality in JupyterLab
# Create a new Python notebook and run:
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")`
    ],
    notes: [
      "The dashboard is the easiest way to update system packages and firmware when working remotely",
      "You can monitor GPU performance in real-time from the dashboard",
      "JupyterLab instances come with pre-configured Python environments"
    ]
  },

  "open-webui": {
    launchable_id: "open-webui",
    title: "Open WebUI with Ollama",
    overview: "Deploy Open WebUI with an integrated Ollama server on your DGX Spark device to access the web interface from your local browser while the models run on Spark's GPU.",
    prerequisites: [
      "DGX Spark device is set up and accessible",
      "Local Network Access to your DGX Spark",
      "Enough disk space for the container image and model downloads"
    ],
    commands: [
      `# Configure Docker permissions
docker ps

# If you see a permission denied error, add your user to the docker group
sudo usermod -aG docker $USER
newgrp docker`,

      `# Pull the Open WebUI container image with integrated Ollama
docker pull ghcr.io/open-webui/open-webui:ollama`,

      `# Start the Open WebUI container
docker run -d -p 8080:8080 --gpus=all \\
  -v open-webui:/app/backend/data \\
  -v open-webui-ollama:/root/.ollama \\
  --name open-webui ghcr.io/open-webui/open-webui:ollama`,

      `# Access the web interface
# Open browser and navigate to: http://localhost:8080
# Click "Get Started" and create an administrator account`,

      `# Download a model through the interface
# 1. Click on "Select a model" dropdown
# 2. Type "gpt-oss:20b" in the search field
# 3. Click "Pull 'gpt-oss:20b' from Ollama.com"
# 4. Wait for download to complete`,

      `# Test the model
# In the chat area, type: "Write me a haiku about GPUs"
# Press Enter and wait for the response`
    ],
    notes: [
      "Application data stored in 'open-webui' volume",
      "Model data stored in 'open-webui-ollama' volume",
      "Try different models from https://ollama.com/library"
    ]
  },

  "comfy-ui": {
    launchable_id: "comfy-ui",
    title: "Comfy UI",
    overview: "Install and configure ComfyUI on your NVIDIA DGX Spark device for AI image generation using diffusion-based models like SDXL, Flux, and others.",
    prerequisites: [
      "NVIDIA Grace Blackwell GB10 Superchip System",
      "Minimum 8GB GPU memory for Stable Diffusion models",
      "At least 20GB available storage space",
      "Python 3.8+, pip, CUDA toolkit, Git installed"
    ],
    commands: [
      `# Verify system prerequisites
python3 --version
pip3 --version
nvcc --version
nvidia-smi`,

      `# Create Python virtual environment
python3 -m venv comfyui-env
source comfyui-env/bin/activate`,

      `# Install PyTorch with CUDA 13.0 support
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130`,

      `# Clone ComfyUI repository
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI/`,

      `# Install ComfyUI dependencies
pip install -r requirements.txt`,

      `# Download Stable Diffusion checkpoint (~2GB)
cd models/checkpoints/
wget https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors
cd ../../`,

      `# Launch ComfyUI server
python main.py --listen 0.0.0.0`,

      `# Validate installation
# Open browser: http://<SPARK_IP>:8188
curl -I http://localhost:8188`
    ],
    notes: [
      "The server will bind to all network interfaces on port 8188",
      "Image generation should complete within 30-60 seconds",
      "Workflows are saved as JSON files for versioning and reproducibility"
    ]
  },

  "nemotron": {
    launchable_id: "nemotron",
    title: "Nemotron-3-Nano with llama.cpp",
    overview: "Run Nemotron-3-Nano-30B model using llama.cpp on DGX Spark. Features 30B parameter MoE architecture with only 3B active parameters, ideal for GB10 GPU.",
    prerequisites: [
      "NVIDIA DGX Spark with GB10 GPU",
      "At least 40GB available GPU memory",
      "At least 50GB available storage space",
      "Git, CMake 3.14+, CUDA Toolkit installed"
    ],
    commands: [
      `# Verify prerequisites
git --version
cmake --version
nvcc --version`,

      `# Install Hugging Face CLI
python3 -m venv nemotron-venv
source nemotron-venv/bin/activate
pip install -U "huggingface_hub[cli]"

# Verify installation
hf version`,

      `# Clone llama.cpp repository
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp`,

      `# Build llama.cpp with CUDA support for GB10
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="121" -DLLAMA_CURL=OFF
make -j8`,

      `# Download the Nemotron GGUF model (~38GB)
hf download unsloth/Nemotron-3-Nano-30B-A3B-GGUF \\
  Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf \\
  --local-dir ~/models/nemotron3-gguf`,

      `# Start the llama.cpp server
./bin/llama-server \\
  --model ~/models/nemotron3-gguf/Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf \\
  --host 0.0.0.0 \\
  --port 30000 \\
  --n-gpu-layers 99 \\
  --ctx-size 8192 \\
  --threads 8`,

      `# Test the API (in a new terminal)
curl http://localhost:30000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "nemotron",
    "messages": [{"role": "user", "content": "New York is a great city because..."}],
    "max_tokens": 100
  }'`,

      `# Test reasoning capabilities
curl http://localhost:30000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "nemotron",
    "messages": [{"role": "user", "content": "Solve this step by step: If a train travels 120 miles in 2 hours, what is its average speed?"}],
    "max_tokens": 500
  }'`
    ],
    notes: [
      "Server provides OpenAI-compatible API endpoint",
      "Supports context window up to 1M tokens (increase --ctx-size)",
      "Includes built-in reasoning and tool calling capabilities"
    ]
  },

  "speculative-decoding": {
    launchable_id: "speculative-decoding",
    title: "Speculative Decoding",
    overview: "Speculative decoding speeds up text generation by using a small, fast model to draft several tokens ahead, then having the larger model quickly verify or adjust them.",
    prerequisites: [
      "NVIDIA Spark device with sufficient GPU memory",
      "Docker with GPU support enabled",
      "Active HuggingFace Token for model access"
    ],
    commands: [
      `# Verify Docker GPU support
docker run --gpus all nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6 nvidia-smi`,

      `# Pull TensorRT-LLM container
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6`,

      `# Start container for EAGLE-3 Speculative Decoding
docker run --gpus all -it --rm \\
  -p 8000:8000 \\
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6 \\
  bash`,

      `# Inside the container, run speculative decoding example
# Follow the TensorRT-LLM documentation for EAGLE-3 or Draft-Target setup`
    ],
    notes: [
      "Uses EAGLE-3 and Draft-Target approaches",
      "Reduces latency while maintaining output quality",
      "GPU memory exhaustion possible with large models"
    ]
  },

  "pytorch-fine-tune": {
    launchable_id: "pytorch-fine-tune",
    title: "Fine-tune with PyTorch",
    overview: "Use PyTorch to fine-tune models locally on your DGX Spark device.",
    prerequisites: [
      "DGX Spark with GPU access",
      "Python 3.8+",
      "CUDA toolkit installed"
    ],
    commands: [
      `# Create virtual environment
python3 -m venv pytorch-env
source pytorch-env/bin/activate`,

      `# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130`,

      `# Verify GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"`,

      `# Install additional training dependencies
pip install transformers datasets accelerate peft`,

      `# Start fine-tuning (example with a small model)
# See PyTorch documentation for specific fine-tuning scripts`
    ]
  },

  "vllm": {
    launchable_id: "vllm",
    title: "vLLM for Inference",
    overview: "vLLM is an inference engine using PagedAttention for memory efficiency and continuous batching for high throughput. Provides OpenAI-compatible API.",
    prerequisites: [
      "Docker with GPU support",
      "HuggingFace API key (for gated models)"
    ],
    commands: [
      `# Configure Docker permissions
docker ps

# If permission denied, add user to docker group
sudo usermod -aG docker $USER
newgrp docker`,

      `# Pull vLLM container image
export LATEST_VLLM_VERSION=25.12.post1-py3
docker pull nvcr.io/nvidia/vllm:\${LATEST_VLLM_VERSION}`,

      `# Test vLLM with a small model
docker run -it --gpus all -p 8000:8000 \\
  nvcr.io/nvidia/vllm:\${LATEST_VLLM_VERSION} \\
  vllm serve "Qwen/Qwen2.5-Math-1.5B-Instruct"`,

      `# Test the server (in another terminal)
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "messages": [{"role": "user", "content": "12*17"}],
    "max_tokens": 500
  }'`
    ],
    notes: [
      "For Nemotron3-Nano support, use version 25.12.post1-py3",
      "OpenAI-compatible API for easy integration",
      "Uses PagedAttention for efficient memory management"
    ]
  },

  "sglang": {
    launchable_id: "sglang",
    title: "SGLang for Inference",
    overview: "SGLang is a fast serving framework for LLMs and VLMs with optimized backend runtime and frontend language for faster, more controllable inference.",
    prerequisites: [
      "NVIDIA Spark device with Blackwell architecture",
      "Docker with GPU support",
      "Sufficient disk space for models"
    ],
    commands: [
      `# Verify system prerequisites
docker --version
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all lmsysorg/sglang:spark nvidia-smi

# Check available disk space
df -h /`,

      `# Configure Docker permissions if needed
sudo usermod -aG docker $USER
newgrp docker`,

      `# Pull the SGLang container
docker pull lmsysorg/sglang:spark

# Verify the image was downloaded
docker images | grep sglang`,

      `# Launch SGLang container for server mode
docker run --gpus all -it --rm \\
  -p 30000:30000 \\
  -v /tmp:/tmp \\
  lmsysorg/sglang:spark \\
  bash`,

      `# Inside container: Start the inference server
python3 -m sglang.launch_server \\
  --model-path deepseek-ai/DeepSeek-V2-Lite \\
  --host 0.0.0.0 \\
  --port 30000 \\
  --trust-remote-code \\
  --tp 1 \\
  --attention-backend flashinfer \\
  --mem-fraction-static 0.75 &

# Wait for server to initialize
sleep 30

# Check server status
curl http://localhost:30000/health`,

      `# Test from host (new terminal)
curl -X POST http://localhost:30000/generate \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "What does NVIDIA love?",
    "sampling_params": {
      "temperature": 0.7,
      "max_new_tokens": 100
    }
  }'`
    ],
    notes: [
      "Supports text generation, chat completion, and vision-language tasks",
      "Use --tp for tensor parallelism with multiple GPUs",
      "Optimized for NVIDIA Spark's Blackwell architecture"
    ]
  },

  "trt-llm": {
    launchable_id: "trt-llm",
    title: "TRT LLM for Inference",
    overview: "NVIDIA TensorRT-LLM is an open-source library for optimizing and accelerating LLM inference with efficient kernels, memory management, and parallelism strategies.",
    prerequisites: [
      "DGX Spark with GPU access",
      "Docker with GPU support",
      "Python proficiency with PyTorch"
    ],
    commands: [
      `# Pull TensorRT-LLM container
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6`,

      `# Run container with GPU support
docker run --gpus all -it --rm \\
  -p 8000:8000 \\
  -v ~/.cache/huggingface:/root/.cache/huggingface \\
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6 \\
  bash`,

      `# Inside container: Build and run TRT-LLM
# Follow the TensorRT-LLM documentation for your specific model`
    ],
    notes: [
      "Significantly higher throughput than standard PyTorch inference",
      "Supports tensor, pipeline, and sequence parallelism",
      "Integrates with Hugging Face and PyTorch"
    ]
  },

  "single-cell": {
    launchable_id: "single-cell",
    title: "Single-cell RNA Sequencing",
    overview: "An end-to-end GPU-powered workflow for scRNA-seq using RAPIDS for accelerated data processing and analysis.",
    prerequisites: [
      "DGX Spark with GPU access",
      "Python with RAPIDS libraries"
    ],
    commands: [
      `# Pull RAPIDS container
docker pull nvcr.io/nvidia/rapidsai/notebooks:24.12-cuda12.5-py3.12`,

      `# Run RAPIDS notebook environment
docker run --gpus all -it --rm \\
  -p 8888:8888 \\
  nvcr.io/nvidia/rapidsai/notebooks:24.12-cuda12.5-py3.12`,

      `# Access JupyterLab at http://localhost:8888
# Follow the single-cell RNA-seq workflow notebooks`
    ]
  },

  "portfolio-optimization": {
    launchable_id: "portfolio-optimization",
    title: "Portfolio Optimization",
    overview: "GPU-Accelerated portfolio optimization using cuOpt and cuML for financial analysis and optimization.",
    prerequisites: [
      "DGX Spark with GPU access",
      "RAPIDS and cuOpt libraries"
    ],
    commands: [
      `# Pull RAPIDS container with cuOpt
docker pull nvcr.io/nvidia/rapidsai/notebooks:24.12-cuda12.5-py3.12`,

      `# Run container
docker run --gpus all -it --rm \\
  -p 8888:8888 \\
  nvcr.io/nvidia/rapidsai/notebooks:24.12-cuda12.5-py3.12`,

      `# Install cuOpt (inside container)
pip install cuopt`,

      `# Follow portfolio optimization examples in the notebooks`
    ]
  },

  "cuda-x-data-science": {
    launchable_id: "cuda-x-data-science",
    title: "CUDA-X Data Science",
    overview: "Install and use NVIDIA cuML and cuDF to accelerate UMAP, HDBSCAN, pandas and more with zero code changes.",
    prerequisites: [
      "DGX Spark with GPU access",
      "Python 3.8+"
    ],
    commands: [
      `# Pull RAPIDS container
docker pull nvcr.io/nvidia/rapidsai/notebooks:24.12-cuda12.5-py3.12`,

      `# Run RAPIDS environment
docker run --gpus all -it --rm \\
  -p 8888:8888 \\
  nvcr.io/nvidia/rapidsai/notebooks:24.12-cuda12.5-py3.12`,

      `# Use cuDF for GPU-accelerated pandas
import cudf
df = cudf.read_csv('data.csv')  # GPU-accelerated!`,

      `# Use cuML for GPU-accelerated ML
from cuml.cluster import HDBSCAN
clusterer = HDBSCAN()
labels = clusterer.fit_predict(data)`
    ],
    notes: [
      "Drop-in replacement for pandas with cuDF",
      "UMAP, HDBSCAN, and more accelerated with zero code changes"
    ]
  },

  "vibe-coding": {
    launchable_id: "vibe-coding",
    title: "Vibe Coding in VS Code",
    overview: "Use DGX Spark as a local or remote Vibe Coding assistant with Ollama and Continue extension.",
    prerequisites: [
      "VS Code installed",
      "Ollama running on DGX Spark",
      "Continue extension installed"
    ],
    commands: [
      `# Install Continue extension in VS Code
# Open VS Code > Extensions > Search "Continue" > Install`,

      `# Start Ollama on DGX Spark (if not already running)
ollama serve`,

      `# Pull a code-focused model
ollama pull codellama:13b`,

      `# Configure Continue to use your Spark's Ollama
# In VS Code, open Continue settings
# Set API base URL to: http://<SPARK_IP>:11434`
    ]
  },

  "flux-finetuning": {
    launchable_id: "flux-finetuning",
    title: "FLUX.1 Dreambooth LoRA Fine-tuning",
    overview: "Fine-tune FLUX.1-dev 12B model using Dreambooth LoRA for custom image generation.",
    prerequisites: [
      "DGX Spark with sufficient GPU memory",
      "HuggingFace API key",
      "Training images for your subject"
    ],
    commands: [
      `# Create virtual environment
python3 -m venv flux-env
source flux-env/bin/activate`,

      `# Install dependencies
pip install torch torchvision diffusers transformers accelerate peft`,

      `# Clone the training repository
git clone https://github.com/huggingface/diffusers
cd diffusers/examples/dreambooth`,

      `# Install example requirements
pip install -r requirements.txt`,

      `# Prepare your training images in a folder
# Place 5-10 images of your subject in: ./training_images/`,

      `# Run LoRA fine-tuning
accelerate launch train_dreambooth_lora_flux.py \\
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \\
  --instance_data_dir="./training_images" \\
  --instance_prompt="a photo of sks person" \\
  --output_dir="./flux-lora-output" \\
  --resolution=512 \\
  --train_batch_size=1 \\
  --gradient_accumulation_steps=4 \\
  --learning_rate=1e-4 \\
  --max_train_steps=500`
    ],
    notes: [
      "Requires HuggingFace API key for gated model access",
      "Training takes approximately 1 hour with 500 steps"
    ]
  },

  "llama-factory": {
    launchable_id: "llama-factory",
    title: "LLaMA Factory",
    overview: "Install and fine-tune models with LLaMA Factory, an easy-to-use framework for LLM fine-tuning.",
    prerequisites: [
      "DGX Spark with GPU access",
      "HuggingFace API key",
      "Python 3.8+"
    ],
    commands: [
      `# Clone LLaMA Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory`,

      `# Create virtual environment
python3 -m venv llama-factory-env
source llama-factory-env/bin/activate`,

      `# Install dependencies
pip install -e ".[torch,metrics]"`,

      `# Launch the WebUI
python -m llama_factory.webui`,

      `# Access the WebUI at http://localhost:7860
# Configure your model, dataset, and training parameters`
    ],
    notes: [
      "Supports LoRA, QLoRA, and full fine-tuning",
      "WebUI for easy configuration",
      "Requires HuggingFace API key for gated models"
    ]
  },

  "unsloth": {
    launchable_id: "unsloth",
    title: "Unsloth on DGX Spark",
    overview: "Optimized fine-tuning with Unsloth for 2x faster training with 50% less memory.",
    prerequisites: [
      "DGX Spark with GPU access",
      "HuggingFace API key",
      "Python 3.8+"
    ],
    commands: [
      `# Create virtual environment
python3 -m venv unsloth-env
source unsloth-env/bin/activate`,

      `# Install Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`,

      `# Install additional dependencies
pip install --no-deps trl peft accelerate bitsandbytes`,

      `# Example: Fine-tune Llama with Unsloth
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
)`
    ],
    notes: [
      "2x faster training than standard methods",
      "50% less memory usage",
      "Requires HuggingFace API key for gated models"
    ]
  },

  "nim-llm": {
    launchable_id: "nim-llm",
    title: "NIM on Spark",
    overview: "Deploy NVIDIA Inference Microservices (NIM) on your DGX Spark for optimized model serving.",
    prerequisites: [
      "DGX Spark with GPU access",
      "NGC API key",
      "Docker with GPU support"
    ],
    commands: [
      `# Login to NGC
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key>`,

      `# Pull NIM container (example with Llama)
docker pull nvcr.io/nim/meta/llama-3.1-8b-instruct:latest`,

      `# Run NIM container
docker run --gpus all -it --rm \\
  -p 8000:8000 \\
  -e NGC_API_KEY=<your_key> \\
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest`,

      `# Test the API
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'`
    ],
    notes: [
      "Optimized for NVIDIA GPUs",
      "OpenAI-compatible API",
      "Requires NGC API key"
    ]
  },

  "tailscale": {
    launchable_id: "tailscale",
    title: "Set up Tailscale on Your Spark",
    overview: "Use Tailscale to connect to your Spark on your home network no matter where you are.",
    prerequisites: [
      "Tailscale account",
      "DGX Spark with internet access"
    ],
    commands: [
      `# Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh`,

      `# Start Tailscale and authenticate
sudo tailscale up`,

      `# Follow the link to authenticate in your browser
# Your Spark will appear in your Tailscale network`,

      `# Get your Tailscale IP
tailscale ip -4`,

      `# Connect from anywhere using your Tailscale IP
ssh user@<tailscale-ip>`
    ],
    notes: [
      "Works behind NAT and firewalls",
      "Encrypted peer-to-peer connections",
      "Free for personal use"
    ]
  },

  "nccl": {
    launchable_id: "nccl",
    title: "NCCL for Two Sparks",
    overview: "Install and test NCCL on two Sparks for multi-GPU distributed training.",
    prerequisites: [
      "Two DGX Spark devices",
      "Network connectivity between Sparks",
      "NCCL installed on both devices"
    ],
    commands: [
      `# Verify NCCL installation
python3 -c "import torch.distributed.nccl as nccl; print('NCCL available')"`,

      `# Check network connectivity between Sparks
ping <other_spark_ip>`,

      `# Run NCCL test (on primary Spark)
# Follow NCCL documentation for all-reduce tests`
    ]
  }
};

// Get instructions for a specific launchable
export function getInstructions(launchableId: string): LaunchableInstructions | null {
  return launchableInstructions[launchableId] || null;
}

// Get all launchable IDs that have instructions
export function getAvailableLaunchableIds(): string[] {
  return Object.keys(launchableInstructions);
}
