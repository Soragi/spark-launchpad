// Pre-scraped installation instructions for all launchables
// Each step includes instructional text and optional code block

export interface InstructionStep {
  title: string;
  description: string;
  code?: string;
  note?: string;
  warning?: string;
}

export interface LaunchableInstructions {
  launchable_id: string;
  title: string;
  overview: string;
  prerequisites: string[];
  steps: InstructionStep[];
  nextSteps?: string[];
  resources?: { title: string; url: string }[];
}

export const launchableInstructions: Record<string, LaunchableInstructions> = {
  "vllm": {
    launchable_id: "vllm",
    title: "vLLM for Inference",
    overview: "vLLM is an inference engine designed to run large language models efficiently. It uses PagedAttention for memory efficiency and continuous batching for high throughput. It has an OpenAI-compatible API so applications built for the OpenAI API can switch to a vLLM backend with little or no modification.",
    prerequisites: [
      "Docker with GPU support enabled",
      "Experience building and configuring containers with Docker",
      "HuggingFace API key (for gated models)"
    ],
    steps: [
      {
        title: "Configure Docker permissions",
        description: "To easily manage containers without sudo, you must be in the `docker` group. If you choose to skip this step, you will need to run Docker commands with sudo.",
        code: `docker ps

# If you see a permission denied error, add your user to the docker group
sudo usermod -aG docker $USER
newgrp docker`
      },
      {
        title: "Pull vLLM container image",
        description: "Find the latest container build from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm",
        code: `export LATEST_VLLM_VERSION=<latest_container_version>

# example
# export LATEST_VLLM_VERSION=25.11-py3

docker pull nvcr.io/nvidia/vllm:\${LATEST_VLLM_VERSION}`,
        note: "For Nemotron3-Nano model support, please use release version 25.12.post1-py3"
      },
      {
        title: "Pull specific version for Nemotron support",
        description: "If you plan to use Nemotron3-Nano, pull this specific version:",
        code: `docker pull nvcr.io/nvidia/vllm:25.12.post1-py3`
      },
      {
        title: "Test vLLM in container",
        description: "Launch the container and start vLLM server with a test model to verify basic functionality.",
        code: `docker run -it --gpus all -p 8000:8000 \\
  nvcr.io/nvidia/vllm:\${LATEST_VLLM_VERSION} \\
  vllm serve "Qwen/Qwen2.5-Math-1.5B-Instruct"`,
        note: "Expected output should include model loading confirmation, server startup on port 8000, and GPU memory allocation details."
      },
      {
        title: "Test the server",
        description: "In another terminal, test the server with a simple math question:",
        code: `curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "messages": [{"role": "user", "content": "12*17"}],
    "max_tokens": 500
  }'`,
        note: "Expected response should contain \"content\": \"204\" or similar mathematical calculation."
      },
      {
        title: "Cleanup and rollback",
        description: "For container approach (non-destructive):",
        code: `docker rm $(docker ps -aq --filter ancestor=nvcr.io/nvidia/vllm:\${LATEST_VLLM_VERSION})
docker rmi nvcr.io/nvidia/vllm`
      }
    ],
    nextSteps: [
      "Production deployment: Configure vLLM with your specific model requirements",
      "Performance tuning: Adjust batch sizes and memory settings for your workload",
      "Monitoring: Set up logging and metrics collection for production use"
    ],
    resources: [
      { title: "vLLM Documentation", url: "https://docs.vllm.ai/en/latest/" },
      { title: "DGX Spark Documentation", url: "https://docs.nvidia.com/dgx/dgx-spark" }
    ]
  },

  "speculative-decoding": {
    launchable_id: "speculative-decoding",
    title: "Speculative Decoding",
    overview: "Speculative decoding speeds up text generation by using a small, fast model to draft several tokens ahead, then having the larger model quickly verify or adjust them. This way, the big model doesn't need to predict every token step-by-step, reducing latency while keeping output quality.",
    prerequisites: [
      "NVIDIA Spark device with sufficient GPU memory available",
      "Docker with GPU support enabled",
      "HuggingFace Token for model access"
    ],
    steps: [
      {
        title: "Configure Docker permissions",
        description: "To easily manage containers without sudo, you must be in the `docker` group. If you choose to skip this step, you will need to run Docker commands with sudo.",
        code: `docker ps

# If you see a permission denied error, add your user to the docker group
sudo usermod -aG docker $USER
newgrp docker`
      },
      {
        title: "Set Environment Variables",
        description: "Set up the environment variables for downstream services:",
        code: `export HF_TOKEN=<your_huggingface_token>`
      },
      {
        title: "Option 1: EAGLE-3 Speculative Decoding",
        description: "Run EAGLE-3 Speculative Decoding by executing the following command. EAGLE-3 uses a built-in drafting head that generates speculative tokens internally for simpler deployment and better accuracy.",
        code: `docker run \\
  -e HF_TOKEN=$HF_TOKEN \\
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \\
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \\
  --gpus=all --ipc=host --network host \\
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6 \\
  bash -c '
    hf download openai/gpt-oss-120b && \\
    hf download nvidia/gpt-oss-120b-Eagle3-long-context \\
        --local-dir /opt/gpt-oss-120b-Eagle3/ && \\
    cat > /tmp/extra-llm-api-config.yml <<EOF
enable_attention_dp: false
disable_overlap_scheduler: false
enable_autotuner: false
cuda_graph_config:
    max_batch_size: 1
speculative_config:
    decoding_type: Eagle
    max_draft_len: 5
    speculative_model_dir: /opt/gpt-oss-120b-Eagle3/

kv_cache_config:
    free_gpu_memory_fraction: 0.9
    enable_block_reuse: false
EOF
    export TIKTOKEN_ENCODINGS_BASE="/tmp/harmony-reqs" && \\
    mkdir -p $TIKTOKEN_ENCODINGS_BASE && \\
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken && \\
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
    trtllm-serve openai/gpt-oss-120b \\
      --backend pytorch --tp_size 1 \\
      --max_batch_size 1 \\
      --extra_llm_api_options /tmp/extra-llm-api-config.yml'`
      },
      {
        title: "Test EAGLE-3",
        description: "Once the server is running, test it by making an API call from another terminal:",
        code: `curl -X POST http://localhost:8000/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "openai/gpt-oss-120b",
    "prompt": "Solve the following problem step by step. If a train travels 180 km in 3 hours, and then slows down by 20% for the next 2 hours, what is the total distance traveled? Show all intermediate calculations and provide a final numeric answer.",
    "max_tokens": 300,
    "temperature": 0.7
  }'`
      },
      {
        title: "Option 2: Draft Target Speculative Decoding",
        description: "Execute the following command to set up and run draft target speculative decoding. This approach uses an 8B draft model to accelerate a 70B target model with FP4 quantization for reduced memory footprint.",
        code: `docker run \\
  -e HF_TOKEN=$HF_TOKEN \\
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \\
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \\
  --gpus=all --ipc=host --network host nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6 \\
  bash -c "
    # Download models
    hf download nvidia/Llama-3.3-70B-Instruct-FP4 && \\
    hf download nvidia/Llama-3.1-8B-Instruct-FP4 \\
    --local-dir /opt/Llama-3.1-8B-Instruct-FP4/ && \\

    # Create configuration file
    cat <<EOF > extra-llm-api-config.yml
print_iter_log: false
disable_overlap_scheduler: true
speculative_config:
  decoding_type: DraftTarget
  max_draft_len: 4
  speculative_model_dir: /opt/Llama-3.1-8B-Instruct-FP4/
kv_cache_config:
  enable_block_reuse: false
EOF

    # Start TensorRT-LLM server
    trtllm-serve nvidia/Llama-3.3-70B-Instruct-FP4 \\
      --backend pytorch --tp_size 1 \\
      --max_batch_size 1 \\
      --kv_cache_free_gpu_memory_fraction 0.9 \\
      --extra_llm_api_options ./extra-llm-api-config.yml
  "`
      },
      {
        title: "Test Draft Target",
        description: "Once the server is running, test it by making an API call from another terminal:",
        code: `curl -X POST http://localhost:8000/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "nvidia/Llama-3.3-70B-Instruct-FP4",
    "prompt": "Explain the benefits of speculative decoding:",
    "max_tokens": 150,
    "temperature": 0.7
  }'`
      },
      {
        title: "Cleanup",
        description: "Stop the Docker container when finished:",
        code: `# Find and stop the container
docker ps
docker stop <container_id>

# Optional: Clean up downloaded models from cache
# rm -rf $HOME/.cache/huggingface/hub/models--*gpt-oss*`
      }
    ],
    nextSteps: [
      "Experiment with different max_draft_len values (1, 2, 3, 4, 8)",
      "Monitor token acceptance rates and throughput improvements",
      "Test with different prompt lengths and generation parameters"
    ],
    resources: [
      { title: "Speculative Decoding Documentation", url: "https://nvidia.github.io/TensorRT-LLM/1.2.0rc6/features/speculative-decoding.html" },
      { title: "DGX Spark Documentation", url: "https://docs.nvidia.com/dgx/dgx-spark" }
    ]
  },

  "nemotron": {
    launchable_id: "nemotron",
    title: "Nemotron-3-Nano with llama.cpp",
    overview: "Nemotron-3-Nano-30B-A3B is NVIDIA's powerful language model featuring a 30 billion parameter Mixture of Experts (MoE) architecture with only 3 billion active parameters. This efficient design enables high-quality inference with lower computational requirements, making it ideal for DGX Spark's GB10 GPU. The model includes built-in reasoning (thinking mode) and tool calling support via the chat template.",
    prerequisites: [
      "NVIDIA DGX Spark with GB10 GPU",
      "At least 40GB available GPU memory",
      "At least 50GB available storage space",
      "Git, CMake 3.14+, CUDA Toolkit installed"
    ],
    steps: [
      {
        title: "Verify prerequisites",
        description: "Ensure you have the required tools installed on your DGX Spark before proceeding.",
        code: `git --version
cmake --version
nvcc --version`,
        note: "All commands should return version information. If any are missing, install them before continuing."
      },
      {
        title: "Install Hugging Face CLI",
        description: "Install the Hugging Face CLI for downloading models:",
        code: `python3 -m venv nemotron-venv
source nemotron-venv/bin/activate
pip install -U "huggingface_hub[cli]"`,
      },
      {
        title: "Verify HF CLI installation",
        description: "Verify the Hugging Face CLI is installed correctly:",
        code: `hf version`
      },
      {
        title: "Clone llama.cpp repository",
        description: "Clone the llama.cpp repository which provides the inference framework for running Nemotron models.",
        code: `git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp`
      },
      {
        title: "Build llama.cpp with CUDA support",
        description: "Build llama.cpp with CUDA enabled and targeting the GB10's sm_121 compute architecture. This compiles CUDA kernels specifically optimized for your DGX Spark GPU.",
        code: `mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="121" -DLLAMA_CURL=OFF
make -j8`,
        note: "The build process takes approximately 5-10 minutes. You should see compilation progress and eventually a successful build message."
      },
      {
        title: "Download the Nemotron GGUF model",
        description: "Download the Q8 quantized GGUF model from Hugging Face. This model provides excellent quality while fitting within the GB10's memory capacity.",
        code: `hf download unsloth/Nemotron-3-Nano-30B-A3B-GGUF \\
  Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf \\
  --local-dir ~/models/nemotron3-gguf`,
        note: "This downloads approximately 38GB. The download can be resumed if interrupted."
      },
      {
        title: "Start the llama.cpp server",
        description: "Launch the inference server with the Nemotron model. The server provides an OpenAI-compatible API endpoint.",
        code: `./bin/llama-server \\
  --model ~/models/nemotron3-gguf/Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf \\
  --host 0.0.0.0 \\
  --port 30000 \\
  --n-gpu-layers 99 \\
  --ctx-size 8192 \\
  --threads 8`,
        note: "Parameters: --host 0.0.0.0 (listen on all interfaces), --port 30000 (API port), --n-gpu-layers 99 (offload all layers to GPU), --ctx-size 8192 (context window), --threads 8 (CPU threads)"
      },
      {
        title: "Test the API",
        description: "Open a new terminal and test the inference server using the OpenAI-compatible chat completions endpoint.",
        code: `curl http://localhost:30000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "nemotron",
    "messages": [{"role": "user", "content": "New York is a great city because..."}],
    "max_tokens": 100
  }'`
      },
      {
        title: "Test reasoning capabilities",
        description: "Nemotron-3-Nano includes built-in reasoning capabilities. Test with a more complex prompt:",
        code: `curl http://localhost:30000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "nemotron",
    "messages": [{"role": "user", "content": "Solve this step by step: If a train travels 120 miles in 2 hours, what is its average speed?"}],
    "max_tokens": 500
  }'`,
        note: "The model will provide a detailed reasoning chain before giving the final answer."
      },
      {
        title: "Cleanup",
        description: "To stop the server, press Ctrl+C in the terminal where it's running. To completely remove the installation:",
        code: `# Remove llama.cpp build
rm -rf ~/llama.cpp

# Remove downloaded models
rm -rf ~/models/nemotron3-gguf`
      }
    ],
    nextSteps: [
      "Increase context size: For longer conversations, increase --ctx-size up to 1048576 (1M tokens)",
      "Integrate with applications: Use the OpenAI-compatible API with tools like Open WebUI, Continue.dev, or custom applications"
    ],
    resources: [
      { title: "llama.cpp GitHub Repository", url: "https://github.com/ggml-org/llama.cpp" },
      { title: "Nemotron-3-Nano GGUF on Hugging Face", url: "https://huggingface.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF" },
      { title: "DGX Spark Documentation", url: "https://docs.nvidia.com/dgx/dgx-spark" }
    ]
  },

  "open-webui": {
    launchable_id: "open-webui",
    title: "Open WebUI with Ollama",
    overview: "Open WebUI is an extensible, self-hosted AI interface that operates entirely offline. This playbook shows you how to deploy Open WebUI with an integrated Ollama server on your DGX Spark device that lets you access the web interface from your local browser while the models run on Spark's GPU.",
    prerequisites: [
      "DGX Spark device is set up and accessible",
      "Local Network Access to your DGX Spark",
      "Enough disk space for the container image and model downloads"
    ],
    steps: [
      {
        title: "Configure Docker permissions",
        description: "To easily manage containers without sudo, you must be in the `docker` group. If you choose to skip this step, you will need to run Docker commands with sudo.",
        code: `docker ps

# If you see a permission denied error, add your user to the docker group
sudo usermod -aG docker $USER
newgrp docker`
      },
      {
        title: "Verify Docker setup and pull container",
        description: "Pull the Open WebUI container image with integrated Ollama:",
        code: `docker pull ghcr.io/open-webui/open-webui:ollama`
      },
      {
        title: "Start the Open WebUI container",
        description: "Start the Open WebUI container by running:",
        code: `docker run -d -p 8080:8080 --gpus=all \\
  -v open-webui:/app/backend/data \\
  -v open-webui-ollama:/root/.ollama \\
  --name open-webui ghcr.io/open-webui/open-webui:ollama`,
        note: "Application data will be stored in the 'open-webui' volume and model data will be stored in the 'open-webui-ollama' volume."
      },
      {
        title: "Create administrator account",
        description: "Set up the initial administrator account for Open WebUI. This is a local account that you will use to access the Open WebUI interface. Open http://localhost:8080 in your browser, click 'Get Started', fill out the administrator account creation form, and click the registration button."
      },
      {
        title: "Download and configure a model",
        description: "Download a language model through Ollama and configure it for use in Open WebUI. Click on the 'Select a model' dropdown in the top left corner, type 'gpt-oss:20b' in the search field, click 'Pull gpt-oss:20b from Ollama.com', and wait for the download to complete.",
        note: "This download happens on your DGX Spark device and may take several minutes."
      },
      {
        title: "Test the model",
        description: "Verify that the setup is working properly by testing model inference. In the chat text area, enter: 'Write me a haiku about GPUs'. Press Enter and wait for the model's response."
      },
      {
        title: "Cleanup and rollback",
        description: "Steps to completely remove the Open WebUI installation and free up resources.",
        code: `# Stop and remove the Open WebUI container
docker stop open-webui
docker rm open-webui

# Remove the downloaded images
docker rmi ghcr.io/open-webui/open-webui:ollama

# Remove persistent data volumes
docker volume rm open-webui open-webui-ollama`,
        warning: "These commands will permanently delete all Open WebUI data and downloaded models."
      }
    ],
    nextSteps: [
      "Try downloading different models from the Ollama library at https://ollama.com/library",
      "Set up with NVIDIA Sync to monitor GPU and memory usage through the DGX Dashboard"
    ],
    resources: [
      { title: "Open WebUI Documentation", url: "https://docs.openwebui.com/" },
      { title: "DGX Spark Documentation", url: "https://docs.nvidia.com/dgx/dgx-spark" }
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
    steps: [
      {
        title: "Verify system prerequisites",
        description: "Check that your NVIDIA DGX Spark device meets the requirements before proceeding with installation.",
        code: `python3 --version
pip3 --version
nvcc --version
nvidia-smi`,
        note: "Expected output should show Python 3.8+, pip available, CUDA toolkit, and GPU detection."
      },
      {
        title: "Create Python virtual environment",
        description: "You will install ComfyUI on your host system, so you should create an isolated environment to avoid conflicts with system packages.",
        code: `python3 -m venv comfyui-env
source comfyui-env/bin/activate`,
        note: "Verify the virtual environment is active by checking the command prompt shows (comfyui-env)."
      },
      {
        title: "Install PyTorch with CUDA support",
        description: "Install PyTorch with CUDA 13.0 support. This installation targets CUDA 13.0 compatibility with Blackwell architecture GPUs.",
        code: `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130`
      },
      {
        title: "Clone ComfyUI repository",
        description: "Download the ComfyUI source code from the official repository.",
        code: `git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI/`
      },
      {
        title: "Install ComfyUI dependencies",
        description: "Install the required Python packages for ComfyUI operation. This installs all necessary dependencies including web interface components and model handling libraries.",
        code: `pip install -r requirements.txt`
      },
      {
        title: "Download Stable Diffusion checkpoint",
        description: "Navigate to the checkpoints directory and download the Stable Diffusion 1.5 model.",
        code: `cd models/checkpoints/
wget https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors
cd ../../`,
        note: "The download will be approximately 2GB and may take several minutes depending on network speed."
      },
      {
        title: "Launch ComfyUI server",
        description: "Start the ComfyUI web server with network access enabled. The server will bind to all network interfaces on port 8188, making it accessible from other devices.",
        code: `python main.py --listen 0.0.0.0`
      },
      {
        title: "Validate installation",
        description: "Check that ComfyUI is running correctly and accessible. Open a web browser and navigate to http://<SPARK_IP>:8188 where <SPARK_IP> is your device's IP address.",
        code: `curl -I http://localhost:8188`,
        note: "Expected output should show HTTP 200 response indicating the web server is operational."
      },
      {
        title: "Cleanup and rollback",
        description: "If you need to remove the installation completely:",
        code: `deactivate
rm -rf comfyui-env/
rm -rf ComfyUI/`,
        warning: "This will delete all installed packages and downloaded models."
      }
    ],
    nextSteps: [
      "Access the web interface at http://<SPARK_IP>:8188",
      "Load the default workflow and click 'Run' to generate your first image",
      "Monitor GPU usage with nvidia-smi in a separate terminal"
    ],
    resources: [
      { title: "ComfyUI Documentation", url: "https://docs.comfy.org/get_started/first_generation" },
      { title: "DGX Spark Documentation", url: "https://docs.nvidia.com/dgx/dgx-spark" }
    ]
  },

  "sglang": {
    launchable_id: "sglang",
    title: "SGLang for Inference",
    overview: "SGLang is a fast serving framework for LLMs and VLMs with optimized backend runtime and frontend language for faster, more controllable inference. It supports text generation, chat completion, and vision-language tasks.",
    prerequisites: [
      "NVIDIA Spark device with Blackwell architecture",
      "Docker with GPU support",
      "Sufficient disk space for models"
    ],
    steps: [
      {
        title: "Verify system prerequisites",
        description: "Check that your NVIDIA Spark device meets all requirements before proceeding. This step runs on your host system and ensures Docker, GPU drivers, and container toolkit are properly configured.",
        code: `# Verify Docker installation
docker --version

# Check NVIDIA GPU drivers
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all lmsysorg/sglang:spark nvidia-smi

# Check available disk space
df -h /`,
        note: "If you experience timeouts or 'connection refused' errors while pulling the container image, you may need to use a VPN or a proxy."
      },
      {
        title: "Configure Docker permissions",
        description: "If you see a permission denied error, add your user to the docker group:",
        code: `sudo usermod -aG docker $USER
newgrp docker`
      },
      {
        title: "Pull the SGLang Container",
        description: "Download the latest SGLang container. This step runs on the host and may take several minutes depending on your network connection.",
        code: `# Pull the SGLang container
docker pull lmsysorg/sglang:spark

# Verify the image was downloaded
docker images | grep sglang`
      },
      {
        title: "Launch SGLang container for server mode",
        description: "Start the SGLang container in server mode to enable HTTP API access. This runs the inference server inside the container, exposing it on port 30000 for client connections.",
        code: `# Launch container with GPU support and port mapping
docker run --gpus all -it --rm \\
  -p 30000:30000 \\
  -v /tmp:/tmp \\
  lmsysorg/sglang:spark \\
  bash`
      },
      {
        title: "Start the SGLang inference server",
        description: "Inside the container, launch the HTTP inference server with a supported model. This step runs inside the Docker container and starts the SGLang server daemon.",
        code: `# Start the inference server with DeepSeek-V2-Lite model
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
curl http://localhost:30000/health`
      },
      {
        title: "Test client-server inference",
        description: "From a new terminal on your host system, test the SGLang server API to ensure it's working correctly.",
        code: `curl -X POST http://localhost:30000/generate \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "What does NVIDIA love?",
    "sampling_params": {
      "temperature": 0.7,
      "max_new_tokens": 100
    }
  }'`
      },
      {
        title: "Validate installation",
        description: "Confirm that server mode is working correctly:",
        code: `# Check server mode (from host)
curl http://localhost:30000/health
curl -X POST http://localhost:30000/generate -H "Content-Type: application/json" \\
  -d '{"text": "Hello", "sampling_params": {"max_new_tokens": 10}}'

# Check container logs
docker ps
docker logs <CONTAINER_ID>`
      },
      {
        title: "Cleanup and rollback",
        description: "Stop and remove containers to clean up resources.",
        code: `# Stop all SGLang containers
docker ps | grep sglang | awk '{print $1}' | xargs docker stop

# Remove stopped containers
docker container prune -f

# Remove SGLang images (optional)
docker rmi lmsysorg/sglang:spark`,
        warning: "This will stop all SGLang containers and remove temporary data."
      }
    ],
    nextSteps: [
      "Integrate the HTTP API into your applications using the /generate endpoint",
      "Experiment with different models by changing the --model-path parameter",
      "Scale up using multiple GPUs by adjusting the --tp (tensor parallel) setting"
    ],
    resources: [
      { title: "SGLang Documentation", url: "https://docs.sglang.ai/" },
      { title: "DGX Spark Documentation", url: "https://docs.nvidia.com/dgx/dgx-spark" }
    ]
  },

  "vscode": {
    launchable_id: "vscode",
    title: "VS Code",
    overview: "Install and use Visual Studio Code on your DGX Spark device with access to the system's ARM64 architecture and GPU resources.",
    prerequisites: [
      "DGX Spark device is set up",
      "Administrative privileges",
      "Active internet connection"
    ],
    steps: [
      {
        title: "Verify system requirements",
        description: "Verify ARM64 architecture and check available resources:",
        code: `# Verify ARM64 architecture
uname -m
# Expected output: aarch64

# Check available disk space (VS Code requires ~200MB)
df -h /

# Verify desktop environment is running
ps aux | grep -E "(gnome|kde|xfce)"

# Verify GUI desktop environment is available
echo $DISPLAY
# Should return display information like :0 or :10.0`
      },
      {
        title: "Download VS Code ARM64 installer",
        description: "Download the VS Code ARM64 Debian package:",
        code: `wget https://code.visualstudio.com/sha/download?build=stable\\&os=linux-deb-arm64 -O vscode-arm64.deb`
      },
      {
        title: "Install VS Code package",
        description: "Install VS Code and fix any dependency issues:",
        code: `sudo dpkg -i vscode-arm64.deb

# Fix any dependency issues if they occur
sudo apt-get install -f`
      },
      {
        title: "Verify installation",
        description: "Check if VS Code is installed correctly:",
        code: `# Check if VS Code is installed
which code

# Verify version
code --version

# Test launch (will open VS Code GUI)
code &`
      },
      {
        title: "Configure for Spark development",
        description: "Launch VS Code and create a workspace:",
        code: `# Launch VS Code if not already running
code

# Or create a new project directory and open it
mkdir ~/spark-dev-workspace
cd ~/spark-dev-workspace
code .`
      },
      {
        title: "Validate setup",
        description: "Create a test file to validate the setup:",
        code: `# Create test directory and file
mkdir ~/vscode-test
cd ~/vscode-test
echo 'print("Hello from DGX Spark!")' > test.py
code test.py`
      }
    ],
    nextSteps: [
      "Open File > Preferences > Settings to configure your editor",
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
    steps: [
      {
        title: "Access the DGX Dashboard locally",
        description: "Click the DGX Dashboard icon in your application launcher, or open a browser and navigate to http://localhost:8888"
      },
      {
        title: "Remote access via SSH tunneling",
        description: "For remote access, set up an SSH tunnel:",
        code: `ssh -L 8888:localhost:8888 user@<SPARK_IP>

# Then open in your local browser:
# http://localhost:8888`
      },
      {
        title: "Launch JupyterLab from the Dashboard",
        description: "Open the DGX Dashboard in your browser, click 'Launch JupyterLab' button, select your preferred Python environment, and wait for JupyterLab to start."
      },
      {
        title: "Test GPU functionality in JupyterLab",
        description: "Create a new Python notebook and test GPU access:",
        code: `import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")`
      }
    ],
    nextSteps: [
      "Use the dashboard to update system packages and firmware",
      "Monitor GPU performance in real-time from the dashboard",
      "Explore pre-configured Python environments in JupyterLab"
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
    steps: [
      {
        title: "Pull TensorRT-LLM container",
        description: "Download the TensorRT-LLM container from NGC:",
        code: `docker pull nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6`
      },
      {
        title: "Run container with GPU support",
        description: "Launch the container with GPU access and volume mounts:",
        code: `docker run --gpus all -it --rm \\
  -p 8000:8000 \\
  -v ~/.cache/huggingface:/root/.cache/huggingface \\
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6 \\
  bash`
      },
      {
        title: "Build and run TRT-LLM",
        description: "Inside the container, follow the TensorRT-LLM documentation for your specific model. The process typically involves converting your model to TensorRT-LLM format and running the optimized inference."
      }
    ],
    nextSteps: [
      "Explore tensor, pipeline, and sequence parallelism options",
      "Integrate with Hugging Face and PyTorch models",
      "Benchmark throughput improvements over standard PyTorch inference"
    ],
    resources: [
      { title: "TensorRT-LLM Documentation", url: "https://nvidia.github.io/TensorRT-LLM/" },
      { title: "DGX Spark Documentation", url: "https://docs.nvidia.com/dgx/dgx-spark" }
    ]
  },

  "pytorch-fine-tune": {
    launchable_id: "pytorch-fine-tune",
    title: "Fine-tune with PyTorch",
    overview: "Use PyTorch to fine-tune models locally on your DGX Spark device with CUDA acceleration.",
    prerequisites: [
      "DGX Spark with GPU access",
      "Python 3.8+",
      "CUDA toolkit installed"
    ],
    steps: [
      {
        title: "Create virtual environment",
        description: "Create an isolated Python environment for fine-tuning:",
        code: `python3 -m venv pytorch-env
source pytorch-env/bin/activate`
      },
      {
        title: "Install PyTorch with CUDA support",
        description: "Install PyTorch with CUDA 13.0 support for Blackwell GPUs:",
        code: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130`
      },
      {
        title: "Verify GPU access",
        description: "Confirm PyTorch can access the GPU:",
        code: `python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"`,
        note: "You should see CUDA available: True and your GPU name."
      },
      {
        title: "Install training dependencies",
        description: "Install additional libraries for model fine-tuning:",
        code: `pip install transformers datasets accelerate peft`
      },
      {
        title: "Start fine-tuning",
        description: "You can now start fine-tuning models. See the PyTorch and Hugging Face documentation for specific fine-tuning scripts and examples."
      }
    ],
    nextSteps: [
      "Explore LoRA and QLoRA for memory-efficient fine-tuning",
      "Use the Hugging Face Trainer API for simplified training",
      "Monitor training with TensorBoard"
    ]
  },

  "tailscale": {
    launchable_id: "tailscale",
    title: "Set up Tailscale on Your Spark",
    overview: "Use Tailscale to connect to your Spark on your home network no matter where you are. Tailscale creates encrypted peer-to-peer connections that work behind NAT and firewalls.",
    prerequisites: [
      "Tailscale account",
      "DGX Spark with internet access"
    ],
    steps: [
      {
        title: "Install Tailscale",
        description: "Install Tailscale using the official installer:",
        code: `curl -fsSL https://tailscale.com/install.sh | sh`
      },
      {
        title: "Start Tailscale and authenticate",
        description: "Start Tailscale and follow the link to authenticate in your browser:",
        code: `sudo tailscale up`,
        note: "Your Spark will appear in your Tailscale network after authentication."
      },
      {
        title: "Get your Tailscale IP",
        description: "Retrieve the Tailscale IP address assigned to your Spark:",
        code: `tailscale ip -4`
      },
      {
        title: "Connect from anywhere",
        description: "Use your Tailscale IP to connect to your Spark from any device on your Tailscale network:",
        code: `ssh user@<tailscale-ip>`
      }
    ],
    nextSteps: [
      "Install Tailscale on your other devices to access your Spark from anywhere",
      "Set up Tailscale SSH for passwordless authentication",
      "Configure Tailscale exit nodes for secure browsing"
    ]
  },

  "single-cell": {
    launchable_id: "single-cell",
    title: "Single-cell RNA Sequencing",
    overview: "An end-to-end GPU-powered workflow for scRNA-seq using RAPIDS for accelerated data processing and analysis.",
    prerequisites: [
      "DGX Spark with GPU access",
      "Docker with GPU support"
    ],
    steps: [
      {
        title: "Pull RAPIDS container",
        description: "Download the RAPIDS notebook container:",
        code: `docker pull nvcr.io/nvidia/rapidsai/notebooks:24.12-cuda12.5-py3.12`
      },
      {
        title: "Run RAPIDS notebook environment",
        description: "Launch the container with GPU access:",
        code: `docker run --gpus all -it --rm \\
  -p 8888:8888 \\
  nvcr.io/nvidia/rapidsai/notebooks:24.12-cuda12.5-py3.12`
      },
      {
        title: "Access JupyterLab",
        description: "Open http://localhost:8888 in your browser and follow the single-cell RNA-seq workflow notebooks included in the container."
      }
    ]
  },

  "portfolio-optimization": {
    launchable_id: "portfolio-optimization",
    title: "Portfolio Optimization",
    overview: "GPU-Accelerated portfolio optimization using cuOpt and cuML for financial analysis and optimization.",
    prerequisites: [
      "DGX Spark with GPU access",
      "Docker with GPU support"
    ],
    steps: [
      {
        title: "Pull RAPIDS container",
        description: "Download the RAPIDS notebook container:",
        code: `docker pull nvcr.io/nvidia/rapidsai/notebooks:24.12-cuda12.5-py3.12`
      },
      {
        title: "Run container",
        description: "Launch the container with GPU access:",
        code: `docker run --gpus all -it --rm \\
  -p 8888:8888 \\
  nvcr.io/nvidia/rapidsai/notebooks:24.12-cuda12.5-py3.12`
      },
      {
        title: "Install cuOpt",
        description: "Inside the container, install the cuOpt library:",
        code: `pip install cuopt`
      },
      {
        title: "Run portfolio optimization",
        description: "Follow the portfolio optimization examples in the notebooks. Use cuML for GPU-accelerated machine learning and cuOpt for optimization."
      }
    ]
  },

  "cuda-x-data-science": {
    launchable_id: "cuda-x-data-science",
    title: "CUDA-X Data Science",
    overview: "Install and use NVIDIA cuML and cuDF to accelerate UMAP, HDBSCAN, pandas and more with zero code changes.",
    prerequisites: [
      "DGX Spark with GPU access",
      "Docker with GPU support"
    ],
    steps: [
      {
        title: "Pull RAPIDS container",
        description: "Download the RAPIDS notebook container:",
        code: `docker pull nvcr.io/nvidia/rapidsai/notebooks:24.12-cuda12.5-py3.12`
      },
      {
        title: "Run RAPIDS environment",
        description: "Launch the container with GPU access:",
        code: `docker run --gpus all -it --rm \\
  -p 8888:8888 \\
  nvcr.io/nvidia/rapidsai/notebooks:24.12-cuda12.5-py3.12`
      },
      {
        title: "Use cuDF for GPU-accelerated pandas",
        description: "cuDF is a drop-in replacement for pandas:",
        code: `import cudf
df = cudf.read_csv('data.csv')  # GPU-accelerated!`
      },
      {
        title: "Use cuML for GPU-accelerated ML",
        description: "cuML provides GPU-accelerated machine learning algorithms:",
        code: `from cuml.cluster import HDBSCAN
clusterer = HDBSCAN()
labels = clusterer.fit_predict(data)`
      }
    ],
    nextSteps: [
      "UMAP, HDBSCAN, and more accelerated with zero code changes",
      "Drop-in replacement for pandas with cuDF"
    ]
  },

  "vibe-coding": {
    launchable_id: "vibe-coding",
    title: "Vibe Coding in VS Code",
    overview: "Use DGX Spark as a local or remote Vibe Coding assistant with Ollama and Continue extension for AI-powered coding assistance.",
    prerequisites: [
      "VS Code installed",
      "Ollama running on DGX Spark",
      "Continue extension installed"
    ],
    steps: [
      {
        title: "Install Continue extension",
        description: "Open VS Code, go to Extensions, search for 'Continue', and install the extension."
      },
      {
        title: "Start Ollama on DGX Spark",
        description: "If not already running, start the Ollama server:",
        code: `ollama serve`
      },
      {
        title: "Pull a code-focused model",
        description: "Download a model optimized for coding assistance:",
        code: `ollama pull codellama:13b`
      },
      {
        title: "Configure Continue",
        description: "In VS Code, open Continue settings and set the API base URL to point to your Spark's Ollama instance:",
        code: `# Set API base URL to:
# http://<SPARK_IP>:11434`
      }
    ],
    nextSteps: [
      "Use Continue for code completion, refactoring, and explanations",
      "Try different code-focused models like deepseek-coder or starcoder"
    ]
  },

  "flux-finetuning": {
    launchable_id: "flux-finetuning",
    title: "FLUX.1 Dreambooth LoRA Fine-tuning",
    overview: "Fine-tune FLUX.1-dev 12B model using Dreambooth LoRA for custom image generation with your own subjects.",
    prerequisites: [
      "DGX Spark with sufficient GPU memory",
      "HuggingFace API key",
      "Training images for your subject (5-10 images)"
    ],
    steps: [
      {
        title: "Create virtual environment",
        description: "Set up an isolated Python environment:",
        code: `python3 -m venv flux-env
source flux-env/bin/activate`
      },
      {
        title: "Install dependencies",
        description: "Install the required libraries:",
        code: `pip install torch torchvision diffusers transformers accelerate peft`
      },
      {
        title: "Clone the training repository",
        description: "Get the Diffusers examples:",
        code: `git clone https://github.com/huggingface/diffusers
cd diffusers/examples/dreambooth`
      },
      {
        title: "Install example requirements",
        description: "Install the Dreambooth example dependencies:",
        code: `pip install -r requirements.txt`
      },
      {
        title: "Prepare training images",
        description: "Place 5-10 images of your subject in a folder:",
        code: `# Create folder and add your training images
mkdir ./training_images/
# Copy your subject images to this folder`
      },
      {
        title: "Run LoRA fine-tuning",
        description: "Start the fine-tuning process:",
        code: `accelerate launch train_dreambooth_lora_flux.py \\
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \\
  --instance_data_dir="./training_images" \\
  --instance_prompt="a photo of sks person" \\
  --output_dir="./flux-lora-output" \\
  --resolution=512 \\
  --train_batch_size=1 \\
  --gradient_accumulation_steps=4 \\
  --learning_rate=1e-4 \\
  --max_train_steps=500`,
        note: "Training takes approximately 1 hour with 500 steps. Requires HuggingFace API key for gated model access."
      }
    ]
  },

  "llama-factory": {
    launchable_id: "llama-factory",
    title: "LLaMA Factory",
    overview: "Install and fine-tune models with LLaMA Factory, an easy-to-use framework for LLM fine-tuning with a WebUI.",
    prerequisites: [
      "DGX Spark with GPU access",
      "HuggingFace API key",
      "Python 3.8+"
    ],
    steps: [
      {
        title: "Clone LLaMA Factory",
        description: "Download the LLaMA Factory repository:",
        code: `git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory`
      },
      {
        title: "Create virtual environment",
        description: "Set up an isolated Python environment:",
        code: `python3 -m venv llama-factory-env
source llama-factory-env/bin/activate`
      },
      {
        title: "Install dependencies",
        description: "Install LLaMA Factory with PyTorch and metrics support:",
        code: `pip install -e ".[torch,metrics]"`
      },
      {
        title: "Launch the WebUI",
        description: "Start the LLaMA Factory web interface:",
        code: `python -m llama_factory.webui`,
        note: "Access the WebUI at http://localhost:7860"
      },
      {
        title: "Configure training",
        description: "In the WebUI, configure your model, dataset, and training parameters. LLaMA Factory supports LoRA, QLoRA, and full fine-tuning."
      }
    ],
    nextSteps: [
      "Supports LoRA, QLoRA, and full fine-tuning",
      "WebUI for easy configuration",
      "Requires HuggingFace API key for gated models"
    ]
  },

  "unsloth": {
    launchable_id: "unsloth",
    title: "Unsloth on DGX Spark",
    overview: "Optimized fine-tuning with Unsloth for 2x faster training with 50% less memory usage.",
    prerequisites: [
      "DGX Spark with GPU access",
      "HuggingFace API key",
      "Python 3.8+"
    ],
    steps: [
      {
        title: "Create virtual environment",
        description: "Set up an isolated Python environment:",
        code: `python3 -m venv unsloth-env
source unsloth-env/bin/activate`
      },
      {
        title: "Install Unsloth",
        description: "Install Unsloth from the official repository:",
        code: `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`
      },
      {
        title: "Install additional dependencies",
        description: "Install training libraries:",
        code: `pip install --no-deps trl peft accelerate bitsandbytes`
      },
      {
        title: "Fine-tune with Unsloth",
        description: "Example code to fine-tune Llama with Unsloth:",
        code: `from unsloth import FastLanguageModel

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
      }
    ],
    nextSteps: [
      "2x faster training than standard methods",
      "50% less memory usage",
      "Requires HuggingFace API key for gated models"
    ]
  },

  "nim-llm": {
    launchable_id: "nim-llm",
    title: "NIM on Spark",
    overview: "Deploy NVIDIA Inference Microservices (NIM) on your DGX Spark for optimized model serving with an OpenAI-compatible API.",
    prerequisites: [
      "DGX Spark with GPU access",
      "NGC API key",
      "Docker with GPU support"
    ],
    steps: [
      {
        title: "Login to NGC",
        description: "Authenticate with the NGC container registry:",
        code: `docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key>`
      },
      {
        title: "Pull NIM container",
        description: "Download the NIM container (example with Llama):",
        code: `docker pull nvcr.io/nim/meta/llama-3.1-8b-instruct:latest`
      },
      {
        title: "Run NIM container",
        description: "Start the NIM container with your API key:",
        code: `docker run --gpus all -it --rm \\
  -p 8000:8000 \\
  -e NGC_API_KEY=$NGC_API_KEY \\
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest`
      },
      {
        title: "Test the API",
        description: "Verify the NIM is working with a test request:",
        code: `curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'`
      }
    ],
    nextSteps: [
      "Optimized for NVIDIA GPUs",
      "OpenAI-compatible API",
      "Explore other NIM containers on NGC"
    ]
  },

  "nccl": {
    launchable_id: "nccl",
    title: "NCCL for Two Sparks",
    overview: "Install and test NCCL on two Sparks for multi-GPU distributed training across devices.",
    prerequisites: [
      "Two DGX Spark devices",
      "Network connectivity between Sparks",
      "NCCL installed on both devices"
    ],
    steps: [
      {
        title: "Verify NCCL installation",
        description: "Check that NCCL is available through PyTorch:",
        code: `python3 -c "import torch.distributed.nccl as nccl; print('NCCL available')"`
      },
      {
        title: "Check network connectivity",
        description: "Verify the two Sparks can communicate:",
        code: `ping <other_spark_ip>`
      },
      {
        title: "Run NCCL test",
        description: "On the primary Spark, run NCCL all-reduce tests. Follow the NCCL documentation for specific test configurations and benchmarks."
      }
    ]
  },

  "live-vlm-webui": {
    launchable_id: "live-vlm-webui",
    title: "Live VLM WebUI",
    overview: "Real-time Vision Language Model interaction with webcam streaming. This playbook sets up a web interface that captures video from your webcam and processes it through a VLM for real-time visual understanding and conversation.",
    prerequisites: [
      "DGX Spark with GPU access",
      "Docker with GPU support",
      "Webcam connected to your system",
      "HuggingFace API key (for some models)"
    ],
    steps: [
      {
        title: "Configure Docker permissions",
        description: "Ensure your user is in the docker group for easy container management:",
        code: `docker ps

# If you see a permission denied error, add your user to the docker group
sudo usermod -aG docker $USER
newgrp docker`
      },
      {
        title: "Clone the Live VLM repository",
        description: "Download the Live VLM WebUI project:",
        code: `git clone https://github.com/NVIDIA/GenerativeAIExamples.git
cd GenerativeAIExamples/community/live-vlm-webui`
      },
      {
        title: "Set up environment variables",
        description: "Configure the necessary environment variables:",
        code: `export HF_TOKEN=<your_huggingface_token>
export CUDA_VISIBLE_DEVICES=0`
      },
      {
        title: "Build the Docker container",
        description: "Build the Live VLM container image:",
        code: `docker build -t live-vlm-webui .`
      },
      {
        title: "Run the container with webcam access",
        description: "Start the container with GPU and webcam device access:",
        code: `docker run --gpus all -it --rm \\
  -p 7860:7860 \\
  --device=/dev/video0:/dev/video0 \\
  -e HF_TOKEN=$HF_TOKEN \\
  live-vlm-webui`,
        note: "Replace /dev/video0 with your webcam device path if different. Check available devices with 'ls /dev/video*'."
      },
      {
        title: "Access the WebUI",
        description: "Open your browser and navigate to the web interface:",
        code: `# Open in browser:
# http://localhost:7860`,
        note: "Allow webcam access when prompted by the browser."
      },
      {
        title: "Test the VLM",
        description: "In the WebUI, enable your webcam stream and ask questions about what the model sees. Try prompts like 'What do you see?' or 'Describe the scene'."
      },
      {
        title: "Cleanup",
        description: "Stop and remove the container when finished:",
        code: `# Stop the container with Ctrl+C
# Remove the image if no longer needed
docker rmi live-vlm-webui`
      }
    ],
    nextSteps: [
      "Experiment with different VLM models for varying capabilities",
      "Adjust streaming resolution for performance optimization",
      "Integrate with other applications using the API endpoints"
    ],
    resources: [
      { title: "NVIDIA Generative AI Examples", url: "https://github.com/NVIDIA/GenerativeAIExamples" },
      { title: "DGX Spark Documentation", url: "https://docs.nvidia.com/dgx/dgx-spark" }
    ]
  },

  "isaac": {
    launchable_id: "isaac",
    title: "Install and Use Isaac Sim and Isaac Lab",
    overview: "Build Isaac Sim and Isaac Lab from source for DGX Spark. Isaac Sim is NVIDIA's robotics simulation platform, and Isaac Lab provides tools for robot learning and manipulation.",
    prerequisites: [
      "DGX Spark with GPU access",
      "At least 100GB available storage",
      "Ubuntu 22.04 or later",
      "NVIDIA Container Toolkit installed"
    ],
    steps: [
      {
        title: "Verify system requirements",
        description: "Check GPU and driver compatibility:",
        code: `nvidia-smi
# Verify driver version 535 or higher

# Check available storage
df -h /`
      },
      {
        title: "Pull Isaac Sim container",
        description: "Download the Isaac Sim container from NGC:",
        code: `docker pull nvcr.io/nvidia/isaac-sim:4.2.0`,
        note: "This download is approximately 20GB and may take some time."
      },
      {
        title: "Run Isaac Sim container",
        description: "Launch Isaac Sim with GPU access and display forwarding:",
        code: `xhost +local:docker

docker run --gpus all -it --rm \\
  -e DISPLAY=$DISPLAY \\
  -v /tmp/.X11-unix:/tmp/.X11-unix \\
  -v ~/isaac-sim-data:/root/.local/share/ov/data \\
  --network host \\
  nvcr.io/nvidia/isaac-sim:4.2.0`,
        note: "Requires X11 display. For headless mode, use the --headless flag."
      },
      {
        title: "Launch Isaac Sim",
        description: "Inside the container, start Isaac Sim:",
        code: `./isaac-sim.sh`,
        note: "First launch will take several minutes to compile shaders and cache assets."
      },
      {
        title: "Install Isaac Lab",
        description: "Clone and install Isaac Lab for robot learning:",
        code: `git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Create conda environment
conda create -n isaaclab python=3.10 -y
conda activate isaaclab

# Install Isaac Lab
./isaaclab.sh --install`,
        note: "Isaac Lab requires Isaac Sim to be installed first."
      },
      {
        title: "Run a sample environment",
        description: "Test Isaac Lab with a sample robot environment:",
        code: `python source/standalone/demos/quadrupeds.py`
      },
      {
        title: "Cleanup",
        description: "Remove containers and images when finished:",
        code: `docker rmi nvcr.io/nvidia/isaac-sim:4.2.0

# Remove cached data (optional)
rm -rf ~/isaac-sim-data`
      }
    ],
    nextSteps: [
      "Explore the Isaac Sim asset library for pre-built robots and environments",
      "Train reinforcement learning policies with Isaac Lab",
      "Connect to real robots using ROS 2 integration"
    ],
    resources: [
      { title: "Isaac Sim Documentation", url: "https://docs.omniverse.nvidia.com/isaacsim/latest" },
      { title: "Isaac Lab Documentation", url: "https://isaac-sim.github.io/IsaacLab/" },
      { title: "DGX Spark Documentation", url: "https://docs.nvidia.com/dgx/dgx-spark" }
    ]
  },

  "multi-agent-chatbot": {
    launchable_id: "multi-agent-chatbot",
    title: "Build and Deploy a Multi-Agent Chatbot",
    overview: "Deploy a multi-agent chatbot system that coordinates multiple AI agents to handle complex tasks. This setup enables specialized agents to collaborate on user queries.",
    prerequisites: [
      "DGX Spark with GPU access",
      "Docker with GPU support",
      "Python 3.10+"
    ],
    steps: [
      {
        title: "Clone the multi-agent example",
        description: "Download the NVIDIA multi-agent chatbot example:",
        code: `git clone https://github.com/NVIDIA/GenerativeAIExamples.git
cd GenerativeAIExamples/community/multi-agent-chatbot`
      },
      {
        title: "Create virtual environment",
        description: "Set up an isolated Python environment:",
        code: `python3 -m venv multi-agent-env
source multi-agent-env/bin/activate`
      },
      {
        title: "Install dependencies",
        description: "Install the required packages:",
        code: `pip install -r requirements.txt`
      },
      {
        title: "Start the Ollama backend",
        description: "Run Ollama to serve the language models:",
        code: `# Pull a capable model
ollama pull llama3.1:8b

# Start Ollama server
ollama serve`
      },
      {
        title: "Configure agent settings",
        description: "Edit the configuration file to define your agents:",
        code: `# config.yaml defines agent roles and capabilities
# Example agents: researcher, coder, reviewer
cat config.yaml`
      },
      {
        title: "Launch the chatbot",
        description: "Start the multi-agent chatbot interface:",
        code: `python app.py`,
        note: "Access the chatbot at http://localhost:7860"
      },
      {
        title: "Test the agents",
        description: "In the chat interface, try queries that require multiple agents to collaborate, such as 'Research the latest GPU architectures and write a summary'."
      },
      {
        title: "Cleanup",
        description: "Stop all services when finished:",
        code: `# Stop the chatbot with Ctrl+C
# Stop Ollama
pkill ollama

# Deactivate virtual environment
deactivate`
      }
    ],
    nextSteps: [
      "Customize agent personas and capabilities in config.yaml",
      "Add specialized agents for domain-specific tasks",
      "Integrate with external tools and APIs"
    ],
    resources: [
      { title: "NVIDIA Generative AI Examples", url: "https://github.com/NVIDIA/GenerativeAIExamples" },
      { title: "DGX Spark Documentation", url: "https://docs.nvidia.com/dgx/dgx-spark" }
    ]
  },

  "multi-modal-inference": {
    launchable_id: "multi-modal-inference",
    title: "Multi-modal Inference",
    overview: "Set up multi-modal inference with TensorRT for processing both text and images. This enables vision-language models that can understand and generate content across modalities.",
    prerequisites: [
      "DGX Spark with GPU access",
      "Docker with GPU support",
      "At least 32GB GPU memory"
    ],
    steps: [
      {
        title: "Pull TensorRT-LLM container",
        description: "Download the TensorRT-LLM container with multi-modal support:",
        code: `docker pull nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6`
      },
      {
        title: "Run the container",
        description: "Launch the container with GPU access:",
        code: `docker run --gpus all -it --rm \\
  -p 8000:8000 \\
  -v ~/.cache/huggingface:/root/.cache/huggingface \\
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6 \\
  bash`
      },
      {
        title: "Download a multi-modal model",
        description: "Inside the container, download a vision-language model:",
        code: `pip install huggingface_hub

huggingface-cli download llava-hf/llava-v1.6-mistral-7b-hf \\
  --local-dir /models/llava-v1.6`
      },
      {
        title: "Build the TensorRT engine",
        description: "Convert the model to TensorRT format for optimized inference:",
        code: `python examples/llava/build.py \\
  --model_path /models/llava-v1.6 \\
  --output_dir /engines/llava`,
        note: "Engine building may take 15-30 minutes depending on model size."
      },
      {
        title: "Start the inference server",
        description: "Launch the multi-modal inference server:",
        code: `python examples/llava/run.py \\
  --engine_dir /engines/llava \\
  --hf_model_dir /models/llava-v1.6 \\
  --host 0.0.0.0 \\
  --port 8000`
      },
      {
        title: "Test with an image",
        description: "Send a test request with an image and text prompt:",
        code: `curl -X POST http://localhost:8000/generate \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "Describe this image in detail",
    "image_url": "https://example.com/image.jpg"
  }'`
      },
      {
        title: "Cleanup",
        description: "Exit the container and remove cached data:",
        code: `# Exit container
exit

# Remove downloaded models (optional)
rm -rf ~/.cache/huggingface/hub/models--llava*`
      }
    ],
    nextSteps: [
      "Explore different vision-language models like LLaVA, Phi-3-Vision, or InternVL",
      "Integrate with image processing pipelines",
      "Benchmark throughput for production deployments"
    ],
    resources: [
      { title: "TensorRT-LLM Multi-modal Examples", url: "https://nvidia.github.io/TensorRT-LLM/" },
      { title: "DGX Spark Documentation", url: "https://docs.nvidia.com/dgx/dgx-spark" }
    ]
  },

  "nemo-fine-tune": {
    launchable_id: "nemo-fine-tune",
    title: "Fine-tune with NeMo",
    overview: "Use NVIDIA NeMo to fine-tune large language models on your DGX Spark. NeMo provides optimized training recipes and supports LoRA, P-tuning, and full fine-tuning.",
    prerequisites: [
      "DGX Spark with GPU access",
      "Docker with GPU support",
      "HuggingFace API key for gated models"
    ],
    steps: [
      {
        title: "Pull NeMo Framework container",
        description: "Download the NeMo Framework container from NGC:",
        code: `docker pull nvcr.io/nvidia/nemo:24.07`
      },
      {
        title: "Run the NeMo container",
        description: "Launch the container with GPU access and volume mounts:",
        code: `docker run --gpus all -it --rm \\
  -v ~/nemo-data:/data \\
  -v ~/.cache/huggingface:/root/.cache/huggingface \\
  -e HF_TOKEN=$HF_TOKEN \\
  nvcr.io/nvidia/nemo:24.07`
      },
      {
        title: "Prepare training data",
        description: "Format your data in JSONL format for fine-tuning:",
        code: `# Example format for instruction tuning:
# {"input": "What is AI?", "output": "AI is..."}
# {"input": "Explain GPUs", "output": "GPUs are..."}

mkdir -p /data/training
# Copy your training data to /data/training/train.jsonl`
      },
      {
        title: "Download base model",
        description: "Download a base model to fine-tune:",
        code: `python -c "
from nemo.collections.llm import Llama3Config8B
from huggingface_hub import snapshot_download

snapshot_download('meta-llama/Llama-3.1-8B',
                  local_dir='/data/models/llama-3.1-8b')
"`
      },
      {
        title: "Configure LoRA fine-tuning",
        description: "Create a training configuration file:",
        code: `cat > /data/lora_config.yaml << 'EOF'
trainer:
  devices: 1
  max_steps: 1000
  val_check_interval: 100

model:
  peft:
    peft_scheme: "lora"
    lora_rank: 16
    lora_alpha: 32

data:
  train_ds:
    file_path: /data/training/train.jsonl
    batch_size: 4
EOF`
      },
      {
        title: "Run LoRA fine-tuning",
        description: "Start the fine-tuning process:",
        code: `python -m nemo.collections.llm.recipes.llama3_8b_lora \\
  --config-path=/data \\
  --config-name=lora_config`,
        note: "Training time depends on dataset size. Monitor GPU usage with nvidia-smi."
      },
      {
        title: "Export the fine-tuned model",
        description: "Convert the NeMo checkpoint to HuggingFace format:",
        code: `python -m nemo.collections.llm.export \\
  --model-path /data/checkpoints/latest \\
  --output-path /data/models/finetuned-llama`
      },
      {
        title: "Cleanup",
        description: "Exit container and manage data:",
        code: `exit

# Remove large model files if needed
rm -rf ~/nemo-data/models`
      }
    ],
    nextSteps: [
      "Experiment with different PEFT methods: LoRA, P-tuning, Adapter tuning",
      "Scale training with tensor and pipeline parallelism",
      "Deploy fine-tuned models with NIM or vLLM"
    ],
    resources: [
      { title: "NeMo Framework Documentation", url: "https://docs.nvidia.com/nemo-framework/user-guide/latest/" },
      { title: "DGX Spark Documentation", url: "https://docs.nvidia.com/dgx/dgx-spark" }
    ]
  },

  "nvfp4-quantization": {
    launchable_id: "nvfp4-quantization",
    title: "NVFP4 Quantization",
    overview: "Quantize a model to NVFP4 (4-bit floating point) using TensorRT Model Optimizer. This enables running larger models on DGX Spark by reducing memory requirements while maintaining accuracy.",
    prerequisites: [
      "DGX Spark with GPU access",
      "Docker with GPU support",
      "Python 3.10+"
    ],
    steps: [
      {
        title: "Pull TensorRT-LLM container",
        description: "Download the container with Model Optimizer:",
        code: `docker pull nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6`
      },
      {
        title: "Run the container",
        description: "Launch with GPU access:",
        code: `docker run --gpus all -it --rm \\
  -v ~/.cache/huggingface:/root/.cache/huggingface \\
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6 \\
  bash`
      },
      {
        title: "Install Model Optimizer",
        description: "Install the TensorRT Model Optimizer package:",
        code: `pip install nvidia-modelopt`
      },
      {
        title: "Download a model to quantize",
        description: "Get a model from HuggingFace:",
        code: `pip install huggingface_hub

huggingface-cli download meta-llama/Llama-3.1-8B \\
  --local-dir /models/llama-3.1-8b`,
        note: "Requires HuggingFace token for gated models."
      },
      {
        title: "Run NVFP4 quantization",
        description: "Quantize the model to NVFP4 format:",
        code: `python -m modelopt.llm.quantization.quantize_model \\
  --model_dir /models/llama-3.1-8b \\
  --output_dir /models/llama-3.1-8b-nvfp4 \\
  --qformat nvfp4 \\
  --calib_size 512`,
        note: "Calibration uses a subset of data to determine optimal quantization parameters."
      },
      {
        title: "Build TensorRT engine",
        description: "Convert the quantized model to a TensorRT engine:",
        code: `trtllm-build \\
  --checkpoint_dir /models/llama-3.1-8b-nvfp4 \\
  --output_dir /engines/llama-3.1-8b-nvfp4 \\
  --gemm_plugin auto \\
  --max_batch_size 4 \\
  --max_input_len 2048 \\
  --max_seq_len 4096`
      },
      {
        title: "Test the quantized model",
        description: "Run inference with the NVFP4 quantized engine:",
        code: `python examples/run.py \\
  --engine_dir /engines/llama-3.1-8b-nvfp4 \\
  --tokenizer_dir /models/llama-3.1-8b \\
  --input_text "What is the capital of France?"`
      },
      {
        title: "Compare memory usage",
        description: "Check GPU memory reduction:",
        code: `nvidia-smi

# NVFP4 typically uses ~75% less memory than FP16`
      },
      {
        title: "Cleanup",
        description: "Remove models and engines:",
        code: `exit

# Remove large files
rm -rf ~/.cache/huggingface/hub/models--meta-llama*`
      }
    ],
    nextSteps: [
      "Benchmark accuracy degradation vs. FP16 baseline",
      "Try different quantization formats: INT8, INT4, FP8",
      "Deploy quantized models with TensorRT-LLM server"
    ],
    resources: [
      { title: "TensorRT Model Optimizer Documentation", url: "https://nvidia.github.io/TensorRT-Model-Optimizer/" },
      { title: "DGX Spark Documentation", url: "https://docs.nvidia.com/dgx/dgx-spark" }
    ]
  },

  "rag-ai-workbench": {
    launchable_id: "rag-ai-workbench",
    title: "RAG Application in AI Workbench",
    overview: "Install and use NVIDIA AI Workbench to clone and run a reproducible RAG (Retrieval-Augmented Generation) application. AI Workbench provides a desktop app for managing AI projects with built-in GPU support.",
    prerequisites: [
      "DGX Spark with GPU access",
      "Docker installed",
      "Desktop environment with browser"
    ],
    steps: [
      {
        title: "Download AI Workbench",
        description: "Download the AI Workbench installer:",
        code: `wget https://workbench.download.nvidia.com/stable/workbench-linux-arm64.run -O workbench-installer.run
chmod +x workbench-installer.run`
      },
      {
        title: "Install AI Workbench",
        description: "Run the installer:",
        code: `./workbench-installer.run`,
        note: "Follow the installation prompts. This installs the AI Workbench desktop application."
      },
      {
        title: "Launch AI Workbench",
        description: "Start the AI Workbench application:",
        code: `# Launch from applications menu or run:
nvwb`
      },
      {
        title: "Clone the RAG example project",
        description: "In AI Workbench, click 'Clone Project' and enter the RAG example URL:",
        code: `# Project URL:
https://github.com/NVIDIA/workbench-example-hybrid-rag`
      },
      {
        title: "Start the project environment",
        description: "In the project view, click 'Start Environment'. AI Workbench will build the container and set up dependencies automatically.",
        note: "First build may take 10-15 minutes."
      },
      {
        title: "Launch JupyterLab",
        description: "Click 'Open JupyterLab' to access the RAG notebooks and follow the included tutorials."
      },
      {
        title: "Run the RAG application",
        description: "In JupyterLab, open the main notebook and follow the steps to ingest documents and query the RAG system."
      },
      {
        title: "Cleanup",
        description: "Stop the environment when finished:",
        code: `# In AI Workbench, click 'Stop Environment'
# Or uninstall AI Workbench:
nvwb uninstall`
      }
    ],
    nextSteps: [
      "Customize the RAG application with your own documents",
      "Explore other AI Workbench example projects",
      "Create your own reproducible AI projects"
    ],
    resources: [
      { title: "AI Workbench Documentation", url: "https://docs.nvidia.com/ai-workbench/" },
      { title: "DGX Spark Documentation", url: "https://docs.nvidia.com/dgx/dgx-spark" }
    ]
  },

  "connect-two-sparks": {
    launchable_id: "connect-two-sparks",
    title: "Connect Two Sparks",
    overview: "Connect two DGX Spark devices and set them up for distributed inference and fine-tuning. This enables running larger models across multiple GPUs.",
    prerequisites: [
      "Two DGX Spark devices",
      "Network connectivity (Ethernet recommended)",
      "Same software versions on both devices"
    ],
    steps: [
      {
        title: "Verify network connectivity",
        description: "Ensure both Sparks can communicate over the network:",
        code: `# On Spark 1 - get IP address
hostname -I

# On Spark 2 - ping Spark 1
ping <spark1_ip>`,
        note: "Use Ethernet for best performance. WiFi may have higher latency."
      },
      {
        title: "Configure SSH key authentication",
        description: "Set up passwordless SSH between the two Sparks:",
        code: `# On Spark 1 - generate SSH key
ssh-keygen -t ed25519 -f ~/.ssh/id_spark

# Copy to Spark 2
ssh-copy-id -i ~/.ssh/id_spark.pub user@<spark2_ip>

# Test connection
ssh user@<spark2_ip> hostname`
      },
      {
        title: "Verify GPU visibility on both devices",
        description: "Confirm GPUs are accessible on both Sparks:",
        code: `# On Spark 1
nvidia-smi

# On Spark 2 (via SSH)
ssh user@<spark2_ip> nvidia-smi`
      },
      {
        title: "Install distributed training framework",
        description: "Ensure PyTorch with distributed support is available:",
        code: `# On both Sparks
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Verify NCCL support
python -c "import torch.distributed as dist; print('Distributed available:', dist.is_available())"`
      },
      {
        title: "Set up hostfile for distributed training",
        description: "Create a hostfile listing both machines:",
        code: `cat > hostfile << EOF
<spark1_ip> slots=1
<spark2_ip> slots=1
EOF`
      },
      {
        title: "Test distributed communication",
        description: "Run a simple NCCL test across both Sparks:",
        code: `# Create test script
cat > test_dist.py << 'EOF'
import torch
import torch.distributed as dist
import os

dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()
print(f"Rank {rank}/{world_size} initialized on {torch.cuda.get_device_name(0)}")

# Test all-reduce
tensor = torch.ones(1).cuda() * rank
dist.all_reduce(tensor)
print(f"Rank {rank}: all-reduce result = {tensor.item()}")
EOF

# Run across both Sparks
torchrun --nnodes=2 --nproc_per_node=1 \\
  --rdzv_backend=c10d --rdzv_endpoint=<spark1_ip>:29500 \\
  test_dist.py`
      },
      {
        title: "Run distributed inference",
        description: "Use tensor parallelism to run a large model across both GPUs. Example with vLLM:",
        code: `# On Spark 1 (coordinator)
docker run --gpus all -it --rm --network host \\
  nvcr.io/nvidia/vllm:25.12.post1-py3 \\
  vllm serve "meta-llama/Llama-3.1-70B" --tensor-parallel-size 2`,
        note: "Tensor parallelism with tp=2 splits the model across both Spark GPUs."
      }
    ],
    nextSteps: [
      "Run larger models that don't fit on a single Spark",
      "Set up distributed fine-tuning with DeepSpeed or FSDP",
      "Configure monitoring across both devices"
    ],
    resources: [
      { title: "PyTorch Distributed Training", url: "https://pytorch.org/tutorials/intermediate/ddp_tutorial.html" },
      { title: "DGX Spark Documentation", url: "https://docs.nvidia.com/dgx/dgx-spark" }
    ]
  },

  "txt2kg": {
    launchable_id: "txt2kg",
    title: "Text to Knowledge Graph",
    overview: "Transform unstructured text into interactive knowledge graphs with LLM inference and graph visualization. This playbook sets up a pipeline that extracts entities and relationships from text using an LLM.",
    prerequisites: [
      "DGX Spark with GPU access",
      "Docker with GPU support",
      "Python 3.10+"
    ],
    steps: [
      {
        title: "Clone the txt2kg repository",
        description: "Download the text-to-knowledge-graph project:",
        code: `git clone https://github.com/NVIDIA/GenerativeAIExamples.git
cd GenerativeAIExamples/community/txt2kg`
      },
      {
        title: "Create virtual environment",
        description: "Set up an isolated Python environment:",
        code: `python3 -m venv txt2kg-env
source txt2kg-env/bin/activate`
      },
      {
        title: "Install dependencies",
        description: "Install the required packages:",
        code: `pip install -r requirements.txt`
      },
      {
        title: "Start the LLM backend",
        description: "Run Ollama with a capable model:",
        code: `# Pull a model good at extraction
ollama pull llama3.1:8b

# Start Ollama
ollama serve`
      },
      {
        title: "Configure the application",
        description: "Set up the connection to your LLM backend:",
        code: `cat > config.yaml << 'EOF'
llm:
  base_url: http://localhost:11434
  model: llama3.1:8b
graph:
  output_format: html
EOF`
      },
      {
        title: "Run text to knowledge graph extraction",
        description: "Process a sample text file:",
        code: `python extract.py --input sample_text.txt --output knowledge_graph.html`,
        note: "The LLM extracts entities and relationships, then generates an interactive graph."
      },
      {
        title: "Launch the visualization server",
        description: "Start the web interface for interactive exploration:",
        code: `python app.py`,
        note: "Access the visualization at http://localhost:7860"
      },
      {
        title: "Explore the knowledge graph",
        description: "In the web interface, upload text documents and explore the generated knowledge graphs. You can click on nodes to see relationships and filter by entity types."
      },
      {
        title: "Cleanup",
        description: "Stop all services when finished:",
        code: `# Stop the app with Ctrl+C
pkill ollama
deactivate`
      }
    ],
    nextSteps: [
      "Process larger documents and combine multiple knowledge graphs",
      "Export graphs to Neo4j or other graph databases",
      "Fine-tune the extraction prompts for domain-specific entities"
    ],
    resources: [
      { title: "NVIDIA Generative AI Examples", url: "https://github.com/NVIDIA/GenerativeAIExamples" },
      { title: "DGX Spark Documentation", url: "https://docs.nvidia.com/dgx/dgx-spark" }
    ]
  },

  "vss": {
    launchable_id: "vss",
    title: "Build a Video Search and Summarization Agent",
    overview: "Run the Video Search and Summarization (VSS) Blueprint on your DGX Spark. This enables intelligent video content analysis with natural language queries.",
    prerequisites: [
      "DGX Spark with GPU access",
      "Docker with GPU support",
      "At least 32GB GPU memory"
    ],
    steps: [
      {
        title: "Clone the VSS Blueprint",
        description: "Download the Video Search and Summarization project:",
        code: `git clone https://github.com/NVIDIA/GenerativeAIExamples.git
cd GenerativeAIExamples/community/video-search-and-summarization`
      },
      {
        title: "Set up environment variables",
        description: "Configure required environment variables:",
        code: `export NGC_API_KEY=<your_ngc_api_key>
export HF_TOKEN=<your_huggingface_token>`
      },
      {
        title: "Pull required containers",
        description: "Download the necessary container images:",
        code: `docker compose pull`,
        note: "This downloads several containers including LLM, embedding, and video processing services."
      },
      {
        title: "Start the VSS services",
        description: "Launch all services with Docker Compose:",
        code: `docker compose up -d`,
        note: "Services include: LLM inference, embedding model, vector database, and web interface."
      },
      {
        title: "Access the web interface",
        description: "Open the VSS web interface in your browser:",
        code: `# Open in browser:
# http://localhost:8080`
      },
      {
        title: "Upload and process videos",
        description: "In the web interface, upload video files. The system will automatically extract audio, transcribe speech, and index visual content."
      },
      {
        title: "Search and summarize",
        description: "Use natural language queries to search video content. Ask questions like 'Find all mentions of product launch' or 'Summarize the main topics discussed'."
      },
      {
        title: "Cleanup",
        description: "Stop all services when finished:",
        code: `docker compose down

# Remove volumes with video data
docker compose down -v`
      }
    ],
    nextSteps: [
      "Process longer videos and video collections",
      "Customize the summarization prompts for your use case",
      "Integrate with your existing video management system"
    ],
    resources: [
      { title: "NVIDIA Generative AI Examples", url: "https://github.com/NVIDIA/GenerativeAIExamples" },
      { title: "DGX Spark Documentation", url: "https://docs.nvidia.com/dgx/dgx-spark" }
    ]
  },

  "jax": {
    launchable_id: "jax",
    title: "Optimized JAX",
    overview: "Optimize JAX to run on DGX Spark with CUDA acceleration. JAX is a high-performance numerical computing library that combines NumPy-like syntax with automatic differentiation and GPU/TPU support.",
    prerequisites: [
      "DGX Spark with GPU access",
      "Python 3.10+",
      "CUDA toolkit installed"
    ],
    steps: [
      {
        title: "Create virtual environment",
        description: "Set up an isolated Python environment for JAX:",
        code: `python3 -m venv jax-env
source jax-env/bin/activate`
      },
      {
        title: "Install JAX with CUDA support",
        description: "Install JAX with GPU support for CUDA 12:",
        code: `pip install --upgrade pip
pip install --upgrade "jax[cuda12]"`,
        note: "This installs JAX with NVIDIA GPU support via CUDA 12."
      },
      {
        title: "Verify GPU detection",
        description: "Confirm JAX can see and use the GPU:",
        code: `python -c "
import jax
print('Devices:', jax.devices())
print('Default backend:', jax.default_backend())
"`,
        note: "You should see 'gpu' as the default backend and your GPU listed."
      },
      {
        title: "Test GPU computation",
        description: "Run a simple GPU-accelerated computation:",
        code: `python << 'EOF'
import jax
import jax.numpy as jnp
from jax import random

# Create a random matrix on GPU
key = random.PRNGKey(0)
x = random.normal(key, (5000, 5000))

# Matrix multiplication on GPU
result = jnp.dot(x, x.T)
print(f"Result shape: {result.shape}")
print(f"Computation device: {result.devices()}")
EOF`
      },
      {
        title: "Install additional JAX libraries",
        description: "Install common JAX ecosystem packages:",
        code: `pip install flax optax orbax-checkpoint`
      },
      {
        title: "Optimize memory allocation",
        description: "Configure JAX memory settings for large models:",
        code: `# Add to your script or .bashrc
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"`,
        note: "These settings optimize GPU memory usage for DGX Spark."
      },
      {
        title: "Run a JAX neural network example",
        description: "Test a simple Flax neural network:",
        code: `python << 'EOF'
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

# Initialize model
model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
print(f"Model parameters: {jax.tree_util.tree_map(lambda x: x.shape, params)}")
EOF`
      },
      {
        title: "Cleanup",
        description: "Deactivate the environment when finished:",
        code: `deactivate

# Remove environment if no longer needed
rm -rf jax-env`
      }
    ],
    nextSteps: [
      "Explore JAX's JIT compilation for faster execution",
      "Use vmap for automatic vectorization",
      "Train large models with pjit for model parallelism"
    ],
    resources: [
      { title: "JAX Documentation", url: "https://jax.readthedocs.io/" },
      { title: "Flax Documentation", url: "https://flax.readthedocs.io/" },
      { title: "DGX Spark Documentation", url: "https://docs.nvidia.com/dgx/dgx-spark" }
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
