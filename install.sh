#!/usr/bin/env bash
set -euo pipefail

echo "==> GPUFlow installer"

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker is required"; exit 1; }
command -v docker-compose >/dev/null 2>&1 || command -v "docker compose" >/dev/null 2>&1 || {
  echo "ERROR: docker-compose is required"; exit 1;
}

# Check for NVIDIA runtime
if ! docker info 2>/dev/null | grep -q "nvidia"; then
  echo "WARNING: NVIDIA container toolkit not detected. GPU passthrough may not work."
  echo "  Install guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# Setup .env
if [ ! -f .env ]; then
  cp .env.example .env
  echo ""
  read -rp "Enter your desired API key (leave blank for a random one): " user_key
  if [ -z "$user_key" ]; then
    user_key=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    echo "Generated API key: $user_key"
  fi
  sed -i "s/change-me-to-a-secure-key/$user_key/" .env
  echo "==> .env created"
else
  echo "==> .env already exists, skipping"
fi

mkdir -p logs

# Build and start
echo ""
echo "==> Building and starting GPUFlow..."
docker-compose up -d --build

echo ""
echo "==> GPUFlow is running!"
echo ""
echo "  Dashboard:  http://localhost:8000/dashboard"
echo "  API docs:   http://localhost:8000/docs"
echo ""
echo "  Install CLI:  pip install -e ."
echo "  Example:      gpuflow run train.py --gpus 2 --name my-experiment"
echo ""
echo "  Your API key is in .env — set it in the dashboard and CLI."
