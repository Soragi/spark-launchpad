#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# Sparky - DGX Spark WebUI Launcher
# =============================================================================
# A custom script for launching Sparky from the DGX Spark custom scripts 
# launcher. This script handles Docker container management, repository 
# updates, and provides real-time feedback.
# =============================================================================

# --- Configuration ---
PORT="${PORT:-8080}"
REPO_URL="${REPO_URL:-https://github.com/YOUR_USERNAME/sparky.git}"
INSTALL_DIR="${INSTALL_DIR:-${HOME}/.sparky}"

# --- Cleanup handler ---
cleanup() {
  echo ""
  echo "Stopping Sparky..."
  cd "${INSTALL_DIR}" 2>/dev/null && docker compose down 2>/dev/null || true
  echo "Cleanup complete."
  exit 0
}

trap cleanup INT TERM HUP QUIT

# --- Header ---
echo "============================================================"
echo "  Sparky - DGX Spark WebUI Launcher"
echo "============================================================"
echo "Port: ${PORT}"
echo "Install directory: ${INSTALL_DIR}"
echo ""

# --- Ensure Docker is installed ---
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: Docker is not installed."
  echo "Please install Docker first: https://docs.docker.com/get-docker/"
  exit 1
fi

# --- Ensure Docker daemon is running ---
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker daemon is not running."
  echo "Please start Docker and try again."
  exit 1
fi

# --- Ensure Docker Compose is available ---
if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: Docker Compose (v2) is not available."
  echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
  exit 1
fi

echo "✓ Docker and Docker Compose are available"

# --- Ensure Git is installed ---
if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: Git is not installed."
  echo "Please install Git: sudo apt install git"
  exit 1
fi

echo "✓ Git is available"

# --- Clone or update repository ---
if [ -d "${INSTALL_DIR}" ]; then
  echo ""
  echo "Updating existing installation..."
  cd "${INSTALL_DIR}"
  git fetch origin
  git reset --hard origin/main 2>/dev/null || git reset --hard origin/master
  echo "✓ Repository updated"
else
  echo ""
  echo "Cloning repository..."
  git clone "${REPO_URL}" "${INSTALL_DIR}"
  cd "${INSTALL_DIR}"
  echo "✓ Repository cloned"
fi

# --- Configure custom port ---
if [ "${PORT}" != "8080" ]; then
  echo ""
  echo "Configuring custom port: ${PORT}"
  sed -i "s/\"8080:80\"/\"${PORT}:80\"/" docker-compose.yml
  echo "✓ Port configured"
fi

# --- Build containers ---
echo ""
echo "Building Docker containers..."
echo "------------------------------------------------------------"
docker compose build

echo ""
echo "✓ Build complete"

# --- Start containers ---
echo ""
echo "============================================================"
echo "  Starting Sparky"
echo "============================================================"
echo ""
echo "  Frontend URL: http://localhost:${PORT}"
echo "  Backend API:  http://localhost:8000"
echo ""
echo "  Press Ctrl+C to stop"
echo ""
echo "------------------------------------------------------------"

# 🔑 IMPORTANT: run in foreground with exec for proper signal handling
exec docker compose up
