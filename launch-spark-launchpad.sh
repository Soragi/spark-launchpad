#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# Spark Launchpad Launcher
# =============================================================================
# A custom script for launching Spark Launchpad from the DGX Spark
# custom scripts launcher. This script handles Docker container management
# and provides real-time feedback.
# =============================================================================

# --- Configuration ---
PORT="${PORT:-8080}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Cleanup handler ---
cleanup() {
  echo ""
  echo "Stopping Spark Launchpad..."
  docker compose down 2>/dev/null || true
  echo "Cleanup complete."
  exit 0
}

trap cleanup INT TERM HUP QUIT

# --- Header ---
echo "============================================================"
echo "  Spark Launchpad"
echo "============================================================"
echo "Port: ${PORT}"
echo "Directory: ${SCRIPT_DIR}"
echo ""

# --- Change to script directory ---
cd "${SCRIPT_DIR}"

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
echo "  Starting Spark Launchpad"
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
