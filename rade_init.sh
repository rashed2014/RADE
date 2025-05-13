#!/bin/bash
set -e

echo "🔧 Starting FAISS & RAGatouille environment setup..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="$SCRIPT_DIR/requirements.txt"

if [ -f "$REQ_FILE" ]; then
    echo "📦 Installing from $REQ_FILE..."
    pip install -r "$REQ_FILE"
else
    echo "❌ requirements.txt not found in $SCRIPT_DIR"
fi

# Install torch separately using the custom CUDA index
echo "⚙️ Installing torch with CUDA 12.6 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Clean up conflicting FAISS versions
echo "🔄 Uninstalling existing FAISS versions..."
/databricks/python/bin/pip uninstall -y faiss-cpu faiss-gpu faiss-gpu-cu12 || true

# Reinstall correct FAISS version
echo "🔁 Installing faiss-gpu-cu12..."
/databricks/python/bin/pip install --no-cache-dir faiss-gpu-cu12

echo "✅ Init script complete."
