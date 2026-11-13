#!/usr/bin/env bash
# ==============================================================================
# Google Colab GPU Environment Setup for LexsiLabs (AKCR)
# Compatible with Colab T4 / L4 / A100 GPUs and Remote IDE / Kernel connections
# ==============================================================================
set -e

echo "=================================================================="
echo "🚀 Setting up LexsiLabs on Google Colab GPU Runtime..."
echo "=================================================================="

# Ensure we are at the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 1. Clone cage-challenge-4 if missing (e.g., if repo was cloned without submodules)
if [ ! -f "cage-challenge-4/setup.py" ] && [ ! -f "cage-challenge-4/pyproject.toml" ]; then
    echo "📥 Fetching CybORG CAGE-4 simulator..."
    rm -rf cage-challenge-4
    git clone https://github.com/cage-challenge/cage-challenge-4.git cage-challenge-4
fi

# 2. Prevent torchaudio C++ extension mismatch on Colab when upgrading torch/transformers
python3 -m pip uninstall -y torchaudio >/dev/null 2>&1 || true

# 3. Install core dependencies while preserving Colab's native CUDA PyTorch build
echo "📦 Installing Lexsi Labs toolkits, TRL/PEFT/BitsAndBytes, and simulation packages..."
python3 -m pip install -q --upgrade \
    "numpy>=2.0.0,<3.0.0" \
    tabulate \
    openai \
    jsonschema \
    "gym==0.26.2" \
    "gymnasium>=0.29.1,<1.0.0" \
    "pygame>=2.5.2" \
    "prettytable>=3.9.0" \
    "networkx>=3.2.1" \
    "matplotlib>=3.8.0" \
    "tqdm>=4.66.0" \
    "pyyaml>=6.0" \
    "pytest>=8.0.0"

python3 -m pip install -q -r requirements.txt || {
    echo "ℹ️ Installing core HF + AlignTune packages with flexible Colab resolution..."
    python3 -m pip install -q \
        "aligntune>=0.1.30" \
        "safetune>=0.1.12" \
        "circuitkit>=0.2.0" \
        "curatorkit>=0.1.22" \
        "bitsandbytes>=0.45.5" \
        "peft>=0.18.0" \
        "trl>=1.7.1" \
        "accelerate>=1.6.0" \
        "datasets>=2.20.0" \
        "transformers>=4.46.0"
}

# 4. Install CybORG CAGE-4 in editable mode without overriding dependencies
echo "🛡️ Installing CybORG CAGE-4 environment..."
python3 -m pip install -q --no-deps -e cage-challenge-4

# 5. Verify GPU, Lexsi toolkits, and CybORG setup
echo "🔍 Running setup verification..."
python3 verify-setup.py

echo "=================================================================="
echo "✅ Google Colab GPU Setup Complete! Ready to run:"
echo "   python3 scripts/run_colab_pipeline.py --model-name meta-llama/Llama-3.1-8B-Instruct --episodes 30 --max-steps 120"
echo "=================================================================="

