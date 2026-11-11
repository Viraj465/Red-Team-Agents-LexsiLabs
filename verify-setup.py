# verify_setup.py
import sys
import torch

print(f"Python version: {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

# Test Lexsi Toolkits
try:
    import aligntune
    import safetune
    import circuitkit
    import curatorkit
    print("✅ Lexsi Labs toolkits (aligntune, safetune, circuitkit, curatorkit) imported successfully.")
except ImportError as e:
    print(f"⚠️ Lexsi toolkit import warning: {e}")

# Test CybORG Gym Environment
try:
    from CybORG import CybORG
    from CybORG.Agents import SleepAgent
    
    # Initialize a test CybORG instance
    env = CybORG()
    print("✅ CybORG CAGE-4 simulation environment loaded successfully.")
except Exception as e:
    print(f"⚠️ CybORG initialization error: {e}")

# Test Hugging Face Hub
try:
    import huggingface_hub
    from huggingface_hub import HfApi
    user = HfApi().whoami()
    print(f"✅ Hugging Face authenticated as: {user['name']}")
except Exception as e:
    print("⚠️ Hugging Face unauthenticated or offline. Run 'huggingface-cli login'.")