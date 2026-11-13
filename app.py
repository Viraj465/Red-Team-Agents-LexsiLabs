# app.py
"""
Modal Cloud Orchestrator for Lexsi Labs Hybrid Red-Team Framework.
Orchestrates:
1. Dual-Vector DPO Dataset preparation from darkknight25/linux_window_priv_esic_dataset
2. AlignTune DPO training of target LLM (Qwen, Llama 3.1) with 4-bit LoRA (with volume cache detection)
3. 4-Quadrant Comparative Stage-Gated Kill-Chain Benchmark on an L4 GPU (N=30 seeds)
4. Continuous checkpointing per configuration and publication-grade figure generation
"""

import os
import sys
import subprocess
import modal

app = modal.App("lexsi-hybrid-redteam")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .pip_install(
        "torch",
        "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install_from_requirements(
        "requirements.txt",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "tabulate",
        "matplotlib",
        "gym==0.26.2",
        "pygame",
        "prettytable",
        "tqdm",
    )
    # Install CybORG CAGE-4 without overwriting PyTorch or other dependencies
    .run_commands(
        "git clone https://github.com/cage-challenge/cage-challenge-4.git /root/cage-challenge-4",
        "pip install --no-deps -e /root/cage-challenge-4",
    )
    .add_local_dir(
        ".",
        remote_path="/root/LexsiLabs",
        ignore=[".venv", ".git", "results", "__pycache__", "*.pyc"],
    )
)

volume = modal.Volume.from_name("lexsi-experiment-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(
    image=image,
    gpu="L4",
    secrets=[hf_secret],
    volumes={"/root/LexsiLabs/results": volume},
    timeout=86400,  # 24-hour timeout (Modal maximum) to allow full 30-episode benchmarks
    min_containers=1,  # Keep container warm for seamless continuation
)
def train_and_benchmark(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    episodes: int = 30,
    max_steps: int = 120,
    force_train: bool = False,
    fresh: bool = False,
):
    # 1. Hot-fix: Ensure numpy is >= 2.0 in the container before importing pyarrow/trl
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "numpy>=2.0.0",
            "openai",
            "jsonschema",
            "gym==0.26.2",
            "pygame",
            "prettytable",
            "--no-cache-dir",
        ],
        check=False,
    )

    # 2. Hot-fix: Remove mismatched torchaudio so transformers doesn't crash on _torchaudio.so
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"], check=False)

    os.chdir("/root/LexsiLabs")
    sys.path.insert(0, "/root/LexsiLabs")
    sys.path.insert(0, "/root/cage-challenge-4")

    from src.training.prepare_dpo_data import prepare_hf_dpo_dataset
    from src.training.train_dpo import run_dpo_alignment
    from scripts.run_benchmark import run_comparative_benchmark

    clean_tag = model_name.replace("/", "_").replace("-", "_").replace(".", "_").lower()
    adapter_dir = f"/root/LexsiLabs/results/checkpoints/aligntune_{clean_tag}_v2_cyborg_lora"
    checkpoint_file = f"/root/LexsiLabs/results/reports/ablation_{clean_tag}_v2_checkpoint.json"

    # Step 1: Build Consolidated HF + CybORG DPO Dataset
    dataset_file = prepare_hf_dpo_dataset("data/processed/dpo_privesc_hf.json")

    # Step 2: Run AlignTune DPO Training (check if already trained on volume)
    adapter_weights = os.path.join(adapter_dir, "adapter_model.safetensors")
    adapter_config = os.path.join(adapter_dir, "adapter_config.json")
    if (os.path.exists(adapter_config) or os.path.exists(adapter_weights)) and not force_train:
        print(f"✅ Found existing AlignTune DPO LoRA adapter at {adapter_dir}. Skipping re-training.")
    else:
        print(f"🚀 Training AlignTune DPO LoRA adapter for {model_name}...")
        run_dpo_alignment(
            model_name=model_name,
            dataset_path=dataset_file,
            output_dir=adapter_dir,
            num_epochs=3,
        )
        volume.commit()

    # Step 3: Run Full 4-Quadrant Ablation Benchmark with Live GPU Inference & Intermediate Volume Commits
    table_md = run_comparative_benchmark(
        episodes_per_config=episodes,
        max_steps=max_steps,
        model_name=model_name,
        use_gpu_llm=True,
        adapter_path=adapter_dir,
        checkpoint_file=checkpoint_file,
        fresh=fresh,
        on_config_complete=volume.commit,
    )
    volume.commit()
    return table_md


@app.function(
    image=image,
    gpu="L4",
    secrets=[hf_secret],
    volumes={"/root/LexsiLabs/results": volume},
    timeout=7200,
)
def train_model(model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
    """Standalone trainer function for running training only."""
    return train_and_benchmark(model_name=model_name, episodes=0, force_train=True)


@app.local_entrypoint()
def main(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    episodes: int = 30,
    max_steps: int = 120,
    force_train: bool = False,
    fresh: bool = False,
):
    print("=" * 80)
    print(f"🚀 LAUNCHING MODAL RUN ON NVIDIA L4 GPU")
    print(f"Model ID   : {model_name}")
    print(f"Episodes   : {episodes}")
    print(f"Max Steps  : {max_steps}")
    print(f"Force Train: {force_train}")
    print(f"Fresh Sweep: {fresh}")
    print("=" * 80)
    table = train_and_benchmark.remote(
        model_name=model_name,
        episodes=episodes,
        max_steps=max_steps,
        force_train=force_train,
        fresh=fresh,
    )
    print("\n🎉 MODAL PIPELINE COMPLETE! FINAL ABLATION TABLE:\n")
    print(table)
