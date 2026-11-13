"""
Google Colab GPU End-to-End Orchestrator for Lexsi Labs Hybrid Red-Team Framework.
Replaces Modal cloud execution with native Google Colab GPU execution (T4 / L4 / A100).

Features:
1. Automatic Hugging Face authentication via HF_TOKEN env var or Colab Secrets.
2. Optional Google Drive checkpoint synchronization so long 30-episode sweeps
   survive Colab session timeouts/disconnects.
3. End-to-end execution:
   - Step 1: CybORG-native DPO preference dataset validation/generation
   - Step 2: AlignTune 4-bit LoRA DPO training (with full provenance metadata)
   - Step 3: 4-Quadrant Stage-Gated Kill-Chain Ablation Benchmark (N=30 seeds, 120 steps)
   - Step 4: Artifact packaging into results/<model_tag>_v2/ & cross-model report generation
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "cage-challenge-4"))


def _configure_hf_auth():
    """Authenticates with Hugging Face using HF_TOKEN env var or Google Colab Secrets."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from google.colab import userdata  # type: ignore
            token = userdata.get("HF_TOKEN")
            if token:
                os.environ["HF_TOKEN"] = token
                print("🔑 [Colab] Loaded HF_TOKEN from Google Colab Secrets.")
        except Exception:
            pass

    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            print("✅ [Hugging Face] Authenticated via token.")
        except Exception as e:
            print(f"⚠️ [Hugging Face] Token login warning: {e}")
    else:
        print(
            "ℹ️ [Hugging Face] No HF_TOKEN found in environment. "
            "For gated models like meta-llama/Llama-3.1-8B-Instruct, export HF_TOKEN='hf_...' first."
        )


def _sync_to_drive(local_dir: str, drive_dir: str):
    """Copies updated checkpoints and reports to Google Drive if mounted."""
    if not drive_dir:
        return
    try:
        os.makedirs(drive_dir, exist_ok=True)
        if os.path.exists(local_dir):
            shutil.copytree(local_dir, drive_dir, dirs_exist_ok=True)
            print(f"💾 [Colab Drive Sync] Synced artifacts to: {drive_dir}")
    except Exception as e:
        print(f"⚠️ [Colab Drive Sync] Warning: {e}")


def run_colab_pipeline(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    episodes: int = 30,
    max_steps: int = 120,
    num_epochs: int = 3,
    force_train: bool = False,
    fresh: bool = False,
    drive_backup_dir: str = "",
):
    import torch
    from src.training.prepare_dpo_data import prepare_hf_dpo_dataset
    from src.training.train_dpo import run_dpo_alignment
    from scripts.run_benchmark import run_comparative_benchmark
    from scripts.generate_cross_model_report import generate_report

    print("=" * 80)
    print("🚀 LEXSILABS GOOGLE COLAB GPU PIPELINE")
    print(f"Model ID        : {model_name}")
    print(f"Episodes / Cfg  : {episodes}")
    print(f"Max Steps / Ep  : {max_steps}")
    print(f"CUDA Available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device      : {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU VRAM        : {vram_gb:.2f} GB")
    print("=" * 80)

    _configure_hf_auth()

    clean_tag = model_name.replace("/", "_").replace("-", "_").replace(".", "_").lower()
    adapter_dir = str(PROJECT_ROOT / "results" / "checkpoints" / f"aligntune_{clean_tag}_v2_cyborg_lora")
    bundle_dir = str(PROJECT_ROOT / "results" / f"{clean_tag}_v2")
    os.makedirs(bundle_dir, exist_ok=True)
    checkpoint_file = os.path.join(bundle_dir, f"ablation_{clean_tag}_v2_checkpoint.json")

    # Restore from Google Drive backup if resuming on a fresh Colab VM
    if drive_backup_dir and os.path.exists(drive_backup_dir) and not fresh:
        print(f"🔄 [Colab Drive Sync] Checking Drive backup at {drive_backup_dir}...")
        try:
            shutil.copytree(drive_backup_dir, str(PROJECT_ROOT / "results"), dirs_exist_ok=True)
        except Exception as e:
            print(f"⚠️ [Colab Drive Sync] Restore notice: {e}")

    # Step 1: Prepare / Validate CybORG-Native DPO Dataset
    dataset_file = prepare_hf_dpo_dataset(str(PROJECT_ROOT / "data" / "processed" / "dpo_privesc_hf.json"))

    # Step 2: Run AlignTune DPO Training
    adapter_weights = os.path.join(adapter_dir, "adapter_model.safetensors")
    adapter_config = os.path.join(adapter_dir, "adapter_config.json")
    if (os.path.exists(adapter_config) or os.path.exists(adapter_weights)) and not force_train:
        print(f"✅ Found existing AlignTune DPO LoRA adapter at {adapter_dir}. Skipping re-training.")
    else:
        if not torch.cuda.is_available():
            print("⚠️ No GPU detected; skipping live HF LoRA training (will use mock mode if benchmarking on CPU).")
        else:
            print(f"🚀 Training AlignTune DPO LoRA adapter for {model_name}...")
            run_dpo_alignment(
                model_name=model_name,
                dataset_path=dataset_file,
                output_dir=adapter_dir,
                num_epochs=num_epochs,
            )
            _sync_to_drive(str(PROJECT_ROOT / "results"), drive_backup_dir)

    # Copy adapter_config.json and provenance.json into bundle_dir for complete reproducibility
    for meta_fname, dest_fname in [
        ("adapter_config.json", "adapter_config.json"),
        ("provenance.json", f"{clean_tag}_v2_provenance.json"),
    ]:
        src_meta = os.path.join(adapter_dir, meta_fname)
        if os.path.exists(src_meta):
            shutil.copy2(src_meta, os.path.join(bundle_dir, dest_fname))

    if episodes <= 0:
        print("✅ Training complete (episodes=0). Exiting.")
        return

    # Step 3: Run 4-Quadrant Ablation Benchmark
    use_gpu = torch.cuda.is_available()
    table_md = run_comparative_benchmark(
        episodes_per_config=episodes,
        max_steps=max_steps,
        model_name=model_name,
        use_gpu_llm=use_gpu,
        adapter_path=adapter_dir,
        checkpoint_file=checkpoint_file,
        fresh=fresh,
        on_config_complete=lambda: _sync_to_drive(str(PROJECT_ROOT / "results"), drive_backup_dir),
    )

    # Step 4: Also mirror outputs into results/meta_llama_3_1_8b_v2 if running Llama-3.1-8B
    if "llama" in clean_tag and "8b" in clean_tag:
        llama_v2_dir = str(PROJECT_ROOT / "results" / "meta_llama_3_1_8b_v2")
        if os.path.abspath(bundle_dir) != os.path.abspath(llama_v2_dir):
            os.makedirs(llama_v2_dir, exist_ok=True)
            shutil.copytree(bundle_dir, llama_v2_dir, dirs_exist_ok=True)

    try:
        generate_report()
    except Exception as e:
        print(f"ℹ️ Cross-model report generation notice: {e}")

    _sync_to_drive(str(PROJECT_ROOT / "results"), drive_backup_dir)

    print("\n🎉 COLAB GPU PIPELINE COMPLETE! ARTIFACTS SAVED TO:")
    print(f"   📂 {bundle_dir}")
    return table_md


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LexsiLabs DPO + 4-Quadrant Ablation on Google Colab GPU")
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face model ID (e.g. meta-llama/Llama-3.1-8B-Instruct or Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument("--episodes", type=int, default=30, help="Episodes per configuration (default: 30)")
    parser.add_argument("--max-steps", type=int, default=120, help="Horizon steps per episode (default: 120)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of DPO training epochs (default: 3)")
    parser.add_argument("--force-train", action="store_true", help="Force re-training of DPO LoRA adapter")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing benchmark checkpoints and start fresh")
    parser.add_argument(
        "--drive-backup-dir",
        type=str,
        default="",
        help="Optional Google Drive path (e.g. /content/drive/MyDrive/LexsiLabs_Results) for automatic checkpoint sync",
    )
    args = parser.parse_args()

    run_colab_pipeline(
        model_name=args.model_name,
        episodes=args.episodes,
        max_steps=args.max_steps,
        num_epochs=args.epochs,
        force_train=args.force_train,
        fresh=args.fresh,
        drive_backup_dir=args.drive_backup_dir,
    )

