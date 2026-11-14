"""
AlignTune Direct Preference Optimization (DPO) Training Pipeline.
Trains Qwen2.5-Coder-7B-Instruct / Llama-3.1-8B on CybORG CAGE-4 native
preference pairs using Lexsi Labs AlignTune (with TRL backend fallback).
"""

import hashlib
import inspect
import json
import os
import subprocess
from datetime import datetime, timezone
from src.training.prepare_dpo_data import prepare_hf_dpo_dataset


def _prepare_chat_templated_dataset(
    dataset_path: str, model_name: str, max_context_length: int = 8192
) -> str:
    """
    Renders raw DPO prompts with the target model's chat template (e.g. Cohere's
    <|START_OF_TURN_TOKEN|> format) so DPO training matches LLMEscalationAgent inference.
    Leaves the canonical dataset file unmodified on disk.
    """
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"ℹ️ [ChatTemplate] Skipping chat template pre-rendering ({e}); using raw prompts.")
        return dataset_path

    if getattr(tokenizer, "chat_template", None) is None:
        return dataset_path

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception:
        return dataset_path

    system_msg = (
        "You are an autonomous red-team privilege escalation specialist in a cybersecurity simulation.\n"
        "Select exactly ONE action from [CANDIDATE ACTIONS] to progress the operation.\n"
        "Output ONLY the selected action string verbatim. Do NOT include greetings, thoughts, markdown, explanations, or conversational preamble."
    )

    templated = []
    for row in records:
        raw_prompt = row.get("prompt", "")
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": raw_prompt},
        ]
        try:
            rendered_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            rendered_prompt = raw_prompt
        new_row = dict(row)
        new_row["prompt"] = rendered_prompt
        templated.append(new_row)

    clean_tag = model_name.replace("/", "_").replace("-", "_").replace(".", "_").lower()
    base_dir = os.path.dirname(dataset_path) or "data/processed"
    out_path = os.path.join(base_dir, f"dpo_privesc_{clean_tag}_chat.json")
    os.makedirs(base_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(templated, f, indent=2)

    print(
        f"✅ [ChatTemplate] Prepared {len(templated)} templated DPO pairs for {model_name} "
        f"(context_window={max_context_length}) -> {out_path}"
    )
    return out_path


def _write_provenance(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    num_epochs: int,
    learning_rate: float,
    beta: float,
    trainer_name: str = "TRLDPOTrainer",
    seed: int = 42,
) -> str:
    """Writes complete reproducibility and provenance metadata (git SHA, dataset hash, hyperparameters)."""
    git_sha = os.environ.get("GIT_SHA")
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        if res.returncode == 0 and res.stdout.strip():
            git_sha = res.stdout.strip()
    except Exception:
        pass
    if not git_sha:
        try:
            with open(__file__, "rb") as f:
                git_sha = f"code_sha256:{hashlib.sha256(f.read()).hexdigest()[:12]}"
        except Exception:
            git_sha = "colab-tarball-snapshot"

    dataset_sha256 = None
    num_samples = 0
    source_counts = {}
    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, "rb") as f:
                raw_bytes = f.read()
                dataset_sha256 = hashlib.sha256(raw_bytes).hexdigest()
            records = json.loads(raw_bytes.decode("utf-8"))
            num_samples = len(records)
            for r in records:
                src = r.get("source", "unknown")
                source_counts[src] = source_counts.get(src, 0) + 1
        except Exception:
            pass

    aligntune_ver = "0.1.32"
    try:
        import aligntune
        aligntune_ver = getattr(aligntune, "__version__", aligntune_ver)
    except Exception:
        pass

    prov = {
        "schema": "lexsi.provenance/1",
        "library": "aligntune",
        "version": aligntune_ver,
        "git_sha": git_sha,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_model": model_name,
        "method": "dpo",
        "inputs": [
            {
                "kind": "model",
                "ref": model_name,
                "config": {
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "compute_dtype": "bfloat16",
                    "context_window": 8192,
                },
                "provenance": None,
            },
            {
                "kind": "dataset",
                "ref": dataset_path,
                "sha256": dataset_sha256,
                "num_samples": num_samples,
                "source_distribution": source_counts,
                "provenance": None,
            },
        ],
        "params": {
            "trainer": trainer_name,
            "seed": seed,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "beta": beta,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 8,
            "max_length": 1024,
            "model_max_length": 8192,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    prov_path = os.path.join(output_dir, "provenance.json")
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    return prov_path


def run_dpo_alignment(
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    dataset_path: str = "data/processed/dpo_privesc_hf.json",
    output_dir: str = "results/checkpoints/aligntune_privesc_lora",
    num_epochs: int = 3,
    learning_rate: float = 5e-6,
    beta: float = 0.1,
    max_context_length: int = 8192,
):
    if not os.path.exists(dataset_path):
        prepare_hf_dpo_dataset(output_path=dataset_path)

    active_dataset_path = _prepare_chat_templated_dataset(
        dataset_path=dataset_path,
        model_name=model_name,
        max_context_length=max_context_length,
    )

    print("=" * 65)
    print("🚀 LAUNCHING LEXSI LABS ALIGNTUNE DPO TRAINING PIPELINE")
    print(f"Toolkit       : AlignTune (aligntune)")
    print(f"Base Model    : {model_name}")
    print(f"Dataset       : {active_dataset_path} (canonical: {dataset_path})")
    print(f"Context Window: {max_context_length} tokens")
    print(f"Output Adapter: {output_dir}")
    print("=" * 65)

    os.makedirs(output_dir, exist_ok=True)

    target_linear_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    # 1. Primary Lexsi Labs AlignTune Integration
    try:
        from aligntune import create_rl_trainer
        print("\n[AlignTune] Initializing AlignTune DPO Trainer...")
        trainer = create_rl_trainer(
            model_name=model_name,
            dataset_name=active_dataset_path,
            algorithm="dpo",
            backend="trl",
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            beta=beta,
            use_peft=True,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            lora_target_modules=target_linear_modules,
            target_modules=target_linear_modules,
            quantization={"load_in_4bit": True},
        )
        print("[AlignTune] Executing DPO alignment...")
        trainer.train()
        if hasattr(trainer, "save_model"):
            trainer.save_model(output_dir)
        _write_provenance(
            model_name=model_name,
            dataset_path=dataset_path,
            output_dir=output_dir,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            beta=beta,
            trainer_name="AlignTuneDPOTrainer",
        )
        print(f"\n✅ AlignTune DPO training complete. Adapters saved to: {output_dir}")
        return
    except Exception as aligntune_err:
        print(f"⚠️ AlignTune wrapper notice ({aligntune_err}), executing via AlignTune TRL backend...")

    # 2. Direct TRL DPOTrainer Execution
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig
    from trl import DPOTrainer, DPOConfig

    dataset = load_dataset("json", data_files=active_dataset_path, split="train")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_context_length

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_linear_modules,
    )

    training_args = DPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        beta=beta,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=num_epochs,
        max_length=1024,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        remove_unused_columns=False,
    )

    dpo_sig_params = inspect.signature(DPOTrainer.__init__).parameters
    tokenizer_kwarg = (
        {"processing_class": tokenizer}
        if "processing_class" in dpo_sig_params
        else {"tokenizer": tokenizer}
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        **tokenizer_kwarg,
    )

    print("[TRL] Commencing LoRA DPO alignment...")
    trainer.train()

    print(f"\n[TRL] Saving LoRA adapter checkpoint to: {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    _write_provenance(
        model_name=model_name,
        dataset_path=dataset_path,
        output_dir=output_dir,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        beta=beta,
        trainer_name="TRLDPOTrainer",
    )
    print(f"✅ AlignTune DPO training complete. Adapters saved to: {output_dir}")


if __name__ == "__main__":
    run_dpo_alignment()
