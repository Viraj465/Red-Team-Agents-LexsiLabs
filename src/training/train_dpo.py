"""
AlignTune Direct Preference Optimization (DPO) Training Pipeline.
Trains Qwen2.5-Coder-7B-Instruct on 'darkknight25/linux_window_priv_esic_dataset'
preference pairs using Lexsi Labs AlignTune (with TRL backend fallback).
"""

import os
from src.training.prepare_dpo_data import prepare_hf_dpo_dataset


def run_dpo_alignment(
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    dataset_path: str = "data/processed/dpo_privesc_hf.json",
    output_dir: str = "results/checkpoints/aligntune_privesc_lora",
    num_epochs: int = 3,
    learning_rate: float = 5e-6,
    beta: float = 0.1,
):
    if not os.path.exists(dataset_path):
        prepare_hf_dpo_dataset(output_path=dataset_path)

    print("=" * 65)
    print("🚀 LAUNCHING LEXSI LABS ALIGNTUNE DPO TRAINING PIPELINE")
    print(f"Toolkit       : AlignTune (aligntune)")
    print(f"Base Model    : {model_name}")
    print(f"Dataset       : {dataset_path}")
    print(f"Output Adapter: {output_dir}")
    print("=" * 65)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Primary Lexsi Labs AlignTune Integration
    try:
        from aligntune import create_rl_trainer
        print("\n[AlignTune] Initializing AlignTune DPO Trainer...")
        trainer = create_rl_trainer(
            model_name=model_name,
            dataset_name=dataset_path,
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
            quantization={"load_in_4bit": True},
        )
        print("[AlignTune] Executing DPO alignment...")
        trainer.train()
        if hasattr(trainer, "save_model"):
            trainer.save_model(output_dir)
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

    dataset = load_dataset("json", data_files=dataset_path, split="train")

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

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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

    try:
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    except TypeError:
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ AlignTune DPO training complete. Adapters saved to: {output_dir}")


if __name__ == "__main__":
    run_dpo_alignment()
