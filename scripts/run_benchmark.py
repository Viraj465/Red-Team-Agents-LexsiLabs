"""
Runs the 4-Quadrant Stage-Gated Kill-Chain Ablation Suite:
1. Baseline Pure RL (RL+RL, allow_privesc=False, privesc_success_prob=0.25)
2. Baseline Pure LLM (LLM+LLM)
3. Hybrid Router + Frozen Base LLM
4. Hybrid Router + AlignTune DPO LLM

Supports dual-horizon ablation (--max-steps 80 and 120),
two-proportion z-tests for statistical significance, and checkpoint resumption.
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Callable, Optional

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "cage-challenge-4"))

from tabulate import tabulate
from src.agents.rl_agent import RLAgent
from src.agents.llm_agent import LLMEscalationAgent
from src.agents.pure_llm_agent import PureLLMAgent
from src.router.killchain_router import AdaptiveKillChainRouter
from src.evaluation.evaluator import RedTeamEvaluator
from src.evaluation.metrics import two_proportion_z_test
from scripts.generate_report import plot_ablation_chart


def _load_checkpoint(checkpoint_file: str, target_episodes: int) -> list:
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                valid = [s for s in data if s.get("Episodes") == target_episodes]
                if valid:
                    print(f"🔄 [Resumption] Found checkpoint with {len(valid)} completed configurations.")
                return valid
        except Exception:
            return []
    return []


def _save_checkpoint(summaries: list, checkpoint_file: str):
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)


def run_comparative_benchmark(
    episodes_per_config: int = 10,
    max_steps: int = 80,
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    use_gpu_llm: bool = False,
    adapter_path: str = "results/checkpoints/aligntune_privesc_lora",
    checkpoint_file: str = "results/reports/ablation_checkpoint.json",
    fresh: bool = False,
    on_config_complete: Optional[Callable[[], None]] = None,
):
    print("=" * 80)
    print("🚀 RUNNING 4-QUADRANT KILL-CHAIN BENCHMARK & ABLATION SWEEP")
    print(f"Target LLM Model           : {model_name}")
    print(f"Episodes per configuration : {episodes_per_config}")
    print(
        f"LLM Inference Mode         : {'Live HF + AlignTune GPU' if use_gpu_llm else 'Local CPU Mock'}"
    )
    print(f"Horizon per Episode        : {max_steps} Steps")
    print("=" * 80)

    if fresh and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("🧹 [Fresh Run] Cleared existing ablation checkpoint.")

    summaries = _load_checkpoint(checkpoint_file, episodes_per_config)
    evaluator = RedTeamEvaluator(num_episodes=episodes_per_config, max_steps=max_steps)

    # -------------------------------------------------------------
    # 1. Baseline Pure RL (RL+RL)
    # -------------------------------------------------------------
    cfg1 = "1. Baseline Pure RL"
    if any(s.get("Configuration") == cfg1 for s in summaries):
        print(f"\n[1/4] Configuration '{cfg1}' already completed. Skipping and continuing...")
    else:
        print(f"\n[1/4] Evaluating Configuration 1: Baseline Pure RL (privesc_success_prob=0.25)...")
        pure_rl = RLAgent(policy_type="heuristic", allow_privesc=False, privesc_success_prob=0.25)
        rl_collector = evaluator.evaluate_agent(pure_rl, config_name=cfg1)
        summaries.append(rl_collector.compute_summary())
        _save_checkpoint(summaries, checkpoint_file)
        if on_config_complete:
            on_config_complete()

    # -------------------------------------------------------------
    # 2. Baseline Pure LLM (LLM+LLM)
    # -------------------------------------------------------------
    cfg2 = f"2. Baseline Pure LLM ({model_name.split('/')[-1]})"
    if any(s.get("Configuration") == cfg2 for s in summaries):
        print(f"\n[2/4] Configuration '{cfg2}' already completed. Skipping and continuing...")
    else:
        print(f"\n[2/4] Evaluating Configuration 2: Baseline Pure LLM ({model_name})...")
        pure_llm_backend = LLMEscalationAgent(
            model_mode="hf_base" if use_gpu_llm else "mock_base",
            base_model_name=model_name,
        )
        pure_llm = PureLLMAgent(llm_agent=pure_llm_backend)
        llm_collector = evaluator.evaluate_agent(pure_llm, config_name=cfg2)
        summaries.append(llm_collector.compute_summary())
        pure_llm_backend.unload_model()
        _save_checkpoint(summaries, checkpoint_file)
        if on_config_complete:
            on_config_complete()

    # -------------------------------------------------------------
    # 3. Hybrid Router + Frozen Base LLM
    # -------------------------------------------------------------
    cfg3 = f"3. Hybrid + Frozen Base ({model_name.split('/')[-1]})"
    if any(s.get("Configuration") == cfg3 for s in summaries):
        print(f"\n[3/4] Configuration '{cfg3}' already completed. Skipping and continuing...")
    else:
        print(f"\n[3/4] Evaluating Configuration 3: Hybrid Router + Frozen Base ({model_name})...")
        base_llm = LLMEscalationAgent(
            model_mode="hf_base" if use_gpu_llm else "mock_base",
            base_model_name=model_name,
        )
        router_base = AdaptiveKillChainRouter(
            rl_agent=RLAgent(allow_privesc=False, privesc_success_prob=0.25), llm_agent=base_llm
        )
        base_collector = evaluator.evaluate_agent(
            router_base, config_name=cfg3
        )
        summaries.append(base_collector.compute_summary())
        base_llm.unload_model()
        _save_checkpoint(summaries, checkpoint_file)
        if on_config_complete:
            on_config_complete()

    # -------------------------------------------------------------
    # 4. Hybrid Router + AlignTune DPO Fine-Tuned LLM
    # -------------------------------------------------------------
    cfg4 = f"4. Hybrid + AlignTune DPO ({model_name.split('/')[-1]})"
    if any(s.get("Configuration") == cfg4 for s in summaries):
        print(f"\n[4/4] Configuration '{cfg4}' already completed. Skipping and continuing...")
    else:
        print(f"\n[4/4] Evaluating Configuration 4: Hybrid Router + AlignTune DPO ({model_name})...")
        dpo_llm = LLMEscalationAgent(
            model_mode="hf_dpo" if use_gpu_llm else "mock_dpo",
            base_model_name=model_name,
            adapter_path=adapter_path,
        )
        router_dpo = AdaptiveKillChainRouter(
            rl_agent=RLAgent(allow_privesc=False, privesc_success_prob=0.25), llm_agent=dpo_llm
        )
        dpo_collector = evaluator.evaluate_agent(
            router_dpo, config_name=cfg4
        )
        summaries.append(dpo_collector.compute_summary())
        dpo_llm.unload_model()
        _save_checkpoint(summaries, checkpoint_file)
        if on_config_complete:
            on_config_complete()

    # Format Markdown Table (excluding internal raw dictionary)
    headers = [k for k in summaries[0].keys() if not k.startswith("_")]
    rows = [[s[k] for k in headers] for s in summaries]
    md_table = tabulate(rows, headers=headers, tablefmt="github")

    # Statistical Significance Testing (Two-Proportion Z-Test with Explicit Numerators/Denominators)
    sig_notes = []
    try:
        cfg_map = {s["Configuration"]: s.get("_raw", {}) for s in summaries}

        def _fmt_z(label: str, c_a: int, n_a: int, c_b: int, n_b: int) -> str:
            z_val, p_val = two_proportion_z_test(c_a, n_a, c_b, n_b)
            p_a = (c_a / n_a * 100) if n_a > 0 else 0.0
            p_b = (c_b / n_b * 100) if n_b > 0 else 0.0
            p_str = "p < 0.0001" if p_val < 0.0001 else f"p = {p_val:.4f}"
            sig_str = "p < 0.05, statistically significant" if p_val < 0.05 else "p >= 0.05"
            return (
                f"- **{label}**: {c_a}/{n_a} ({p_a:.1f}%) vs {c_b}/{n_b} ({p_b:.1f}%), "
                f"z = {z_val:.2f}, {p_str} ({sig_str})"
            )

        if cfg1 in cfg_map and cfg3 in cfg_map:
            sig_notes.append(
                _fmt_z(
                    "Hybrid Frozen vs Pure RL [P(Root|Comp)]",
                    cfg_map[cfg3].get("total_root", 0),
                    cfg_map[cfg3].get("total_compro", 0),
                    cfg_map[cfg1].get("total_root", 0),
                    cfg_map[cfg1].get("total_compro", 0),
                )
            )
        if cfg1 in cfg_map and cfg4 in cfg_map:
            sig_notes.append(
                _fmt_z(
                    "Hybrid DPO vs Pure RL [P(Root|Comp)]",
                    cfg_map[cfg4].get("total_root", 0),
                    cfg_map[cfg4].get("total_compro", 0),
                    cfg_map[cfg1].get("total_root", 0),
                    cfg_map[cfg1].get("total_compro", 0),
                )
            )
        if cfg3 in cfg_map and cfg4 in cfg_map:
            sig_notes.append(
                _fmt_z(
                    "Hybrid DPO vs Hybrid Frozen [P(Root|Comp)]",
                    cfg_map[cfg4].get("total_root", 0),
                    cfg_map[cfg4].get("total_compro", 0),
                    cfg_map[cfg3].get("total_root", 0),
                    cfg_map[cfg3].get("total_compro", 0),
                )
            )
        if cfg1 in cfg_map and cfg4 in cfg_map:
            sig_notes.append(
                _fmt_z(
                    "Hybrid DPO vs Pure RL [P(Impact|Root)]",
                    cfg_map[cfg4].get("total_impacted", 0),
                    cfg_map[cfg4].get("total_root", 0),
                    cfg_map[cfg1].get("total_impacted", 0),
                    cfg_map[cfg1].get("total_root", 0),
                )
            )
        if cfg3 in cfg_map and cfg4 in cfg_map:
            sig_notes.append(
                _fmt_z(
                    "Hybrid DPO vs Hybrid Frozen [P(Impact|Root)]",
                    cfg_map[cfg4].get("total_impacted", 0),
                    cfg_map[cfg4].get("total_root", 0),
                    cfg_map[cfg3].get("total_impacted", 0),
                    cfg_map[cfg3].get("total_root", 0),
                )
            )
        if cfg1 in cfg_map and cfg4 in cfg_map:
            sig_notes.append(
                _fmt_z(
                    "Hybrid DPO vs Pure RL [End-to-End Yield P(Impact|Disc)]",
                    cfg_map[cfg4].get("total_impacted", 0),
                    cfg_map[cfg4].get("total_discov", 0),
                    cfg_map[cfg1].get("total_impacted", 0),
                    cfg_map[cfg1].get("total_discov", 0),
                )
            )
    except Exception as e:
        sig_notes.append(f"Statistical significance calculation notice: {e}")

    sig_text = "\n".join(sig_notes) if sig_notes else "No statistical tests computed."

    print("\n" + "=" * 80)
    print("📊 4-QUADRANT KILL-CHAIN EMPIRICAL ABLATION RESULTS")
    print("=" * 80)
    print(md_table)
    print("\n📈 Statistical Significance Tests:")
    print(sig_text)
    print("=" * 80)

    report_dir = os.path.dirname(checkpoint_file) if checkpoint_file else "results/reports"
    os.makedirs(report_dir, exist_ok=True)
    clean_tag = model_name.replace("/", "_").replace("-", "_").replace(".", "_").lower()

    # Save generic and model-tagged summaries
    summary_json_file = os.path.join(report_dir, "ablation_summary.json")
    model_summary_file = os.path.join(report_dir, f"ablation_{clean_tag}_summary.json")
    for fpath in (summary_json_file, model_summary_file):
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)

    report_file = os.path.join(report_dir, "ablation_results.md")
    model_report_file = os.path.join(report_dir, f"ablation_{clean_tag}_results.md")
    for rpath in (report_file, model_report_file):
        with open(rpath, "w", encoding="utf-8") as f:
            f.write(f"# Empirical 4-Quadrant Kill-Chain Ablation Results ({max_steps} Steps)\n\n")
            f.write(md_table + "\n\n")
            f.write("### Statistical Significance Matrix (Two-Proportion Z-Test)\n\n")
            f.write(sig_text + "\n")

    chart_file = os.path.join(report_dir, "killchain_ablation_chart.png")
    model_chart_file = os.path.join(report_dir, f"killchain_{clean_tag}_chart.png")
    try:
        plot_ablation_chart(summaries, output_png=chart_file)
        if chart_file != model_chart_file:
            plot_ablation_chart(summaries, output_png=model_chart_file)
    except Exception as e:
        print(f"Chart plotting notice: {e}")

    print(f"✅ Saved JSON logs to {summary_json_file} and {model_summary_file}")
    print(f"✅ Saved Markdown report to {report_file} and {model_report_file}")
    return md_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes per configuration")
    parser.add_argument("--max-steps", type=int, default=80, help="Maximum steps per episode horizon")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--adapter-path", type=str, default="results/checkpoints/aligntune_privesc_lora", help="Path to LoRA adapter")
    parser.add_argument("--force-gpu", action="store_true", help="Force GPU inference mode")
    parser.add_argument("--fresh", action="store_true", help="Clear checkpoints and re-run all configurations")
    args = parser.parse_args()

    import torch
    has_gpu = args.force_gpu or (
        torch.cuda.is_available() and os.path.exists(args.adapter_path)
    )
    run_comparative_benchmark(
        episodes_per_config=args.episodes,
        max_steps=args.max_steps,
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        use_gpu_llm=has_gpu,
        fresh=args.fresh,
    )
