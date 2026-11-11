# 🎯 Adaptive Kill-Chain Router (AKCR)
### Hybrid Reinforcement Learning & AlignTune DPO for Autonomous Red-Teaming

[![Lexsi Labs Hackathon](https://img.shields.io/badge/Lexsi_Labs-Hackathon_2026-blueviolet.svg)](https://hackathon.lexsi.ai/tracks)
[![AlignTune](https://img.shields.io/badge/Powered_by-AlignTune_DPO-orange.svg)](https://github.com/Lexsi-Labs/aligntune)
[![CybORG CAGE-4](https://img.shields.io/badge/Benchmark-CybORG_CAGE--4-green.svg)](https://github.com/cage-challenge/cage-challenge-4)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📌 Executive Summary

Modern cybersecurity relies on autonomous defense agents (Blue Teams) to protect critical enterprise infrastructure. However, automated stress-testing tools face a fundamental dilemma:
1. **Pure Reinforcement Learning (RL)** excels at high-speed network reconnaissance and lateral movement, but consistently stalls at localized **privilege escalation** ($P(\text{Root} \mid \text{Comp}) < 15\%$), creating a dangerous false sense of defensive security.
2. **Pure Large Language Models (LLMs)** possess extensive semantic knowledge of system misconfigurations, but burn excessive tokens wandering aimlessly, suffer from hallucination, and fail to sustain long-horizon tactical campaigns.

**Adaptive Kill-Chain Router (AKCR)** solves this through an inverted, stage-gated orchestration framework:
- **Broad Tactical Operator:** Reinforcement Learning drives continuous subnet discovery, initial exploitation, and post-root operational impact.
- **Privilege Escalation Specialist:** An unprivileged shell foothold ($\text{UID} \neq 0$) automatically triggers a handoff to a fine-tuned, open-weight LLM (`Qwen2.5-Coder-7B-Instruct`).
- **Lexsi Labs AlignTune Alignment:** Using **Direct Preference Optimization (DPO)** via [`aligntune`](https://github.com/Lexsi-Labs/aligntune), the LLM is post-trained on real Linux privilege escalation primitives (GTFOBins, SUID misconfigurations, and CVEs) to eliminate conversational preamble and output deterministic, terminal-executable actions.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Env["CybORG CAGE-4 Enterprise Environment"]
        Obs["Raw Observation Vector & Active Sessions"]
        Actions["Simulated CybORG Action Space"]
    end

    subgraph Parser["State Parsing & Extraction"]
        SP["CyBORGStateParser"]
        Obs --> SP
        SP --> State["Discovered / Compromised / Rooted Host Sets"]
    end

    subgraph Router["Adaptive Kill-Chain Router (AKCR)"]
        State --> Gating{"Stage-Gating Trigger<br/>Host Compromised AND UID != 0<br/>AND Attempts < 5?"}
    end

    subgraph RL["Tactical Network Operator (RL)"]
        RLAgent["RLAgent<br/>(FiniteStateRedAgent / PPO)"]
        Gating -- "NO (Recon / Lateral / Impact)" --> RLAgent
        RLAgent --> ActRL["DiscoverRemoteSystems<br/>ExploitRemoteService<br/>Impact"]
    end

    subgraph LLM["Escalation Specialist (AlignTune DPO LLM)"]
        LLMAgent["LLMEscalationAgent<br/>(Qwen2.5-Coder-7B + LoRA)"]
        PromptGen["Format Privesc Prompt + Distractor Actions"]
        Gating -- "YES (Host Trapped at User Shell)" --> PromptGen
        PromptGen --> LLMAgent
        LLMAgent --> TermParse["Strict Terminal Parser<br/>(Command len < 80, no preamble)"]
        TermParse --> ActLLM["PrivilegeEscalate user_host"]
    end

    ActRL --> Actions
    ActLLM --> Actions
    Actions --> Env
```

---

## 🔬 Scientific Disentanglement: 3-Way Ablation

To rigorously prove *where* performance gains originate, AKCR evaluates three distinct configurations across identical random seeds:

| Configuration | Tactical Driver | Local Escalation Module | Scientific Question Isolated |
| :--- | :--- | :--- | :--- |
| **1. Baseline Pure RL** | RL (CybORG Heuristic / PPO) | Pure RL / Heuristic | Baseline operational capability without LLM reasoning. |
| **2. Hybrid + Frozen Base LLM** | RL (Tactical Operator) | Frozen `Qwen2.5-Coder-7B` | Isolates the **structural gain** of the stage-gated router. |
| **3. Hybrid + AlignTune DPO LLM** | RL (Tactical Operator) | `Qwen2.5-Coder-7B` + AlignTune LoRA | Isolates the **post-training alignment gain** of Lexsi Labs DPO. |

### Kill-Chain Transition Metrics
* $P(\text{Comp} \mid \text{Disc}) = \frac{\text{Hosts Compromised}}{\text{Hosts Discovered}}$: Lateral exploitation efficiency.
* $P(\text{Root} \mid \text{Comp}) = \frac{\text{Hosts Rooted}}{\text{Hosts Compromised}}$: **The core privilege escalation bottleneck.**
* $P(\text{Impact} \mid \text{Root}) = \frac{\text{Successful Impact Operations}}{\text{Hosts Rooted}}$: Mission completion capability.

---

## 🛠️ Lexsi Labs Toolchain Integration

This framework directly leverages the Lexsi Labs ecosystem:
- **`aligntune`**: Powers the DPO alignment pipeline (`src/training/train_dpo.py`). Fine-tunes `Qwen/Qwen2.5-Coder-7B-Instruct` using 4-bit NormalFloat (`nf4`) quantization and low-rank adapters (`r=16, alpha=32`), penalizing chatty conversational output and rewarding exact command syntax.
- **`safetune` & `curatorkit`**: Used to curate, sanitize, and validate preference pairs across Linux privilege escalation vectors in `src/training/prepare_dpo_data.py`.
- **`circuitkit`**: Model activation inspection and route verification.

---

## 📁 Repository Structure

```text
LexsiLabs/
├── app.py                         # Modal cloud orchestrator (L4 GPU training & 3-way ablation)
├── requirements.txt               # Pinned dependencies (aligntune, torch, transformers, peft, trl)
├── configs/                       # Centralized YAML configuration files
│   ├── env/cage4_scenario.yaml    # CybORG enterprise simulation settings
│   ├── agents/rl_ppo.yaml         # Tactical network operator hyperparameters
│   ├── agents/llm_agent.yaml      # LLM inference, quantization, and parsing constraints
│   ├── training/dpo_aligntune.yaml# AlignTune DPO training hyperparameters
│   └── eval.yaml                  # 3-Way ablation benchmark parameters
├── src/
│   ├── agents/
│   │   ├── base.py                # Abstract BaseAgent interface
│   │   ├── rl_agent.py            # Tactical RL / FiniteStateRedAgent wrapper
│   │   └── llm_agent.py           # Real 4-bit HF/LoRA inference + token telemetry
│   ├── env/
│   │   ├── cyborg_wrapper.py      # CybORG CAGE-4 environment wrapper with stage metrics
│   │   └── state_parser.py        # Host session extraction & privesc prompt generator
│   ├── evaluation/
│   │   ├── evaluator.py           # Multi-episode evaluator with randomized seeds
│   │   └── metrics.py             # Stage transition probability & token aggregator
│   ├── router/
│   │   ├── killchain_router.py    # AdaptiveKillChainRouter with realistic action pool
│   │   └── rules.py               # Formal containment rules & attempt budgets
│   └── training/
│       ├── prepare_dpo_data.py    # Pulls HF dataset & synthesizes CybORG DPO pairs
│       ├── train_dpo.py           # AlignTune DPO training pipeline
│       └── train_rl.py            # Tactical PPO training / checkpoint exporter
├── scripts/
│   ├── run_episode.py             # Single wargaming episode runner with live telemetry
│   ├── run_benchmark.py           # Full 3-way ablation benchmark runner
│   └── generate_report.py         # Visual chart generator (killchain_ablation.png)
└── tests/
    ├── conftest.py                # Pytest path and CybORG environment resolver
    └── test_router.py             # Unit tests for state transitions and fallback limits
```

---

## ⚡ Quickstart & Installation

### 1. Prerequisites & Environment Setup
```bash
# Clone the repository
git clone https://github.com/sawantviraj465/LexsiLabs.git
cd LexsiLabs

# Create and activate virtual environment (Python 3.12 recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate   # Linux/macOS

# Install dependencies and CybORG challenge submodule
pip install -r requirements.txt
pip install --no-deps -e cage-challenge-4
```

### 2. Verify Unit Tests
```bash
python -m pytest tests/test_router.py -v
```

---

## 🚀 Running Benchmarks & Training

### Local Verification Mode (Fast CPU Run, ~5 seconds)
Verify full router mechanics and multi-stage telemetry locally without needing a GPU:
```bash
# 1. Generate local DPO dataset
python src/training/prepare_dpo_data.py

# 2. Run benchmark sweep with mock specialist
python scripts/run_benchmark.py

# 3. Generate visual ablation report
python scripts/generate_report.py
```

### Cloud GPU Execution via Modal (L4 GPU, ~12–15 minutes)
Train `Qwen2.5-Coder-7B-Instruct` with AlignTune DPO and run live 3-way GPU ablation:
```bash
# Authenticate with Modal
modal setup

# Launch the end-to-end cloud pipeline
modal run app.py
```

The cloud runner will:
1. Stream `darkknight25/linux_window_priv_esic_dataset` from Hugging Face.
2. Train the 4-bit LoRA adapter using Lexsi Labs `aligntune`.
3. Persist checkpoint weights to Modal Volume `lexsi-experiment-cache`.
4. Run live multi-seed wargaming episodes comparing **Pure RL** vs. **Frozen Base LLM** vs. **AlignTune DPO LLM**.
5. Save the ablation report to `results/reports/ablation_results.md`.

---

## 📊 Benchmark Results

| Configuration | Episodes | $P(\text{Comp} \mid \text{Disc})$ | $P(\text{Root} \mid \text{Comp})$ | $P(\text{Impact} \mid \text{Root})$ | Impact Rate | Avg RL Steps | Avg LLM Steps | Avg Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline Pure RL** | 10 | 62.50% | **14.29%** | 100.00% | **10.00%** | 60.0 | 0.0 | 0 |
| **2. Hybrid + Frozen Base LLM** | 10 | 64.00% | **38.46%** | 100.00% | **30.00%** | 48.2 | 11.8 | 1,420 |
| **3. Hybrid + AlignTune DPO LLM** | 10 | 65.20% | **88.89%** | 100.00% | **80.00%** | 39.5 | 8.5 | 680 |

### Visual Ablation Chart
Visual charts generated by `scripts/generate_report.py` are saved directly to `results/reports/killchain_ablation.png`.

---

## 📜 References & Acknowledgements

* **Lexsi Labs Hackathon (2026):** [hackathon.lexsi.ai](https://hackathon.lexsi.ai/tracks)
* **AlignTune Toolkit:** [github.com/Lexsi-Labs/aligntune](https://github.com/Lexsi-Labs/aligntune)
* **CybORG CAGE Challenge 4:** Autonomous Cyber Operations Benchmark
* **Direct Preference Optimization:** Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023.

