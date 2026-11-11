LexsiLabs/
│
├── .github/                     # CI/CD workflows and actions (optional)
├── .gitignore
├── requirements.txt             # Core Python dependencies
├── app.py                       # Modal deployment & cloud execution harness
├── verify-setup.py              # Environment sanity check script
│
├── configs/                     # YAML/JSON configs for experiments & hyperparameters
│   ├── env/
│   │   └── cage4_scenario.yaml  # CybORG scenario & network topology settings
│   ├── agents/
│   │   ├── rl_ppo.yaml          # PPO policy hyperparameters (lr, clip, gamma)
│   │   └── llm_agent.yaml       # LLM inference params (temp, max_tokens, prompt templates)
│   ├── training/
│   │   └── dpo_aligntune.yaml   # AlignTune DPO training config (epochs, beta, lora_r)
│   └── eval.yaml                # Evaluation parameters (num_episodes, seeds, timeout)
│
├── data/                        # Datasets & synthetic generation artifacts
│   ├── raw/                     # Raw CybORG trajectory logs / privilege escalation logs
│   ├── processed/               # Cleaned SFT and DPO preference pairs (chosen vs rejected)
│   └── prompts/                 # Prompt templates for privilege escalation
│
├── src/                         # Core codebase
│   ├── __init__.py
│   │
│   ├── env/                     # CybORG environment wrappers & state parsers
│   │   ├── __init__.py
│   │   ├── cyborg_wrapper.py    # Standardized gym wrapper over CybORG CAGE-4
│   │   └── state_parser.py      # Translates CybORG observation dict to text & vector states
│   │
│   ├── agents/                  # The modular Red Team agents
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract Agent base class (step, reset, get_action)
│   │   ├── rl_agent.py          # PPO scout/lateral/impact operator
│   │   └── llm_agent.py         # Fine-tuned LLM privilege escalation agent
│   │
│   ├── router/                  # Kill-Chain Router (The Core Innovation)
│   │   ├── __init__.py
│   │   ├── killchain_router.py  # Stage-gated router (decides RL <-> LLM handoff)
│   │   └── rules.py             # Heuristic / threshold triggers (e.g. host compromised -> call LLM)
│   │
│   ├── training/                # Training pipelines
│   │   ├── __init__.py
│   │   ├── train_rl.py          # PPO training loop for RL agent
│   │   ├── prepare_dpo_data.py  # Generates chosen/rejected pairs from trajectories
│   │   └── train_dpo.py         # AlignTune SFT/DPO alignment pipeline for the LLM
│   │
│   └── evaluation/              # Stage-wise ablation & metrics collection
│       ├── __init__.py
│       ├── metrics.py           # Calculates P(Root|Compromise), P(Impact|Root), token costs
│       └── evaluator.py         # Benchmarks 4 configurations (RL-only, LLM-only, etc.)
│
├── scripts/                     # CLI entry points to run tasks
│   ├── run_episode.py           # Run a single interactive hybrid episode (with logs/visualizer)
│   ├── run_benchmark.py         # Run full stage-wise ablation benchmark suite
│   └── generate_report.py       # Generates metric charts & markdown tables for presentation
│
├── results/                     # Experiment artifacts (ignored in git)
│   ├── checkpoints/             # Saved model weights (LoRA adapters, PPO policies)
│   ├── logs/                    # Episode execution logs and handoff traces
│   └── reports/                 # Evaluation figures, win-rate charts, ablation tables
│
└── cage-challenge-4/            # CybORG submodule / source clone