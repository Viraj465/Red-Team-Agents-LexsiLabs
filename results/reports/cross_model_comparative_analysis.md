# Consolidated Cross-Model Kill-Chain Empirical Benchmark (30 Episodes, 120-Step Horizon)

## Executive Summary

Empirical validation across **30 distinct CAGE-4 random seeds** and **120-step horizons** on real **NVIDIA L4 GPU hardware** demonstrates:
1. **Privilege Escalation Parity (Qwen2.5-7B-Instruct)**: Hybrid + Frozen Base achieves **59.9%** root conversion versus Pure RL's 61.4% ($z = -0.33, p = 0.7448$), confirming statistical parity on post-fix Qwen2.5-7B. On post-remediation Llama-3.1-8B, Frozen Base achieves **89.2%** root conversion ($z = 5.97, p < 0.0001$ vs Pure RL's 65.3%).
2. **Statistically Significant Superiority on Operational Weaponization**: Hybrid routing achieves **100.0% Impact Rate** across all 30 episodes for both models. Hybrid DPO significantly outperforms Pure RL in post-root weaponization ($P(\text{Impact}\mid\text{Root})$ = **73.7%** vs 60.7%, $z = +2.08, p = 0.0379$ on Qwen; **98.9%** vs 61.5%, $z = +9.03, p < 0.0001$ on Llama).
3. **Decisive Token & Latency Economics**: The hybrid router reduces token consumption by **89.7% to 97.0%** compared to Pure LLM (4,207 vs 40,832 tokens on Qwen; 2,172 vs 72,317 tokens on Llama-3.1-8B-Instruct) and accelerates execution by **6.3x to 10.1x** (11.89s vs 120.38s on Llama).
4. **End-to-End Yield Doubling**: On Llama-3.1-8B-Instruct, Hybrid DPO more than doubles overall end-to-end host disruption yield ($P(\text{Impact}\mid\text{Disc})$ = **32.1%** [188/585, 6.27 impacted hosts/ep] vs Pure RL's **15.7%** [96/613, 3.20 impacted hosts/ep], $z = +6.70, p < 0.0001$).

## Cross-Model Comparative Results Table

| Model Family | Configuration | P(Comp|Disc) | P(Root|Comp) | P(Impact|Root) | Impact Rate | Avg RL Steps | Avg LLM Steps | Avg Tokens | Avg Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen2.5-7B** | 1. Baseline Pure RL | 38.8±5.7% | 61.4±23.1% | 60.7±29.7% | **90.0%** | 120.0±0.0 | 0.0±0.0 | 0±0 | 4.22±0.56s |
| **Qwen2.5-7B** | 2. Baseline Pure LLM (Qwen2.5-7B-Instruct) | 51.2±18.6% | 0.0±0.0% | 0.0±0.0% | **0.0%** | 0.0±0.0 | 120.0±0.0 | 40832±5319 | 141.31±8.12s |
| **Qwen2.5-7B** | 3. Hybrid + Frozen Base (Qwen2.5-7B-Instruct) | 37.6±5.7% | 59.9±19.4% | 78.3±23.7% | **100.0%** | 99.0±6.9 | 21.0±6.9 | 4207±1387 | 22.28±5.83s |
| **Qwen2.5-7B** | 4. Hybrid + AlignTune DPO (Qwen2.5-7B-Instruct) | 38.8±6.3% | 48.5±16.0% | 73.7±22.7% | **100.0%** | 95.9±6.2 | 24.1±6.2 | 4827±1234 | 24.81±5.36s |
| **Llama-3.1-8B (Remediated)** | 1. Baseline Pure RL | 39.0±5.3% | 65.3±22.9% | 61.5±23.5% | **100.0%** | 120.0±0.0 | 0.0±0.0 | 0±0 | 3.35±0.47s |
| **Llama-3.1-8B (Remediated)** | 2. Baseline Pure LLM (Llama-3.1-8B-Instruct) | 24.9±9.9% | 49.4±26.9% | 0.0±0.0% | **0.0%** | 0.0±0.0 | 120.0±0.0 | 72317±13342 | 120.38±3.71s |
| **Llama-3.1-8B (Remediated)** | 3. Hybrid + Frozen Base (Llama-3.1-8B-Instruct) | 36.8±5.5% | 89.2±14.7% | 98.4±4.5% | **100.0%** | 110.3±4.6 | 9.7±4.6 | 2180±1049 | 11.89±4.20s |
| **Llama-3.1-8B (Remediated)** | 4. Hybrid + AlignTune DPO (Llama-3.1-8B-Instruct) | 36.1±5.7% | 90.0±14.0% | 98.9±3.3% | **100.0%** | 110.3±4.6 | 9.7±4.6 | 2172±1061 | 11.89±4.32s |

## Two-Proportion Z-Test Statistical Matrix (30 Seeds, $n \approx 180-240$ Attempts)

| Comparison Hypothesis | Metric | Qwen 2.5-7B ($z, p$) | Qwen Significance | Llama 3.1-8B ($z, p$) | Llama Significance |
|---|---|---|---|---|---|
| **Hybrid Frozen vs Pure RL** | P(Root|Comp) | $z = -0.33, p = 0.7448$ | Parity ($p > 0.05$) | $z = 5.97, p < 0.0001$ | Significant ($p < 0.05$) |
| **Hybrid DPO vs Pure RL** | P(Root|Comp) | $z = -2.70, p = 0.0069$ | $p < 0.01$ | $z = 6.22, p < 0.0001$ | Significant ($p < 0.05$) |
| **Hybrid DPO vs Hybrid Frozen** | P(Root|Comp) | $z = -2.26, p = 0.0239$ | $p < 0.05$ (Frozen superior) | $z = 0.30, p = 0.7627$ | Parity ($p > 0.05$) |
| **Hybrid Frozen vs Pure RL** | P(Impact|Root) | $z = 3.03, p = 0.0025$ | **Significant ($p < 0.01$)** | $z = 8.82, p < 0.0001$ | **Significant ($p < 0.01$)** |
| **Hybrid DPO vs Pure RL** | P(Impact|Root) | $z = 2.08, p = 0.0379$ | **Significant ($p < 0.05$)** | $z = 9.03, p < 0.0001$ | **Significant ($p < 0.01$)** |
| **Hybrid DPO vs Hybrid Frozen** | P(Impact|Root) | $z = -0.78, p = 0.4383$ | Parity ($p > 0.05$) | $z = 0.46, p = 0.6483$ | Parity ($p > 0.05$) |
| **Hybrid vs Pure RL** | Impact Rate | $z = 1.78, p = 0.0756$ | $100.0\% \text{ vs } 90.0\%$ | $z = 0.00, p = 1.0000$ | $100.0\% \text{ vs } 100.0\%$ |

## Methodological & Telemetry Notes

- **Handoff Transition Accounting (`avg_handoff_transitions = 18.3` for `9.7` LLM steps)**: The router increments `handoff_transitions` on every step-to-step policy switch (`RL -> LLM` and `LLM -> RL`). Because 1-step DPO escalations immediately root the target host on the first attempt and return control to RL on the subsequent step, each 1-step escalation produces 2 transitions (`RL -> LLM -> RL`), yielding ~`9.15` distinct escalation events × 2 = `18.3` transitions/episode.
- **Role of AlignTune DPO (`90.0%` vs `89.2%` on `Llama-3.1-8B-Instruct`)**: Once the stage-gated router constrains the prompt to candidate CybORG actions, instruction-tuned 8B models already achieve high structural baseline performance (`89.2%`). AlignTune DPO eliminates residual syntax drift, bash hallucinations (`0.0%` invalid actions), and multi-turn chatter, pushing $P(\text{Root}\mid\text{Comp})$ to **90.0%** (`190/211`) and $P(\text{Impact}\mid\text{Root})$ to **98.9%** (`188/190`).