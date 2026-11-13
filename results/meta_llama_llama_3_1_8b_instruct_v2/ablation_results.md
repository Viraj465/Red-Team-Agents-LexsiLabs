# Empirical 4-Quadrant Kill-Chain Ablation Results (120 Steps)

| Configuration                                     |   Episodes | P(Comp|Disc)   | P(Root|Comp)   | P(Impact|Root)   | Impact Rate   | Avg RL Steps   | Avg LLM Steps   | Avg Tokens   | Avg Time (s)   |
|---------------------------------------------------|------------|----------------|----------------|------------------|---------------|----------------|-----------------|--------------|----------------|
| 1. Baseline Pure RL                               |         30 | 39.0±5.3%      | 65.3±22.9%     | 61.5±23.5%       | 100.0%        | 120.0±0.0      | 0.0±0.0         | 0±0          | 3.35±0.47s     |
| 2. Baseline Pure LLM (Llama-3.1-8B-Instruct)      |         30 | 24.9±9.9%      | 49.4±26.9%     | 0.0±0.0%         | 0.0%          | 0.0±0.0        | 120.0±0.0       | 72317±13342  | 120.38±3.71s   |
| 3. Hybrid + Frozen Base (Llama-3.1-8B-Instruct)   |         30 | 36.8±5.5%      | 89.2±14.7%     | 98.4±4.5%        | 100.0%        | 110.3±4.6      | 9.7±4.6         | 2180±1049    | 11.89±4.20s    |
| 4. Hybrid + AlignTune DPO (Llama-3.1-8B-Instruct) |         30 | 36.1±5.7%      | 90.0±14.0%     | 98.9±3.3%        | 100.0%        | 110.3±4.6      | 9.7±4.6         | 2172±1061    | 11.89±4.32s    |

### Statistical Significance Matrix (Two-Proportion Z-Test)

- **Hybrid Frozen vs Pure RL [P(Root|Comp)]**: 189/212 (89.2%) vs 156/239 (65.3%), z = 5.97, p < 0.0001 (statistically significant)
- **Hybrid DPO vs Pure RL [P(Root|Comp)]**: 190/211 (90.0%) vs 156/239 (65.3%), z = 6.22, p < 0.0001 (statistically significant)
- **Hybrid DPO vs Hybrid Frozen [P(Root|Comp)]**: 190/211 (90.0%) vs 189/212 (89.2%), z = 0.30, p = 0.7627 (p >= 0.05)
- **Hybrid DPO vs Pure RL [P(Impact|Root)]**: 188/190 (98.9%) vs 96/156 (61.5%), z = 9.03, p < 0.0001 (statistically significant)
- **Hybrid DPO vs Hybrid Frozen [P(Impact|Root)]**: 188/190 (98.9%) vs 186/189 (98.4%), z = 0.46, p = 0.6483 (p >= 0.05)
- **Hybrid DPO vs Pure RL [End-to-End Yield P(Impact|Disc)]**: 188/585 (32.1%) vs 96/613 (15.7%), z = 6.70, p < 0.0001 (statistically significant)

### Episode-Clustered 95% Bootstrap CIs & Telemetry Notes

| Configuration | P(Comp\|Disc) 95% CI | P(Root\|Comp) 95% CI | P(Impact\|Root) 95% CI | End-to-End Yield P(Impact\|Disc) | Avg Impacted Hosts/Ep | Avg Handoff Transitions |
|---|---|---|---|---|---|---|
| 1. Baseline Pure RL | [37.1%, 40.8%] | [57.7%, 72.9%] | [52.7%, 70.7%] | 15.7% (96/613) | 3.20 | 0.0 |
| 2. Baseline Pure LLM | [21.9%, 28.3%] | [39.5%, 60.0%] | [0.0%, 0.0%] | 0.0% (0/349) | 0.00 | 0.0 |
| 3. Hybrid + Frozen Base | [35.0%, 38.6%] | [83.8%, 94.2%] | [96.4%, 100.0%] | 32.3% (186/576) | 6.20 | 18.33 |
| 4. Hybrid + AlignTune DPO | [34.2%, 38.0%] | [85.1%, 95.1%] | [97.4%, 100.0%] | 32.1% (188/585) | 6.27 | 18.30 |

- **Handoff Transition Accounting**: `avg_handoff_transitions = 18.3` for `9.7` LLM steps counts both directions (`RL -> LLM` and `LLM -> RL`). Because 1-step DPO escalations immediately root the host and return control to RL on the next step, each escalation produces ~2 transitions (~9.15 distinct host escalations × 2 = 18.3).
