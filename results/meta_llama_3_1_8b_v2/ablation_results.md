# Empirical 4-Quadrant Kill-Chain Ablation Results (120 Steps)

| Configuration                            |   Episodes | P(Comp|Disc)   | P(Root|Comp)   | P(Impact|Root)   | Impact Rate   | Avg RL Steps   | Avg LLM Steps   | Avg Tokens   | Avg Time (s)   |
|------------------------------------------|------------|----------------|----------------|------------------|---------------|----------------|-----------------|--------------|----------------|
| 1. Baseline Pure RL                      |         30 | 37.8±6.5%      | 59.3±24.0%     | 70.1±22.0%       | 100.0%        | 120.0±0.0      | 0.0±0.0         | 0±0          | 4.31±0.51s     |
| 2. Baseline Pure LLM (Llama-3.1-8B)      |         30 | 87.1±20.3%     | 100.0±0.0%     | 56.8±29.1%       | 100.0%        | 0.0±0.0        | 120.0±0.0       | 24345±3055   | 400.72±2.19s   |
| 3. Hybrid + Frozen Base (Llama-3.1-8B)   |         30 | 38.2±5.1%      | 49.5±21.5%     | 76.3±18.7%       | 100.0%        | 96.3±6.0       | 23.7±6.0        | 6598±1666    | 159.58±39.65s  |
| 4. Hybrid + AlignTune DPO (Llama-3.1-8B) |         30 | 38.1±6.3%      | 48.7±20.4%     | 83.5±20.7%       | 100.0%        | 96.1±7.9       | 23.9±7.9        | 6644±2196    | 159.94±50.51s  |

### Statistical Significance Matrix (Two-Proportion Z-Test)

- **Hybrid DPO vs Pure RL [P(Root|Comp)]**: z = -2.18, p = 0.0294 (p < 0.05, statistically significant)
- **Hybrid DPO vs Hybrid Frozen [P(Root|Comp)]**: z = -0.15, p = 0.8821 (p >= 0.05)
