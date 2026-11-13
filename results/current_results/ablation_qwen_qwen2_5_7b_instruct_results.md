# Empirical 4-Quadrant Kill-Chain Ablation Results (120 Steps)

| Configuration                                   |   Episodes | P(Comp|Disc)   | P(Root|Comp)   | P(Impact|Root)   | Impact Rate   | Avg RL Steps   | Avg LLM Steps   | Avg Tokens   | Avg Time (s)   |
|-------------------------------------------------|------------|----------------|----------------|------------------|---------------|----------------|-----------------|--------------|----------------|
| 1. Baseline Pure RL                             |         30 | 38.8±5.7%      | 61.4±23.1%     | 60.7±29.7%       | 90.0%         | 120.0±0.0      | 0.0±0.0         | 0±0          | 4.22±0.56s     |
| 2. Baseline Pure LLM (Qwen2.5-7B-Instruct)      |         30 | 51.2±18.6%     | 0.0±0.0%       | 0.0±0.0%         | 0.0%          | 0.0±0.0        | 120.0±0.0       | 40832±5319   | 141.31±8.12s   |
| 3. Hybrid + Frozen Base (Qwen2.5-7B-Instruct)   |         30 | 37.6±5.7%      | 59.9±19.4%     | 78.3±23.7%       | 100.0%        | 99.0±6.9       | 21.0±6.9        | 4207±1387    | 22.28±5.83s    |
| 4. Hybrid + AlignTune DPO (Qwen2.5-7B-Instruct) |         30 | 38.8±6.3%      | 48.5±16.0%     | 73.7±22.7%       | 100.0%        | 95.9±6.2       | 24.1±6.2        | 4827±1234    | 24.81±5.36s    |

### Statistical Significance Matrix (Two-Proportion Z-Test)

- **Hybrid DPO vs Pure RL [P(Root|Comp)]**: z = -2.70, p = 0.0069 (p < 0.05, statistically significant)
- **Hybrid DPO vs Hybrid Frozen [P(Root|Comp)]**: z = -2.26, p = 0.0239 (p < 0.05, statistically significant)
