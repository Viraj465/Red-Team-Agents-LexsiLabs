# Empirical 4-Quadrant Kill-Chain Ablation Results (120 Steps)

| Configuration                            |   Episodes | P(Comp|Disc)   | P(Root|Comp)   | P(Impact|Root)   | Impact Rate   | Avg RL Steps   | Avg LLM Steps   | Avg Tokens   | Avg Time (s)   |
|------------------------------------------|------------|----------------|----------------|------------------|---------------|----------------|-----------------|--------------|----------------|
| 1. Baseline Pure RL                      |         30 | 37.4±6.5%      | 61.7±22.2%     | 63.5±26.7%       | 93.3%         | 120.0±0.0      | 0.0±0.0         | 0±0          | 4.25±0.53s     |
| 2. Baseline Pure LLM (Llama-3.1-8B)      |         30 | 80.0±24.4%     | 82.5±17.6%     | 0.0±0.0%         | 0.0%          | 0.0±0.0        | 120.0±0.0       | 12099±1420   | 200.12±17.20s  |
| 3. Hybrid + Frozen Base (Llama-3.1-8B)   |         30 | 37.6±6.1%      | 50.0±19.7%     | 83.0±23.7%       | 100.0%        | 98.1±7.5       | 21.9±7.5        | 6088±2098    | 135.58±45.41s  |
| 4. Hybrid + AlignTune DPO (Llama-3.1-8B) |         30 | 38.3±5.7%      | 55.8±20.5%     | 84.0±20.6%       | 100.0%        | 99.0±6.1       | 21.0±6.1        | 5838±1683    | 129.78±36.27s  |

### Statistical Significance Matrix (Two-Proportion Z-Test)

- **Hybrid DPO vs Pure RL [P(Root|Comp)]**: z = -1.23, p = 0.2184 (p >= 0.05)
- **Hybrid DPO vs Hybrid Frozen [P(Root|Comp)]**: z = 1.13, p = 0.2595 (p >= 0.05)
