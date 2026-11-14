# Empirical 4-Quadrant Kill-Chain Ablation Results (120 Steps)

| Configuration                              |   Episodes | P(Comp|Disc)   | P(Root|Comp)   | P(Impact|Root)   | Impact Rate   | Avg RL Steps   | Avg LLM Steps   | Avg Tokens   | Avg Time (s)   |
|--------------------------------------------|------------|----------------|----------------|------------------|---------------|----------------|-----------------|--------------|----------------|
| 1. Baseline Pure RL                        |         30 | 39.8±5.0%      | 63.2±19.1%     | 62.9±21.0%       | 100.0%        | 120.0±0.0      | 0.0±0.0         | 0±0          | 3.34±0.47s     |
| 2. Baseline Pure LLM (aya-expanse-8b)      |         30 | 92.7±8.8%      | 1.3±3.6%       | 0.0±0.0%         | 0.0%          | 0.0±0.0        | 120.0±0.0       | 13106±10767  | 43.92±18.53s   |
| 3. Hybrid + Frozen Base (aya-expanse-8b)   |         30 | 34.1±7.2%      | 91.2±11.3%     | 98.9±4.3%        | 100.0%        | 111.4±2.9      | 8.6±2.9         | 1988±696     | 12.01±3.16s    |
| 4. Hybrid + AlignTune DPO (aya-expanse-8b) |         30 | 34.1±7.2%      | 91.2±11.3%     | 98.9±4.3%        | 100.0%        | 111.4±2.9      | 8.6±2.9         | 1988±696     | 12.13±3.18s    |

### Statistical Significance Matrix (Two-Proportion Z-Test)

- **Hybrid Frozen vs Pure RL [P(Root|Comp)]**: 186/204 (91.2%) vs 151/239 (63.2%), z = 6.88, p < 0.0001 (p < 0.05, statistically significant)
- **Hybrid DPO vs Pure RL [P(Root|Comp)]**: 186/204 (91.2%) vs 151/239 (63.2%), z = 6.88, p < 0.0001 (p < 0.05, statistically significant)
- **Hybrid DPO vs Hybrid Frozen [P(Root|Comp)]**: 186/204 (91.2%) vs 186/204 (91.2%), z = 0.00, p = 1.0000 (p >= 0.05)
- **Hybrid DPO vs Pure RL [P(Impact|Root)]**: 184/186 (98.9%) vs 95/151 (62.9%), z = 8.71, p < 0.0001 (p < 0.05, statistically significant)
- **Hybrid DPO vs Hybrid Frozen [P(Impact|Root)]**: 184/186 (98.9%) vs 184/186 (98.9%), z = 0.00, p = 1.0000 (p >= 0.05)
- **Hybrid DPO vs Pure RL [End-to-End Yield P(Impact|Disc)]**: 184/598 (30.8%) vs 95/600 (15.8%), z = 6.12, p < 0.0001 (p < 0.05, statistically significant)
