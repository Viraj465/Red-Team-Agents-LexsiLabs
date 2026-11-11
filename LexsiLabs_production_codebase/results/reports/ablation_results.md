# Empirical 4-Quadrant Kill-Chain Ablation Results (10 Steps)

| Configuration                                         |   Episodes | P(Comp|Disc)   | P(Root|Comp)   | P(Impact|Root)   | Impact Rate   | Avg RL Steps   | Avg LLM Steps   | Avg Tokens   | Avg Time (s)   |
|-------------------------------------------------------|------------|----------------|----------------|------------------|---------------|----------------|-----------------|--------------|----------------|
| 1. Baseline Pure RL                                   |          1 | 7.7±0.0%       | 100.0±0.0%     | 0.0±0.0%         | 0.0%          | 10.0±0.0       | 0.0±0.0         | 0±0          | 0.24±0.00s     |
| 2. Baseline Pure LLM (Qwen2.5-Coder-7B-Instruct)      |          1 | 7.7±0.0%       | 100.0±0.0%     | 100.0±0.0%       | 100.0%        | 0.0±0.0        | 10.0±0.0        | 6951±0       | 0.17±0.00s     |
| 3. Hybrid + Frozen Base (Qwen2.5-Coder-7B-Instruct)   |          1 | 7.7±0.0%       | 100.0±0.0%     | 100.0±0.0%       | 100.0%        | 9.0±0.0        | 1.0±0.0         | 185±0        | 0.24±0.00s     |
| 4. Hybrid + AlignTune DPO (Qwen2.5-Coder-7B-Instruct) |          1 | 100.0±0.0%     | 100.0±0.0%     | 100.0±0.0%       | 100.0%        | 9.0±0.0        | 1.0±0.0         | 42±0         | 0.25±0.00s     |

### Statistical Significance Matrix (Two-Proportion Z-Test)

- **Hybrid DPO vs Pure RL [P(Root|Comp)]**: z = 0.00, p = 1.0000 (p >= 0.05)
- **Hybrid DPO vs Hybrid Frozen [P(Root|Comp)]**: z = 0.00, p = 1.0000 (p >= 0.05)
