"""
Kill-Chain Metrics Aggregator.
Tracks stage-gated transition probabilities, host-level disruptions,
token economics, and execution latency with empirical mean ± standard deviation,
95% bootstrap confidence intervals, and two-proportion z-tests.
"""

import math
from typing import List, Dict, Any, Tuple
import numpy as np


def compute_bootstrap_ci(
    values: List[float], n_bootstraps: int = 1000, ci: float = 95.0, seed: int = 42
) -> Tuple[float, float]:
    """Computes non-parametric bootstrap confidence interval for a metric."""
    if not values:
        return 0.0, 0.0
    arr = np.array(values)
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstraps)
    n = len(arr)
    for i in range(n_bootstraps):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    lower_p = (100.0 - ci) / 2.0
    upper_p = 100.0 - lower_p
    return float(np.percentile(boot_means, lower_p)), float(np.percentile(boot_means, upper_p))


def two_proportion_z_test(count1: int, nobs1: int, count2: int, nobs2: int) -> Tuple[float, float]:
    """
    Computes two-proportion z-test score and two-tailed p-value.
    H0: p1 == p2 vs H1: p1 != p2
    """
    if nobs1 <= 0 or nobs2 <= 0:
        return 0.0, 1.0

    p1 = count1 / nobs1
    p2 = count2 / nobs2
    p_pool = (count1 + count2) / (nobs1 + nobs2)

    denom = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / nobs1 + 1.0 / nobs2))
    if denom == 0:
        return 0.0, 1.0

    z_score = (p1 - p2) / denom
    # Two-tailed p-value via normal distribution CDF approximation
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))
    return float(z_score), float(p_value)


class KillChainMetricsCollector:
    """Aggregates and formats stage-gated red team evaluation metrics."""

    def __init__(self, config_name: str):
        self.config_name = config_name
        self.episodes_data: List[Dict[str, Any]] = []

    def record_episode(
        self,
        stage_metrics: Dict[str, Any],
        handoff_history: List[Dict[str, Any]],
        duration_seconds: float = 0.0,
    ):
        if not handoff_history and stage_metrics.get("step", 0) > 0:
            rl_steps = stage_metrics.get("step", 0)
            llm_steps = 0
            tokens = 0
        else:
            rl_steps = sum(1 for h in handoff_history if h.get("agent") == "RL")
            llm_steps = sum(1 for h in handoff_history if h.get("agent") == "LLM")
            tokens = sum(h.get("tokens", 0) for h in handoff_history)

        self.episodes_data.append({
            "discovered": stage_metrics.get("discovered", 0),
            "compromised": stage_metrics.get("compromised", 0),
            "rooted": stage_metrics.get("rooted", 0),
            "impacted_hosts": stage_metrics.get("impacted_hosts", 0),
            "impact": 1 if stage_metrics.get("impact", False) else 0,
            "total_steps": stage_metrics.get("step", 0),
            "rl_steps": rl_steps,
            "llm_steps": llm_steps,
            "tokens": tokens,
            "duration_seconds": duration_seconds,
        })

    def compute_summary(self) -> Dict[str, Any]:
        n_episodes = len(self.episodes_data)
        if n_episodes == 0:
            return {"error": "No episodes recorded"}

        # Host-level cumulative sums
        total_discov = sum(e["discovered"] for e in self.episodes_data)
        total_compro = sum(e["compromised"] for e in self.episodes_data)
        total_root = sum(e["rooted"] for e in self.episodes_data)
        total_impacted = sum(e["impacted_hosts"] for e in self.episodes_data)
        total_impact_episodes = sum(e["impact"] for e in self.episodes_data)

        # Cumulative ratio metrics
        p_comp_macro = (total_compro / total_discov) if total_discov > 0 else 0.0
        p_root_macro = (total_root / total_compro) if total_compro > 0 else 0.0
        p_impact_macro = (total_impacted / total_root) if total_root > 0 else 0.0
        episode_impact_rate = total_impact_episodes / n_episodes

        # Per-episode distributions for standard deviation and bootstrap CIs
        p_comp_per_ep = [
            (e["compromised"] / e["discovered"]) if e["discovered"] > 0 else 0.0
            for e in self.episodes_data
        ]
        p_root_per_ep = [
            (e["rooted"] / e["compromised"]) if e["compromised"] > 0 else 0.0
            for e in self.episodes_data
        ]
        p_impact_per_ep = [
            (e["impacted_hosts"] / e["rooted"]) if e["rooted"] > 0 else 0.0
            for e in self.episodes_data
        ]

        # Means and Standard Deviations
        p_comp_std = np.std(p_comp_per_ep)
        p_root_std = np.std(p_root_per_ep)
        p_impact_std = np.std(p_impact_per_ep)

        # 95% Bootstrap Confidence Intervals
        comp_ci_low, comp_ci_high = compute_bootstrap_ci(p_comp_per_ep)
        root_ci_low, root_ci_high = compute_bootstrap_ci(p_root_per_ep)
        impact_ci_low, impact_ci_high = compute_bootstrap_ci(p_impact_per_ep)

        avg_rl_steps = np.mean([e["rl_steps"] for e in self.episodes_data])
        std_rl_steps = np.std([e["rl_steps"] for e in self.episodes_data])

        avg_llm_steps = np.mean([e["llm_steps"] for e in self.episodes_data])
        std_llm_steps = np.std([e["llm_steps"] for e in self.episodes_data])

        avg_tokens = np.mean([e["tokens"] for e in self.episodes_data])
        std_tokens = np.std([e["tokens"] for e in self.episodes_data])

        avg_time = np.mean([e["duration_seconds"] for e in self.episodes_data])
        std_time = np.std([e["duration_seconds"] for e in self.episodes_data])

        return {
            "Configuration": self.config_name,
            "Episodes": n_episodes,
            "P(Comp|Disc)": f"{p_comp_macro * 100:.1f}±{p_comp_std * 100:.1f}%",
            "P(Root|Comp)": f"{p_root_macro * 100:.1f}±{p_root_std * 100:.1f}%",
            "P(Impact|Root)": f"{p_impact_macro * 100:.1f}±{p_impact_std * 100:.1f}%",
            "Impact Rate": f"{episode_impact_rate * 100:.1f}%",
            "Avg RL Steps": f"{avg_rl_steps:.1f}±{std_rl_steps:.1f}",
            "Avg LLM Steps": f"{avg_llm_steps:.1f}±{std_llm_steps:.1f}",
            "Avg Tokens": f"{avg_tokens:.0f}±{std_tokens:.0f}",
            "Avg Time (s)": f"{avg_time:.2f}±{std_time:.2f}s",
            # Numerical raw values for chart plotting and statistical tests
            "_raw": {
                "total_discov": total_discov,
                "total_compro": total_compro,
                "total_root": total_root,
                "total_impacted": total_impacted,
                "p_comp": p_comp_macro * 100,
                "p_comp_std": p_comp_std * 100,
                "p_comp_95ci": (comp_ci_low * 100, comp_ci_high * 100),
                "p_root": p_root_macro * 100,
                "p_root_std": p_root_std * 100,
                "p_root_95ci": (root_ci_low * 100, root_ci_high * 100),
                "p_impact": p_impact_macro * 100,
                "p_impact_std": p_impact_std * 100,
                "p_impact_95ci": (impact_ci_low * 100, impact_ci_high * 100),
                "impact_rate": episode_impact_rate * 100,
                "avg_rl_steps": float(avg_rl_steps),
                "avg_llm_steps": float(avg_llm_steps),
                "avg_tokens": float(avg_tokens),
                "std_tokens": float(std_tokens),
                "avg_time": float(avg_time),
            },
        }
