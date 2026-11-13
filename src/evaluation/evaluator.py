"""
Executes multi-episode benchmark sweeps with distinct random seeds per episode.
Measures stage transitions, token economics, and wall-clock execution latency.
"""

import time
from typing import Any
from tqdm.auto import tqdm
from src.env.cyborg_wrapper import StageGatedCybORGEnv
from src.evaluation.metrics import KillChainMetricsCollector


class RedTeamEvaluator:
    def __init__(
        self,
        num_episodes: int = 20,
        max_steps: int = 80,
        show_progress: bool = True,
    ):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.show_progress = show_progress

    def evaluate_agent(self, agent: Any, config_name: str) -> KillChainMetricsCollector:
        collector = KillChainMetricsCollector(config_name)

        ep_range = range(1, self.num_episodes + 1)
        if self.show_progress:
            ep_iter = tqdm(
                ep_range,
                desc=f"[{config_name}]",
                unit="ep",
                dynamic_ncols=True,
                leave=True,
            )
        else:
            ep_iter = ep_range

        for ep in ep_iter:
            env = StageGatedCybORGEnv(seed=100 + ep, steps=self.max_steps)
            obs, action_space = env.reset()
            agent.reset()

            start_time = time.time()
            step_range = range(1, self.max_steps + 1)
            if self.show_progress:
                step_iter = tqdm(
                    step_range,
                    desc=f"  └─ Ep {ep}/{self.num_episodes}",
                    unit="step",
                    leave=False,
                    dynamic_ncols=True,
                )
            else:
                step_iter = step_range

            for step in step_iter:
                action = agent.get_action(obs, action_space)
                obs, reward, done, info = env.step(action)
                action_space = getattr(info.get("raw_results"), "action_space", action_space)
                if done:
                    break

            duration_seconds = time.time() - start_time
            metrics = env.get_stage_metrics()
            handoffs = getattr(agent, "handoff_history", [])
            collector.record_episode(metrics, handoffs, duration_seconds=duration_seconds)

            if self.show_progress and hasattr(ep_iter, "set_postfix"):
                compro = metrics.get("compromised_hosts", 0)
                root = metrics.get("root_hosts", 0)
                impact = "✓" if metrics.get("impact_achieved") else "✗"
                ep_iter.set_postfix({
                    "time": f"{duration_seconds:.1f}s",
                    "root/comp": f"{root}/{compro}",
                    "impact": impact,
                })

        return collector
