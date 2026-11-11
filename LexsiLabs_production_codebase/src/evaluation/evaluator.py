"""
Executes multi-episode benchmark sweeps with distinct random seeds per episode.
Measures stage transitions, token economics, and wall-clock execution latency.
"""

import time
from typing import Any
from src.env.cyborg_wrapper import StageGatedCybORGEnv
from src.evaluation.metrics import KillChainMetricsCollector


class RedTeamEvaluator:
    def __init__(self, num_episodes: int = 20, max_steps: int = 80):
        self.num_episodes = num_episodes
        self.max_steps = max_steps

    def evaluate_agent(self, agent: Any, config_name: str) -> KillChainMetricsCollector:
        collector = KillChainMetricsCollector(config_name)

        for ep in range(1, self.num_episodes + 1):
            env = StageGatedCybORGEnv(seed=100 + ep, steps=self.max_steps)
            obs, action_space = env.reset()
            agent.reset()

            start_time = time.time()
            for step in range(1, self.max_steps + 1):
                action = agent.get_action(obs, action_space)
                obs, reward, done, info = env.step(action)
                action_space = getattr(info.get("raw_results"), "action_space", action_space)
                if done:
                    break

            duration_seconds = time.time() - start_time
            metrics = env.get_stage_metrics()
            handoffs = getattr(agent, "handoff_history", [])
            collector.record_episode(metrics, handoffs, duration_seconds=duration_seconds)

        return collector
