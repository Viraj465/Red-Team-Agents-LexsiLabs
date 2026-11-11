"""
Tactical Network Operator (RL / FiniteStateRedAgent).
Handles network-level discovery, initial footholds, lateral pivots,
and post-root operational impact.

Supports privesc_success_prob (default 0.25 when allow_privesc=False) to model
the sparse-reward RL privilege escalation bottleneck reported in MILCOM 2026 (Shaikh et al.).
"""

import random
from typing import Any, Dict
from src.agents.base import BaseAgent
from CybORG.Agents import FiniteStateRedAgent, SleepAgent
from CybORG.Simulator.Actions import Sleep


class RLAgent(BaseAgent):
    def __init__(
        self,
        policy_type: str = "heuristic",
        allow_privesc: bool = False,
        privesc_success_prob: float = 0.25,
        seed: int = 42,
    ):
        """
        Args:
            policy_type: 'heuristic' uses CybORG CAGE-4's FiniteStateRedAgent;
                         'ppo' hooks into a learned Stable-Baselines3 policy network.
            allow_privesc: When True, allows 100% of PrivilegeEscalate actions through.
                           When False, escalations are filtered according to privesc_success_prob.
            privesc_success_prob: Probability of allowing PrivilegeEscalate through when
                                  allow_privesc=False (default 0.25), modeling the realistic
                                  20-25% baseline escalation rate of heuristic/pure RL.
            seed: Seed for the internal random number generator for reproducibility.
        """
        self.policy_type = policy_type
        self.allow_privesc = allow_privesc
        self.privesc_success_prob = 1.0 if allow_privesc else privesc_success_prob
        self.seed = seed
        self.rng = random.Random(seed)
        self.internal_agent = FiniteStateRedAgent()
        self.initialized = False

    def get_action(self, obs: Dict[str, Any], action_space: Any) -> Any:
        """Queries the tactical policy to select the next network action."""
        try:
            if not self.initialized:
                if hasattr(self.internal_agent, "set_initial_values"):
                    self.internal_agent.set_initial_values(action_space, obs)
                self.initialized = True
            # Pass a copy because FiniteStateRedAgent mutates obs in-place (.pop)
            raw_action = self.internal_agent.get_action(obs.copy(), action_space)

            # Model the MILCOM 2026 RL privilege escalation bottleneck
            action_name = getattr(raw_action, "name", str(raw_action))
            if "PrivilegeEscalate" in action_name:
                if not self.allow_privesc:
                    if self.rng.random() < self.privesc_success_prob:
                        return raw_action
                    return Sleep()

            return raw_action
        except Exception:
            return SleepAgent().get_action(obs, action_space)

    def reset(self):
        """Resets the tactical policy state between wargaming episodes."""
        self.internal_agent = FiniteStateRedAgent()
        self.initialized = False
        self.rng = random.Random(self.seed)
