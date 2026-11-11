"""
Formalizes the guardrails and state triggers for when control should pass from RL to LLM,
and when control must be returned back to RL.
"""

from typing import Set, Dict


class RoutingRules:
    """State transition criteria and containment guardrails."""

    # Maximum consecutive escalation attempts per host before forced fallback
    # Set to 5 (accommodates 2-tick action duration and aligns with agent config)
    MAX_ESCALATION_STEPS_PER_HOST: int = 5

    @staticmethod
    def should_escalate(
        target_host: str,
        compromised_hosts: Set[str],
        rooted_hosts: Set[str],
        attempt_tracker: Dict[str, int],
    ) -> bool:
        """
        Determines whether the router should delegate the turn to the LLM.
        Conditions:
          1. Host has an unprivileged foothold (is compromised).
          2. Host does NOT currently have root privileges.
          3. Host has not exceeded the maximum escalation attempt budget.
        """
        is_compromised = target_host in compromised_hosts
        is_rooted = target_host in rooted_hosts
        attempts = attempt_tracker.get(target_host, 0)

        return (
            is_compromised
            and (not is_rooted)
            and (attempts < RoutingRules.MAX_ESCALATION_STEPS_PER_HOST)
        )
