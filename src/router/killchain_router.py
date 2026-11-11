"""
AdaptiveKillChainRouter: Sits between Simulator and Agents.
Routes broad exploration and post-root impact to RL, while delegating
unprivileged host footholds (UID != 0) to the LLMEscalationAgent.

Implements multi-tick IN_PROGRESS awareness, state synchronization
to prevent RL memory amnesia, and dual-stage post-root weaponization.
"""

from typing import Any, List, Dict, Set
from src.agents.base import BaseAgent
from src.agents.rl_agent import RLAgent
from src.agents.llm_agent import LLMEscalationAgent
from src.env.state_parser import CyBORGStateParser
from src.router.rules import RoutingRules
from CybORG.Simulator.Actions import Sleep


class AdaptiveKillChainRouter(BaseAgent):
    def __init__(self, rl_agent: RLAgent, llm_agent: LLMEscalationAgent):
        self.rl_agent = rl_agent
        self.llm_agent = llm_agent
        self.parser = CyBORGStateParser()
        self.rules = RoutingRules()
        self.rules.MAX_ESCALATION_STEPS_PER_HOST = 5  # Aligned with evaluation config
        self.host_attempts: Dict[str, int] = {}
        self.disrupted_hosts: Set[str] = set()
        self.active_escalation_target: str = None
        self.handoff_history: List[Dict[str, Any]] = []

    def get_action(self, obs: Dict[str, Any], action_space: Any) -> Any:
        # 0. If CybORG is mid-execution on a multi-tick action, yield Sleep directly
        # Do NOT query RL agent to prevent advancing or consuming its internal action plan
        status_str = str(obs.get("success", "")).upper()
        if "IN_PROGRESS" in status_str:
            self._log_handoff("RL", "Network (In-Progress Wait)", Sleep(), 0)
            return Sleep()

        # 1. State Synchronization: Parse sessions from observation
        discov, compro, root = self.parser.parse_host_sessions(obs)
        candidates = self.parser.get_escalation_candidates(compro, root)

        # Synchronize verified rooted hosts with RL internal host_states
        # Only mutate when observation genuinely confirms root/SYSTEM session
        if hasattr(self.rl_agent.internal_agent, "host_states"):
            for ip_key, st in self.rl_agent.internal_agent.host_states.items():
                if st.get("hostname") in root:
                    st["state"] = "R"

        # 2. Check if any unprivileged host needs LLM escalation
        target_to_escalate = None
        for candidate in sorted(candidates):
            if self.rules.should_escalate(candidate, compro, root, self.host_attempts):
                target_to_escalate = candidate
                break

        # 3. ROUTE TO LLM: Host is trapped at an unprivileged shell (UID != 0)
        if target_to_escalate is not None:
            self.active_escalation_target = target_to_escalate
            self.host_attempts[target_to_escalate] = (
                self.host_attempts.get(target_to_escalate, 0) + 1
            )
            # Use safe accessor to handle observations keyed by IP or hostname
            host_info = self.parser.get_host_info(obs, target_to_escalate)

            sessions = host_info.get("Sessions", [])
            session_id = 0
            for s in sessions:
                if isinstance(s, dict) and "red" in str(s.get("agent", s.get("Agent", ""))).lower():
                    session_id = s.get("session_id", s.get("ID", 0))
                    break

            try:
                from CybORG.Simulator.Actions.AbstractActions import (
                    PrivilegeEscalate,
                    DegradeServices,
                )

                candidate_actions = [
                    PrivilegeEscalate(
                        hostname=target_to_escalate, session=session_id, agent="red_agent_0"
                    ),
                    DegradeServices(
                        hostname=target_to_escalate, session=session_id, agent="red_agent_0"
                    ),
                    Sleep(),
                ]
            except Exception:
                candidate_actions = [Sleep()]

            prompt = self.parser.format_llm_privesc_prompt(
                target_to_escalate, host_info, candidate_actions
            )
            tokens_before = getattr(self.llm_agent, "total_tokens_used", 0)
            action = self.llm_agent.select_escalation_action(prompt, candidate_actions)
            tokens_delta = getattr(self.llm_agent, "total_tokens_used", 0) - tokens_before

            self._log_handoff("LLM", target_to_escalate, action, tokens_delta)
            return action

        # 4. POST-ROOT WEAPONIZATION: If newly rooted hosts exist and RL hasn't disrupted yet
        # Prioritize mission-critical enterprise servers and operational SCADA hosts
        new_rooted = [h for h in sorted(root) if h not in self.disrupted_hosts]
        if new_rooted:
            # Tier 1: Operational hosts (highest mission impact in CAGE-4: RIA=-10, LWF=-10)
            op_targets = [h for h in new_rooted if "operational" in h]
            # Tier 2: Enterprise servers across restricted, admin, office, and public zones
            server_targets = [h for h in new_rooted if "server" in h and "operational" not in h]
            # Tier 3: Hosts with verified active services from observation
            service_targets = [
                h for h in new_rooted
                if h not in op_targets and h not in server_targets
                and len(self.parser.get_host_info(obs, h).get("Services", [])) > 0
            ]

            if op_targets:
                target_disrupt = op_targets[0]
            elif server_targets:
                target_disrupt = server_targets[0]
            elif service_targets:
                target_disrupt = service_targets[0]
            else:
                target_disrupt = new_rooted[0]

            self.disrupted_hosts.add(target_disrupt)
            h_info = self.parser.get_host_info(obs, target_disrupt)
            sess_id = 0
            for s in h_info.get("Sessions", []):
                if isinstance(s, dict) and "red" in str(s.get("agent", s.get("Agent", ""))).lower():
                    sess_id = s.get("session_id", s.get("ID", 0))
                    break

            try:
                from CybORG.Simulator.Actions.AbstractActions import DegradeServices
                impact_action = DegradeServices(
                    hostname=target_disrupt, session=sess_id, agent="red_agent_0"
                )
                self._log_handoff("RL", target_disrupt, impact_action, 0)
                return impact_action
            except Exception:
                pass

        # 5. ROUTE TO RL: Tactical network reconnaissance, lateral movement, and general actions
        self.active_escalation_target = None
        rl_action = self.rl_agent.get_action(obs, action_space)
        self._log_handoff("RL", "Network", rl_action, 0)
        return rl_action

    def _log_handoff(self, agent_name: str, target: str, action: Any, tokens: int = 0):
        self.handoff_history.append(
            {
                "agent": agent_name,
                "target": target,
                "action": getattr(action, "name", str(action)),
                "tokens": tokens,
            }
        )

    def reset(self):
        self.host_attempts.clear()
        self.disrupted_hosts.clear()
        self.active_escalation_target = None
        self.handoff_history.clear()
        self.rl_agent.reset()
        self.llm_agent.reset()
