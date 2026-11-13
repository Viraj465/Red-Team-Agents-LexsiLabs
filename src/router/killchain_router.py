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
from src.env.state_parser import CyBORGStateParser, get_host_obs
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
        self.compromised_hosts: Set[str] = set()
        self.rooted_hosts: Set[str] = set()
        self.host_info_cache: Dict[str, Dict[str, Any]] = {}
        self.active_escalation_target: str = None
        self.handoff_history: List[Dict[str, Any]] = []

    def _resolve_active_session(self, action_space: Any) -> int:
        """
        Resolves the valid RedAbstractSession ID from action_space (always 0 in CAGE-4).
        In CybORG CAGE-4, PrivilegeEscalate and DegradeServices require the source
        RedAbstractSession (session=0), NOT the target host's local session ID.
        """
        if isinstance(action_space, dict):
            valid_sessions = [s for s, v in action_space.get("session", {}).items() if v]
            if valid_sessions:
                return valid_sessions[0]
        return 0

    def _sync_rl_state(self, obs: Dict[str, Any], action_space: Any, compro: Set[str], root: Set[str]):
        """
        Keeps FiniteStateRedAgent synchronized even when the router intercepts turns
        for LLM privilege escalation or post-root service disruption.
        """
        fsm = getattr(self.rl_agent, "internal_agent", None)
        if fsm is None:
            return

        if not self.rl_agent.initialized and action_space is not None:
            try:
                if hasattr(fsm, "set_initial_values"):
                    fsm.set_initial_values(action_space, obs)
                if hasattr(fsm, "_process_new_observations"):
                    clean_obs = {k: v for k, v in obs.items() if k not in ("success", "action") and isinstance(v, dict)}
                    fsm._process_new_observations(clean_obs)
                self.rl_agent.initialized = True
            except Exception:
                pass
        elif hasattr(fsm, "_process_new_observations"):
            try:
                clean_obs = {k: v for k, v in obs.items() if k not in ("success", "action") and isinstance(v, dict)}
                if clean_obs:
                    fsm._process_new_observations(clean_obs)
            except Exception:
                pass

        if hasattr(fsm, "host_states"):
            for ip_key, st in fsm.host_states.items():
                hname = st.get("hostname", "")
                if hname in root:
                    if st.get("state") not in ("R", "RD"):
                        st["state"] = "R"
                elif hname in compro:
                    if st.get("state") in ("K", "KD", "S", "SD"):
                        st["state"] = "U"

    def get_action(self, obs: Dict[str, Any], action_space: Any) -> Any:
        # 0. If CybORG is mid-execution on a multi-tick action, yield Sleep directly
        # Do NOT query RL agent to prevent advancing or consuming its internal action plan
        status_str = str(obs.get("success", "")).upper()
        if "IN_PROGRESS" in status_str:
            self._log_handoff("RL", "Network (In-Progress Wait)", Sleep(), 0, reason="in_progress_wait")
            return Sleep()

        active_session = self._resolve_active_session(action_space)

        # 1. State Synchronization: Parse sessions from observation and update cumulative state
        discov, compro, root = self.parser.parse_host_sessions(obs)

        # Detect if Blue team restored/removed sessions on any host explicitly reported in obs
        if isinstance(obs, dict):
            for host_key, host_val in obs.items():
                if host_key in ("success", "action") or not isinstance(host_val, dict):
                    continue
                hname = host_val.get("System info", {}).get("Hostname", host_key)
                if "." in str(hname) and str(hname) in self.parser.ip_to_hostname:
                    hname = self.parser.ip_to_hostname[str(hname)]
                if "Sessions" in host_val and hname not in compro:
                    self.compromised_hosts.discard(hname)
                    self.rooted_hosts.discard(hname)

        self.compromised_hosts.update(compro)
        self.rooted_hosts.update(root)

        for h in discov | compro | root:
            h_obs = get_host_obs(obs, h) or self.parser.get_host_info(obs, h)
            if h_obs:
                self.host_info_cache[h] = h_obs

        candidates = self.parser.get_escalation_candidates(self.compromised_hosts, self.rooted_hosts)

        # Synchronize verified compromised/rooted hosts with RL internal host_states
        self._sync_rl_state(obs, action_space, self.compromised_hosts, self.rooted_hosts)

        # 2. Check if any unprivileged host needs LLM escalation
        target_to_escalate = None
        for candidate in sorted(candidates):
            if self.rules.should_escalate(candidate, self.compromised_hosts, self.rooted_hosts, self.host_attempts):
                target_to_escalate = candidate
                break

        # 3. ROUTE TO LLM: Host is trapped at an unprivileged shell (UID != 0)
        if target_to_escalate is not None:
            self.active_escalation_target = target_to_escalate
            self.host_attempts[target_to_escalate] = (
                self.host_attempts.get(target_to_escalate, 0) + 1
            )
            # Use safe accessor get_host_obs with fallback to cached host info
            host_info = (
                get_host_obs(obs, target_to_escalate)
                or self.parser.get_host_info(obs, target_to_escalate)
                or self.host_info_cache.get(target_to_escalate, {})
            )

            try:
                from CybORG.Simulator.Actions.AbstractActions import (
                    PrivilegeEscalate,
                    DegradeServices,
                )

                candidate_actions = [
                    PrivilegeEscalate(
                        hostname=target_to_escalate, session=active_session, agent="red_agent_0"
                    ),
                    DegradeServices(
                        hostname=target_to_escalate, session=active_session, agent="red_agent_0"
                    ),
                    Sleep(),
                ]
            except Exception:
                candidate_actions = [Sleep()]

            prompt = self.parser.format_llm_privesc_prompt(
                target_to_escalate, host_info, candidate_actions
            )
            tokens_before = getattr(self.llm_agent, "total_tokens_used", 0)
            comp_tokens_before = getattr(self.llm_agent, "completion_tokens_used", 0)
            action = self.llm_agent.select_escalation_action(prompt, candidate_actions)
            tokens_delta = getattr(self.llm_agent, "total_tokens_used", 0) - tokens_before
            comp_tokens_delta = getattr(self.llm_agent, "completion_tokens_used", 0) - comp_tokens_before

            self._log_handoff(
                "LLM",
                target_to_escalate,
                action,
                tokens_delta,
                reason="unprivileged_foothold_escalation",
                completion_tokens=comp_tokens_delta,
            )
            return action

        # 4. POST-ROOT WEAPONIZATION: If newly rooted hosts exist and RL hasn't disrupted yet
        # Prioritize mission-critical enterprise servers and operational SCADA hosts
        new_rooted = [h for h in sorted(self.rooted_hosts) if h not in self.disrupted_hosts]
        if new_rooted:
            # Tier 1: Operational hosts (highest mission impact in CAGE-4: RIA=-10, LWF=-10)
            op_targets = [h for h in new_rooted if "operational" in h]
            # Tier 2: Enterprise servers across restricted, admin, office, and public zones
            server_targets = [h for h in new_rooted if "server" in h and "operational" not in h]
            # Tier 3: Hosts with verified active services from observation
            service_targets = [
                h for h in new_rooted
                if h not in op_targets and h not in server_targets
                and len(
                    (
                        get_host_obs(obs, h)
                        or self.parser.get_host_info(obs, h)
                        or self.host_info_cache.get(h, {})
                    ).get("Services", [])
                ) > 0
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

            try:
                from CybORG.Simulator.Actions.AbstractActions import DegradeServices
                impact_action = DegradeServices(
                    hostname=target_disrupt, session=active_session, agent="red_agent_0"
                )
                self._log_handoff("RL", target_disrupt, impact_action, 0, reason="post_root_weaponization")
                return impact_action
            except Exception:
                pass

        # 5. ROUTE TO RL: Tactical network reconnaissance, lateral movement, and general actions
        self.active_escalation_target = None
        rl_action = self.rl_agent.get_action(obs, action_space)
        self._log_handoff("RL", "Network", rl_action, 0, reason="tactical_rl_exploration")
        return rl_action

    def _log_handoff(
        self,
        agent_name: str,
        target: str,
        action: Any,
        tokens: int = 0,
        reason: str = "general",
        completion_tokens: int = 0,
    ):
        self.handoff_history.append(
            {
                "agent": agent_name,
                "target": target,
                "action": getattr(action, "name", str(action)),
                "tokens": tokens,
                "completion_tokens": completion_tokens,
                "reason": reason,
            }
        )

    def get_router_telemetry(self) -> Dict[str, Any]:
        """Returns granular routing telemetry (handoff counts, reasons, and escalation efficiency)."""
        llm_turns = [h for h in self.handoff_history if h.get("agent") == "LLM"]
        rl_turns = [h for h in self.handoff_history if h.get("agent") == "RL"]
        transitions = 0
        for i in range(1, len(self.handoff_history)):
            if self.handoff_history[i].get("agent") != self.handoff_history[i - 1].get("agent"):
                transitions += 1
        escalated_hosts = set(self.host_attempts.keys())
        rooted_via_router = len(escalated_hosts & self.rooted_hosts)
        return {
            "llm_steps": len(llm_turns),
            "rl_steps": len(rl_turns),
            "agent_transitions": transitions,
            "unique_hosts_handed_to_llm": len(escalated_hosts),
            "hosts_rooted_after_handoff": rooted_via_router,
            "mean_attempts_per_escalated_host": (
                sum(self.host_attempts.values()) / len(self.host_attempts)
                if self.host_attempts
                else 0.0
            ),
        }

    def reset(self):
        self.host_attempts.clear()
        self.disrupted_hosts.clear()
        self.compromised_hosts.clear()
        self.rooted_hosts.clear()
        self.host_info_cache.clear()
        self.active_escalation_target = None
        self.handoff_history.clear()
        self.rl_agent.reset()
        self.llm_agent.reset()
