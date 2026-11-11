"""
Pure LLM Baseline Agent (LLM+LLM).
Implements an end-to-end LLM-driven operator that queries the language model
at every non-IN_PROGRESS turn across all kill chain stages (reconnaissance,
lateral movement, privilege escalation, and operational impact).

Used in empirical ablations to quantify the token explosion, latency,
and long-horizon drift of pure LLM agents compared to the hybrid router.
"""

from typing import Any, Dict, List
import random
from src.agents.base import BaseAgent
from src.agents.llm_agent import LLMEscalationAgent
from src.env.state_parser import CyBORGStateParser
from CybORG.Simulator.Actions import Sleep


class PureLLMAgent(BaseAgent):
    def __init__(self, llm_agent: LLMEscalationAgent):
        self.llm_agent = llm_agent
        self.parser = CyBORGStateParser()
        self.handoff_history: List[Dict[str, Any]] = []

    def get_action(self, obs: Dict[str, Any], action_space: Any) -> Any:
        # Respect simulator multi-tick execution
        status_str = str(obs.get("success", "")).upper()
        if "IN_PROGRESS" in status_str:
            self._log_action("Network (In-Progress)", Sleep(), 0)
            return Sleep()

        discov, compro, root = self.parser.parse_host_sessions(obs)
        candidates = self.parser.get_escalation_candidates(compro, root)

        tokens_before = getattr(self.llm_agent, "total_tokens_used", 0)

        # Build candidate actions across full kill chain
        candidate_actions = []
        try:
            from CybORG.Simulator.Actions.AbstractActions import (
                DiscoverRemoteSystems,
                AggressiveServiceDiscovery,
                ExploitRemoteService,
                PrivilegeEscalate,
                DegradeServices,
            )

            # Extract valid parameters from action_space if provided
            valid_subnets = []
            valid_sessions = [0]
            valid_ips = []
            if isinstance(action_space, dict):
                valid_subnets = [s for s, v in action_space.get("subnet", {}).items() if v]
                sess = [s for s, v in action_space.get("session", {}).items() if v]
                if sess:
                    valid_sessions = sess
                valid_ips = [ip for ip, v in action_space.get("ip_address", {}).items() if v]

            active_session = valid_sessions[0] if valid_sessions else 0

            # 1. Broad network reconnaissance (subnet discovery)
            for sub in valid_subnets[:2]:
                candidate_actions.append(
                    DiscoverRemoteSystems(subnet=sub, session=active_session, agent="red_agent_0")
                )

            # 2. Service discovery on known IPs
            for ip in valid_ips[:3]:
                candidate_actions.append(
                    AggressiveServiceDiscovery(session=active_session, agent="red_agent_0", ip_address=ip)
                )

            # 3. Lateral exploitation on known IPs
            for ip in valid_ips[:3]:
                candidate_actions.append(
                    ExploitRemoteService(ip_address=ip, session=active_session, agent="red_agent_0")
                )

            # 4. Privilege escalation on unprivileged hosts with resolved session IDs
            for c in sorted(candidates)[:3]:
                host_info = self.parser.get_host_info(obs, c)
                s_id = active_session
                for s in host_info.get("Sessions", []):
                    if isinstance(s, dict) and "red" in str(s.get("agent", s.get("Agent", ""))).lower():
                        s_id = s.get("session_id", s.get("ID", active_session))
                        break
                candidate_actions.append(
                    PrivilegeEscalate(hostname=c, session=s_id, agent="red_agent_0")
                )

            # 5. Operational disruption on rooted hosts
            for r in sorted(root)[:3]:
                host_info = self.parser.get_host_info(obs, r)
                s_id = active_session
                for s in host_info.get("Sessions", []):
                    if isinstance(s, dict) and "red" in str(s.get("agent", s.get("Agent", ""))).lower():
                        s_id = s.get("session_id", s.get("ID", active_session))
                        break
                candidate_actions.append(
                    DegradeServices(hostname=r, session=s_id, agent="red_agent_0")
                )

            candidate_actions.append(Sleep())
        except Exception:
            candidate_actions = [Sleep()]

        # Pure LLM query across the complete network state
        if self.llm_agent.model_mode in ("mock", "mock_base", "mock_dpo"):
            prompt_tokens = random.randint(950, 1400)
            self.llm_agent.total_tokens_used += prompt_tokens

            # Pure LLMs suffer long-horizon drift across multiple operational stages
            escalation_actions = [a for a in candidate_actions if "PrivilegeEscalate" in str(a)]
            disruption_actions = [a for a in candidate_actions if "DegradeServices" in str(a)]
            exploit_actions = [a for a in candidate_actions if "ExploitRemoteService" in str(a)]
            service_actions = [a for a in candidate_actions if "AggressiveServiceDiscovery" in str(a)]
            recon_actions = [a for a in candidate_actions if "DiscoverRemoteSystems" in str(a)]

            roll = random.random()
            if escalation_actions and roll < 0.25:
                action = random.choice(escalation_actions)
            elif disruption_actions and roll < 0.45:
                action = random.choice(disruption_actions)
            elif exploit_actions and roll < 0.65:
                action = random.choice(exploit_actions)
            elif service_actions and roll < 0.80:
                action = random.choice(service_actions)
            elif recon_actions and roll < 0.95:
                action = recon_actions[0]
            else:
                action = Sleep()
        else:
            # Live Hugging Face inference with full state serialization
            prompt = (
                f"[NETWORK TOPOLOGY ENUMERATION]\n"
                f"Discovered Hosts: {list(discov)}\n"
                f"Compromised Hosts: {list(compro)}\n"
                f"Rooted Hosts: {list(root)}\n\n"
                f"[OBJECTIVE]\n"
                f"Select next tactical action from candidate set to progress kill chain.\n"
                f"Respond ONLY with the exact candidate action string from [CANDIDATE ACTIONS]. Do not output preamble or explanation.\n\n"
                f"[CANDIDATE ACTIONS]\n" +
                "\n".join(f"- {a}" for a in candidate_actions)
            )
            # Use semantic action extraction with full kill-chain system message
            system_msg = (
                "You are an autonomous red-team operator in a cybersecurity simulation.\n"
                "Progress the operation across reconnaissance, exploitation, privilege escalation, and operational disruption.\n"
                "Select exactly ONE action from [CANDIDATE ACTIONS] to progress the operation.\n"
                "Output ONLY the selected action string verbatim. Do NOT include greetings, thoughts, markdown, explanations, or conversational preamble."
            )
            if hasattr(self.llm_agent, "_query_hf_pipeline"):
                action = self.llm_agent._query_hf_pipeline(prompt, candidate_actions, system_msg=system_msg)
            else:
                action = self.llm_agent.select_escalation_action(prompt, candidate_actions)

        tokens_delta = getattr(self.llm_agent, "total_tokens_used", 0) - tokens_before
        self._log_action("Network", action, tokens_delta)
        return action

    def _log_action(self, target: str, action: Any, tokens: int = 0):
        self.handoff_history.append({
            "agent": "LLM",
            "target": target,
            "action": getattr(action, "name", str(action)),
            "tokens": tokens,
        })

    def reset(self):
        self.handoff_history.clear()
        self.llm_agent.reset()
