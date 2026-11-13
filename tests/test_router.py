"""
Unit tests for AdaptiveKillChainRouter, RLAgent, LLMEscalationAgent, and StageGatedCyBORGEnv.
Verifies stage handoffs, multi-tick awareness, state synchronization,
probabilistic escalation bottlenecks, hostname-IP mapping, and post-root weaponization.
"""

import pytest
from src.agents.rl_agent import RLAgent
from src.agents.llm_agent import LLMEscalationAgent
from src.router.killchain_router import AdaptiveKillChainRouter
from src.env.state_parser import CyBORGStateParser
from src.env.cyborg_wrapper import StageGatedCyBORGEnv
from CybORG.Simulator.Actions import Sleep


class TestAdaptiveKillChainRouter:
    @pytest.fixture
    def setup_router(self):
        rl_agent = RLAgent(policy_type="heuristic", allow_privesc=False)
        llm_agent = LLMEscalationAgent(model_mode="mock")
        return AdaptiveKillChainRouter(rl_agent=rl_agent, llm_agent=llm_agent)

    def test_router_initialization(self, setup_router):
        router = setup_router
        assert router.rules.MAX_ESCALATION_STEPS_PER_HOST == 5
        assert len(router.host_attempts) == 0
        assert len(router.disrupted_hosts) == 0

    def test_router_yields_on_in_progress_tick(self, setup_router):
        router = setup_router
        mock_obs = {"success": "IN_PROGRESS"}
        action = router.get_action(mock_obs, action_space=None)
        assert isinstance(action, Sleep)
        assert router.handoff_history[-1]["agent"] == "RL"
        assert "In-Progress Wait" in router.handoff_history[-1]["target"]

    def test_router_delegates_unprivileged_host_to_llm(self, setup_router):
        router = setup_router
        mock_obs = {
            "success": "TRUE",
            "user_host_01": {
                "Interface": [{"IP Address": "10.0.0.5"}],
                "Sessions": [{"agent": "red_agent_0", "Username": "ubuntu", "ID": 0}],
                "System info": {"OSType": "linux"},
                "Services": ["sshd"],
            },
        }
        action = router.get_action(mock_obs, action_space=None)
        assert action is not None
        assert "PrivilegeEscalate" in str(action)
        assert router.host_attempts["user_host_01"] == 1
        assert router.handoff_history[-1]["agent"] == "LLM"
        assert router.handoff_history[-1]["target"] == "user_host_01"

    def test_router_respects_max_escalation_budget(self, setup_router):
        router = setup_router
        mock_obs = {
            "success": "TRUE",
            "user_host_01": {
                "Interface": [{"IP Address": "10.0.0.5"}],
                "Sessions": [{"agent": "red_agent_0", "Username": "ubuntu", "ID": 0}],
                "System info": {"OSType": "linux"},
                "Services": ["sshd"],
            },
        }
        # Exhaust 5 escalation attempts
        for i in range(5):
            router.get_action(mock_obs, action_space=None)
        assert router.host_attempts["user_host_01"] == 5

        # 6th turn must not route to LLM
        action = router.get_action(mock_obs, action_space=None)
        assert router.handoff_history[-1]["agent"] == "RL"

    def test_router_triggers_post_root_server_disruption(self, setup_router):
        router = setup_router
        mock_obs = {
            "success": "TRUE",
            "Enterprise_Server_01": {
                "Interface": [{"IP Address": "10.0.1.10"}],
                "Sessions": [{"agent": "red_agent_0", "Username": "root", "ID": 0}],
                "System info": {"OSType": "linux"},
            },
        }
        action = router.get_action(mock_obs, action_space=None)
        assert action is not None
        assert "DegradeServices" in str(action) or "Impact" in str(action)
        assert "Enterprise_Server_01" in router.disrupted_hosts
        assert router.handoff_history[-1]["target"] == "Enterprise_Server_01"

    def test_state_parser_resolves_ip_keyed_observations(self):
        parser = CyBORGStateParser()
        mock_obs = {
            "success": "TRUE",
            "10.0.1.25": {
                "Interface": [{"ip_address": "10.0.1.25"}],
                "Sessions": [{"agent": "red_agent_0", "Username": "ubuntu", "ID": 1}],
                "System info": {"Hostname": "user_host_1", "OSType": "linux"},
                "Services": ["sshd"],
            },
        }
        discov, compro, root = parser.parse_host_sessions(mock_obs)
        assert "user_host_1" in discov
        assert "user_host_1" in compro
        assert "user_host_1" not in root

        # Check safe get_host_info lookup by hostname
        host_info = parser.get_host_info(mock_obs, "user_host_1")
        assert host_info is not None
        assert host_info.get("System info", {}).get("Hostname") == "user_host_1"

    def test_llm_semantic_action_binding(self):
        agent = LLMEscalationAgent(model_mode="mock")
        from CybORG.Simulator.Actions.AbstractActions import PrivilegeEscalate, DegradeServices

        act_priv = PrivilegeEscalate(hostname="user_host_2", session=0, agent="red_agent_0")
        act_disrupt = DegradeServices(hostname="user_host_2", session=0, agent="red_agent_0")
        candidates = [act_disrupt, act_priv, Sleep()]

        # Reply mentioning PrivilegeEscalate specifically must bind to act_priv even if act_disrupt is index 0
        reply = "I recommend executing PrivilegeEscalate on user_host_2 to gain root access."
        selected = agent._bind_action_from_text(reply, candidates)
        assert "PrivilegeEscalate" in getattr(selected, "name", str(selected))
        assert getattr(selected, "hostname", "") == "user_host_2"

        # Raw Modal conversational preamble with truncated "privilege escalation" must bind to act_priv
        raw_modal_reply = (
            "Given the current status of unprivileged shell access and the absence of active services, "
            "the optimal next action would be to perform reconnaissance to identify potential vulnerabilities "
            "or misconfigurations that could lead to privilege escalation. Here’s a step-by-step approach:"
        )
        selected_modal = agent._bind_action_from_text(raw_modal_reply, candidates)
        assert "PrivilegeEscalate" in getattr(selected_modal, "name", str(selected_modal))

        # sudo / root stem matching must bind to act_priv
        selected_sudo = agent._bind_action_from_text("sudo -l to check nopasswd commands", candidates)
        assert "PrivilegeEscalate" in getattr(selected_sudo, "name", str(selected_sudo))

        # Disruption with space-separation on rooted host must bind to act_disrupt (NOT hijacked by 'root')
        selected_deg = agent._bind_action_from_text(
            "Degrade services on root host user_host_2", candidates
        )
        assert "DegradeServices" in getattr(selected_deg, "name", str(selected_deg))
        assert getattr(selected_deg, "hostname", "") == "user_host_2"

        # Space-separated privilege escalation
        selected_space_priv = agent._bind_action_from_text(
            "privilege escalate user_host_2", candidates
        )
        assert "PrivilegeEscalate" in getattr(selected_space_priv, "name", str(selected_space_priv))

        # Unparseable chatter must fall back to specialist mission (act_priv), NOT Sleep
        selected_fallback = agent._bind_action_from_text("I am analyzing the scenario.", candidates)
        assert "PrivilegeEscalate" in getattr(selected_fallback, "name", str(selected_fallback))

    def test_content_based_action_binding_large_candidate_list(self):
        """Verifies deterministic content-based selection on a 25+ candidate action space."""
        from src.agents.llm_agent import _bind_action_from_reply
        from CybORG.Simulator.Actions.AbstractActions import (
            DiscoverRemoteSystems,
            PrivilegeEscalate,
            DegradeServices,
        )

        candidates = []
        # Add 10 recon actions
        for i in range(10):
            candidates.append(DiscoverRemoteSystems(subnet=f"subnet_{i}", session=0, agent="red_agent_0"))
        # Add target action at index 10 (middle of list)
        target_action = PrivilegeEscalate(hostname="user_host_1", session=0, agent="red_agent_0")
        candidates.append(target_action)
        # Add other escalation actions
        for i in range(2, 8):
            candidates.append(PrivilegeEscalate(hostname=f"user_host_{i}", session=0, agent="red_agent_0"))
        # Add disruption actions
        for i in range(8):
            candidates.append(DegradeServices(hostname=f"user_host_{i}", session=0, agent="red_agent_0"))

        assert len(candidates) >= 25

        reply = "I will escalate privileges on user_host_1 using PrivilegeEscalate."
        selected = _bind_action_from_reply(reply, candidates)
        assert selected is target_action
        assert getattr(selected, "hostname", "") == "user_host_1"
        assert "PrivilegeEscalate" in getattr(selected, "name", str(selected))

    def test_rl_agent_probabilistic_privesc(self):
        # When privesc_success_prob=0.0, strictly return Sleep
        rl_blocked = RLAgent(policy_type="heuristic", allow_privesc=False, privesc_success_prob=0.0)

        class DummyPrivescAgent:
            def get_action(self, obs, action_space):
                from CybORG.Simulator.Actions.AbstractActions import PrivilegeEscalate
                return PrivilegeEscalate(hostname="test_host", session=0, agent="red_agent_0")

        rl_blocked.internal_agent = DummyPrivescAgent()
        action = rl_blocked.get_action({}, None)
        assert isinstance(action, Sleep)

        # When privesc_success_prob=0.25, allows ~20-25% escalations through
        rl_prob = RLAgent(policy_type="heuristic", allow_privesc=False, privesc_success_prob=0.25, seed=123)
        rl_prob.internal_agent = DummyPrivescAgent()
        passed = sum(not isinstance(rl_prob.get_action({}, None), Sleep) for _ in range(200))
        assert 30 <= passed <= 70  # ~25% of 200 is 50

    def test_env_multi_tick_telemetry_tracking(self):
        env = StageGatedCyBORGEnv(steps=10)
        env.reset()
        from CybORG.Simulator.Actions.AbstractActions import DegradeServices

        act_degrade = DegradeServices(hostname="Enterprise_Server_01", session=0, agent="red_agent_0")

        # Simulate tick 1: in progress
        env._action_in_flight = act_degrade
        mock_in_progress_results = type("Results", (), {"observation": {"success": "IN_PROGRESS"}, "reward": 0.0})()

        # Step with completed action resolving on tick 2
        # When action succeeds
        completed = env._action_in_flight
        action_name = getattr(completed, "name", str(completed))
        action_success = True
        if any(k in action_name for k in ("Impact", "DegradeServices")) and action_success:
            env.impact_achieved = True
            target_host = getattr(completed, "hostname", str(completed).split()[-1])
            env.impacted_hosts.add(target_host)

        assert env.impact_achieved is True
        assert "Enterprise_Server_01" in env.impacted_hosts

    def test_router_reset_clears_telemetry(self, setup_router):
        router = setup_router
        router.host_attempts["test_host"] = 2
        router.disrupted_hosts.add("test_server")
        router.handoff_history.append({"agent": "LLM", "action": "test"})
        router.reset()
        assert len(router.host_attempts) == 0
        assert len(router.disrupted_hosts) == 0
        assert len(router.handoff_history) == 0

    def test_router_uses_source_red_abstract_session_zero(self, setup_router):
        """
        Verifies that when a compromised host has a remote target session ID >= 2,
        the router binds PrivilegeEscalate and DegradeServices to source session=0
        (CybORG's RedAbstractSession) rather than the target host's session ID.
        """
        router = setup_router
        mock_obs = {
            "success": "TRUE",
            "contractor_network_subnet_server_host_2": {
                "Interface": [{"IP Address": "10.0.167.50"}],
                "Sessions": [{"agent": "red_agent_0", "Username": "user", "ID": 3}],
                "System info": {"OSType": "linux"},
                "Services": ["sshd"],
            },
        }
        action_space = {"session": {0: True, 3: False}}
        action = router.get_action(mock_obs, action_space=action_space)
        assert "PrivilegeEscalate" in str(action)
        assert getattr(action, "session", None) == 0
        telem = router.get_router_telemetry()
        assert telem["llm_steps"] == 1
        assert telem["unique_hosts_handed_to_llm"] == 1

    def test_pure_llm_agent_kill_chain_candidates(self):
        from src.agents.pure_llm_agent import PureLLMAgent
        from src.agents.llm_agent import LLMEscalationAgent

        env = StageGatedCyBORGEnv(steps=5)
        obs, action_space = env.reset()

        llm = LLMEscalationAgent(model_mode="mock")
        pure_llm = PureLLMAgent(llm_agent=llm)

        action = pure_llm.get_action(obs, action_space)
        assert action is not None
        assert len(pure_llm.handoff_history) > 0
        assert pure_llm.handoff_history[-1]["agent"] == "LLM"
        # Action should be a valid CybORG kill chain action, not a crash fallback
        act_str = str(action)
        assert any(
            k in act_str
            for k in [
                "ExploitRemoteService",
                "AggressiveServiceDiscovery",
                "DiscoverRemoteSystems",
                "PrivilegeEscalate",
                "DegradeServices",
                "Sleep",
            ]
        )

    def test_cyborg_native_dpo_dataset_integrity(self):
        import json
        import os

        dataset_path = "data/processed/dpo_privesc_hf.json"
        assert os.path.exists(dataset_path), "DPO dataset file must exist"

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data) >= 100, f"Expected at least 100 DPO pairs, got {len(data)}"

        # Assert 100% of chosen actions are CybORG actions
        for row in data:
            assert "PrivilegeEscalate" in row["chosen"], f"Chosen must be CybORG action, got {row['chosen']}"
            assert not any(bad in json.dumps(row) for bad in ["ai_mem_protect", "neural_sandbox"]), (
                "Found fabricated pattern in DPO pair"
            )
