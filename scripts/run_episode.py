
"""
initializes the StageGatedCybORGEnv, sets up the RLAgent and LLMEscalationAgent, wraps them in the AdaptiveKillChainRouter, and runs an interactive red-team episode while printing a real-time log of the handoffs in your terminal.
"""


import sys
import warnings
from pathlib import Path

# Suppress deprecation notices from older simulation submodules (e.g. gym)
warnings.filterwarnings("ignore")

# Add project root and cage-challenge-4 to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "cage-challenge-4"))

from src.env.cyborg_wrapper import StageGatedCybORGEnv
from src.agents.rl_agent import RLAgent
from src.agents.llm_agent import LLMEscalationAgent
from src.router.killchain_router import AdaptiveKillChainRouter

def run_single_episode(max_steps: int = 50):
    print("=" * 60)
    print("🎯 STARTING HYBRID RED-TEAM EPISODE (AKCR)")
    print("=" * 60)

    # 1. Initialize environment and agents
    env = StageGatedCybORGEnv()
    rl_agent = RLAgent(policy_type="heuristic")
    llm_agent = LLMEscalationAgent(model_mode="mock")
    router = AdaptiveKillChainRouter(rl_agent=rl_agent, llm_agent=llm_agent)

    # 2. Reset episode
    obs, action_space = env.reset()
    router.reset()

    for step in range(1, max_steps + 1):
        # 3. Router selects which agent drives this step
        action = router.get_action(obs, action_space)

        # 4. Step environment
        obs, reward, done, info = env.step(action)
        action_space = getattr(info.get('raw_results'), 'action_space', action_space)

        # 5. Live telemetry logging
        last_event = router.handoff_history[-1]
        agent_tag = f"[{last_event['agent']}]".ljust(6)
        print(f"Step {step:02d} | {agent_tag} on {last_event['target'].ljust(12)} -> Action: {last_event['action']}")

        if done:
            print("\n🏁 Episode terminated (Impact achieved or max steps reached).")
            break

    # 6. Final Summary
    metrics = env.get_stage_metrics()
    print("\n" + "=" * 60)
    print("📊 FINAL EPISODE STAGE METRICS")
    print("=" * 60)
    print(f"Discovered Hosts  : {metrics['discovered']}")
    print(f"Compromised Hosts : {metrics['compromised']}")
    print(f"Rooted Hosts      : {metrics['rooted']}")
    print(f"Impact Achieved   : {metrics['impact']}")
    print(f"P(Compromise|Disc): {metrics['p_compromise_given_disc']:.2%}")
    print(f"P(Root|Compromise): {metrics['p_root_given_comp']:.2%}")
    print("=" * 60)

if __name__ == "__main__":
    run_single_episode(max_steps=40)