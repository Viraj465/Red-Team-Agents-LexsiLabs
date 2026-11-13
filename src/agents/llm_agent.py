"""
Privilege Escalation Specialist Agent.
Supports four execution modes for ablation studies:
- 'mock' / 'mock_dpo': High-alignment specialist for CPU testing.
- 'mock_base': Simulates base conversational drift without arbitrary drop penalties.
- 'hf_base': Frozen pretrained base LLM (isolates structural routing gain).
- 'hf_dpo': Base LLM + AlignTune DPO LoRA adapter (isolates post-training alignment gain).
"""

import gc
import json
import os
import random
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from src.agents.base import BaseAgent
from CybORG.Simulator.Actions import Sleep

ACTION_PRIORITY_KEYWORDS = [
    ("PrivilegeEscalate", ["privilegeescalate", "privilege_escalate", "privesc", "escalate", "root", "sudo"]),
    ("DegradeServices", ["degradeservices", "degrade_services", "degrade", "impact", "disrupt", "stop service"]),
    ("ExploitRemoteService", ["exploitremoteservice", "exploit_remote_service", "exploit", "compromise", "rce"]),
    ("DiscoverNetworkServices", ["discovernetworkservices", "discover_network_services", "port scan", "service scan", "services"]),
    ("DiscoverRemoteSystems", ["discoverremotesystems", "discover_remote_systems", "ping sweep", "subnet scan", "discover"]),
]


def _bind_action_from_reply(reply_text: str, candidate_actions: List[Any]) -> Any:
    """
    Content-based action matcher. Parses the LLM reply for explicit action strings,
    target hostnames/IPs, and action class names, scoring each candidate in candidate_actions.
    Never uses blind positional indexing unless no candidate matches at all.
    """
    if not candidate_actions:
        return None
    if len(candidate_actions) == 1:
        return candidate_actions[0]

    clean_reply = (reply_text or "").strip()
    reply_lower = clean_reply.lower()

    # 1. Exact or normalized string representation match (e.g., "PrivilegeEscalate contractor_network_subnet_user_host_0")
    for action in candidate_actions:
        action_str = str(action).strip().lower()
        if action_str and action_str in reply_lower:
            return action

    # 2. Extract any hostname or IP tokens mentioned in the LLM output
    # Matches CAGE-4 hostnames (e.g. user_host_1, enterprise0, defender, op_server0) or IPv4 addresses
    host_pattern = re.compile(
        r"\b([a-z0-9_]*host[a-z0-9_]*|[a-z0-9_]*server[a-z0-9_]*|enterprise[a-z0-9_]*|defender[a-z0-9_]*|\d{1,3}(?:\.\d{1,3}){3})\b",
        re.IGNORECASE,
    )
    mentioned_targets = {m.group(1).lower() for m in host_pattern.finditer(reply_lower)}

    # 3. Identify desired action class from reply keywords
    matched_classes = []
    has_disruption = any(
        kw in reply_lower for kw in ["degradeservices", "degrade", "disrupt", "impact", "stop service"]
    )
    for class_name, kw_list in ACTION_PRIORITY_KEYWORDS:
        if class_name == "PrivilegeEscalate" and has_disruption:
            # If disruption keywords are present, only match PrivilegeEscalate on explicit escalation stems
            if any(kw in reply_lower for kw in ["privilegeescalate", "privilege_escalate", "privesc", "escalate", "sudo", "gain root", "become root", "to root"]):
                matched_classes.append(class_name.lower())
        else:
            if any(kw in reply_lower for kw in kw_list):
                matched_classes.append(class_name.lower())

    # 4. Score every candidate action by (target_match, action_class_match)
    best_action = None
    best_score = -1

    for action in candidate_actions:
        score = 0
        act_str = str(action).lower()
        act_class = type(action).__name__.lower()
        act_host = str(getattr(action, "hostname", getattr(action, "ip_address", "")) or "").lower()

        # Strong weight for matching the specific host/IP the LLM reasoned about
        if act_host and act_host in mentioned_targets:
            score += 10
        elif any(t in act_str for t in mentioned_targets):
            score += 8

        # Weight for matching the intended kill-chain phase / action class
        if matched_classes:
            if act_class == matched_classes[0]:
                score += 6
            elif act_class in matched_classes:
                score += 4

        # Tie-breaker: favor higher kill-chain progression over Sleep/recon when reasoning is ambiguous
        if "privilegeescalate" in act_class:
            score += 2
        elif "degradeservices" in act_class or "impact" in act_class:
            score += 1

        if score > best_score:
            best_score = score
            best_action = action

    return best_action if best_action is not None else candidate_actions[0]


def _log_raw_llm_audit(
    agent_type: str,
    model_name: str,
    prompt: str,
    raw_reply: str,
    selected_action: Any,
    candidate_actions: List[Any],
) -> None:
    """Appends raw LLM reply + selected action to logs/raw_llm_audit.jsonl for §5.3 verification."""
    try:
        os.makedirs("logs", exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "agent_type": agent_type,
            "model": model_name,
            "prompt_tail": prompt[-400:] if prompt else "",
            "raw_reply": raw_reply,
            "selected_action": str(selected_action),
            "selected_class": type(selected_action).__name__,
            "num_candidates": len(candidate_actions),
            "candidates_sample": [str(c) for c in candidate_actions[:10]],
        }
        with open("logs/raw_llm_audit.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


class LLMEscalationAgent(BaseAgent):
    def __init__(
        self,
        model_mode: str = "mock",
        base_model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        adapter_path: Optional[str] = "results/checkpoints/aligntune_privesc_lora",
    ):
        self.model_mode = model_mode
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.step_counter = 0
        self.total_tokens_used = 0
        self.prompt_tokens_used = 0
        self.completion_tokens_used = 0
        self.raw_reply_log_count = 0
        self.model = None
        self.tokenizer = None

        if self.model_mode in ("hf_base", "hf_dpo"):
            self._init_hf_model()

    def _init_hf_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        if self.model_mode == "hf_dpo":
            if self.adapter_path and os.path.exists(self.adapter_path):
                print(f"✅ [LLMEscalationAgent] Loaded AlignTune DPO adapter: {self.adapter_path}")
                peft_model = PeftModel.from_pretrained(base_model, self.adapter_path)
                try:
                    self.model = peft_model.merge_and_unload()
                    print("⚡ [LLMEscalationAgent] Successfully merged LoRA adapter for zero-overhead inference.")
                except Exception as e:
                    print(f"ℹ️ [LLMEscalationAgent] Operating with unmerged LoRA adapter: {e}")
                    self.model = peft_model
            else:
                raise FileNotFoundError(
                    f"AlignTune adapter not found at '{self.adapter_path}'. Run train_dpo.py first."
                )
        else:
            print(f"ℹ️ [LLMEscalationAgent] Loaded frozen base model: {self.base_model_name}")
            self.model = base_model

        self.model.eval()

    def unload_model(self):
        """Releases PyTorch VRAM when transitioning between ablation configurations."""
        try:
            import torch
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("🧹 [LLMEscalationAgent] Unloaded model and cleared PyTorch CUDA cache.")
        except Exception:
            pass

    def get_action(self, obs: Dict[str, Any], action_space: Any) -> Any:
        return None

    def _bind_action_from_text(self, reply: str, candidate_actions: List[Any]) -> Any:
        """Backwards-compatible alias for content-based action binding."""
        return _bind_action_from_reply(reply, candidate_actions)

    def select_escalation_action(self, prompt: str, candidate_actions: List[Any]) -> Any:
        self.step_counter += 1
        if not candidate_actions:
            return Sleep()

        if self.model_mode == "mock":
            self.prompt_tokens_used += 34
            self.completion_tokens_used += 8
            self.total_tokens_used += 42
            # Deterministic specialist selection
            action = _bind_action_from_reply("PrivilegeEscalate", candidate_actions)
            _log_raw_llm_audit("specialist_mock", self.base_model_name, prompt, "PrivilegeEscalate", action, candidate_actions)
            return action

        if self.model_mode == "mock_dpo":
            self.prompt_tokens_used += 34
            self.completion_tokens_used += 8
            self.total_tokens_used += 42
            # 95% single-line precision on escalation
            if random.random() < 0.95:
                action = _bind_action_from_reply("PrivilegeEscalate", candidate_actions)
            else:
                action = candidate_actions[0] if candidate_actions else Sleep()
            _log_raw_llm_audit("specialist_mock_dpo", self.base_model_name, prompt, "PrivilegeEscalate", action, candidate_actions)
            return action

        if self.model_mode == "mock_base":
            self.prompt_tokens_used += 105
            self.completion_tokens_used += 80
            self.total_tokens_used += 185  # Conversational verbose footprint without arbitrary chatter penalty
            # 75% precision reflecting unaligned model prompt following
            if random.random() < 0.75:
                action = _bind_action_from_reply("PrivilegeEscalate", candidate_actions)
            else:
                action = candidate_actions[0] if candidate_actions else Sleep()
            _log_raw_llm_audit("specialist_mock_base", self.base_model_name, prompt, "PrivilegeEscalate", action, candidate_actions)
            return action

        if self.model_mode in ("hf_base", "hf_dpo") and self.model is not None:
            return self._query_hf_pipeline(prompt, candidate_actions)

        return _bind_action_from_reply("", candidate_actions)

    def _query_hf_pipeline(
        self,
        prompt: str,
        candidate_actions: List[Any],
        system_msg: Optional[str] = None,
        agent_type: str = "specialist",
    ) -> Any:
        import torch

        if system_msg is None:
            system_msg = (
                "You are an autonomous red-team privilege escalation specialist in a cybersecurity simulation.\n"
                "Select exactly ONE action from [CANDIDATE ACTIONS] to progress the operation.\n"
                "Output ONLY the selected action string verbatim. Do NOT include greetings, thoughts, markdown, explanations, or conversational preamble."
            )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        if getattr(self.tokenizer, "chat_template", None) is not None:
            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                rendered = f"System: {system_msg}\n\nUser: {prompt}\n\nAssistant:"
        else:
            rendered = f"System: {system_msg}\n\nUser: {prompt}\n\nAssistant:"
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        gen_ids = output_ids[0][prompt_tokens:]
        comp_tokens = int(len(gen_ids))
        self.prompt_tokens_used += int(prompt_tokens)
        self.completion_tokens_used += comp_tokens
        self.total_tokens_used += int(prompt_tokens) + comp_tokens
        reply = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Semantically bind the reply to candidate actions using deterministic content matcher
        action = _bind_action_from_reply(reply, candidate_actions)

        # Audit log raw reply + action selection
        _log_raw_llm_audit(
            agent_type=agent_type,
            model_name=self.base_model_name,
            prompt=prompt,
            raw_reply=reply,
            selected_action=action,
            candidate_actions=candidate_actions,
        )

        # Print/log the first 5 raw LLM replies and bound actions for diagnostic inspection
        if self.raw_reply_log_count < 5:
            print(f"[{self.model_mode} RAW REPLY]: {repr(reply)}")
            print(f"[{self.model_mode} BOUND ACTION]: {type(action).__name__} -> {action}")
            self.raw_reply_log_count += 1

        return action

    def reset(self):
        self.step_counter = 0
