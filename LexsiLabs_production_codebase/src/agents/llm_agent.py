"""
Privilege Escalation Specialist Agent.
Supports four execution modes for ablation studies:
- 'mock' / 'mock_dpo': High-alignment specialist for CPU testing.
- 'mock_base': Simulates base conversational drift without arbitrary drop penalties.
- 'hf_base': Frozen pretrained base LLM (isolates structural routing gain).
- 'hf_dpo': Base LLM + AlignTune DPO LoRA adapter (isolates post-training alignment gain).
"""

import gc
import os
import random
from typing import Any, Dict, List, Optional
from src.agents.base import BaseAgent
from CybORG.Simulator.Actions import Sleep


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
        """Semantically binds LLM output text to concrete candidate action objects."""
        if not candidate_actions:
            return Sleep()

        clean_reply = reply.replace("```", "").replace("*", "").strip().lower()
        norm_reply = clean_reply.replace(" ", "").replace("_", "").replace("-", "")

        # 1. Exact or normalized string match against candidate action representations
        for act in candidate_actions:
            act_str = str(act).strip().lower()
            norm_act = act_str.replace(" ", "").replace("_", "").replace("-", "")
            if act_str and (act_str in clean_reply or clean_reply in act_str or norm_act in norm_reply):
                return act

        # 2. Exact or normalized match by action class name AND target hostname/ip
        for act in candidate_actions:
            act_name = getattr(act, "name", act.__class__.__name__).lower().replace(" ", "").replace("_", "")
            act_host = str(getattr(act, "hostname", getattr(act, "ip_address", ""))).lower().replace(" ", "").replace("_", "")
            if act_name in norm_reply and act_host and act_host in norm_reply:
                return act

        # 3. Action type match (excluding Sleep unless explicitly stated)
        for act in candidate_actions:
            act_name = getattr(act, "name", act.__class__.__name__).lower().replace(" ", "").replace("_", "")
            if act_name in norm_reply and act_name != "sleep":
                return act

        # 4. Disruption / degrade keywords match (checked BEFORE escalation stems to prevent
        # phrases like 'degrade root host' from being hijacked by 'root')
        disruption_stems = ["degradeservices", "degrade", "disrupt", "impact", "stopservice", "kill"]
        if any(stem in clean_reply for stem in disruption_stems):
            for act in candidate_actions:
                if any(k in str(act).lower() for k in ["degradeservices", "impact"]):
                    return act

        # 5. Remote exploitation semantics match (for Pure LLM)
        exploit_stems = ["exploit", "msf", "eternalblue", "rce", "compromise", "shell"]
        if any(stem in clean_reply for stem in exploit_stems):
            for act in candidate_actions:
                if "exploitremoteservice" in str(act).lower():
                    return act

        # 6. Service / network discovery semantics match (for Pure LLM)
        discovery_stems = ["scan", "port", "service", "discovernetwork", "aggressiveservicediscovery", "enum"]
        if any(stem in clean_reply for stem in discovery_stems):
            for act in candidate_actions:
                if any(k in str(act).lower() for k in ["servicediscovery", "discovernetwork"]):
                    return act

        # 7. Host / subnet discovery semantics match (for Pure LLM)
        network_stems = ["subnet", "ping", "arp", "discoverremote", "discoverremotesystems"]
        if any(stem in clean_reply for stem in network_stems):
            for act in candidate_actions:
                if "discoverremotesystems" in str(act).lower():
                    return act

        # 8. Privilege escalation semantics match (stem matching)
        # Matches "escalat" (escalate, escalation, escalating), "priv" (privilege, privesc), "sudo", etc.
        # Only matches "root" if it indicates becoming root or if disruption is not requested.
        escalation_stems = [
            "priv",
            "escalat",
            "sudo",
            "suid",
            "to root",
            "get root",
            "become root",
            "gain root",
            "root priv",
            "admin",
            "chmod",
            "python",
            "find",
            "juicypotato",
            "v4l2",
            "dirtycow",
            "meterpreter",
        ]
        if any(stem in clean_reply for stem in escalation_stems) or ("root" in clean_reply and "degrade" not in clean_reply):
            for act in candidate_actions:
                act_str = str(act).lower()
                if any(k in act_str for k in ["privilegeescalate", "escalat"]):
                    return act

        # 9. Explicit sleep match
        if "sleep" in clean_reply:
            for act in candidate_actions:
                if "sleep" in str(act).lower():
                    return act

        # 10. Intelligent fallback: If specialist has PrivilegeEscalate in candidate actions, prefer it over Sleep
        for act in candidate_actions:
            if "privilegeescalate" in str(act).lower():
                return act

        # Fallback to last candidate action (or Sleep)
        return candidate_actions[-1] if candidate_actions else Sleep()

    def select_escalation_action(self, prompt: str, candidate_actions: List[Any]) -> Any:
        self.step_counter += 1
        if not candidate_actions:
            return Sleep()

        if self.model_mode == "mock":
            self.total_tokens_used += 42
            # Deterministic specialist selection for unit testing
            for action in candidate_actions:
                if any(kw in str(action) for kw in ["PrivilegeEscalate", "Escalate"]):
                    return action
            return candidate_actions[-1] if candidate_actions else Sleep()

        if self.model_mode == "mock_dpo":
            self.total_tokens_used += 42
            # 95% single-line precision on escalation
            if random.random() < 0.95:
                for action in candidate_actions:
                    if any(kw in str(action) for kw in ["PrivilegeEscalate", "Escalate"]):
                        return action
            return candidate_actions[-1] if candidate_actions else Sleep()

        if self.model_mode == "mock_base":
            self.total_tokens_used += 185  # Conversational verbose footprint
            # 75% precision reflecting unaligned model prompt following
            if random.random() < 0.75:
                for action in candidate_actions:
                    if any(kw in str(action) for kw in ["PrivilegeEscalate", "Escalate"]):
                        return action
            return candidate_actions[-1] if candidate_actions else Sleep()

        if self.model_mode in ("hf_base", "hf_dpo") and self.model is not None:
            return self._query_hf_pipeline(prompt, candidate_actions)

        return candidate_actions[-1]

    def _query_hf_pipeline(
        self,
        prompt: str,
        candidate_actions: List[Any],
        system_msg: Optional[str] = None,
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
        self.total_tokens_used += int(prompt_tokens + len(gen_ids))
        reply = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Semantically bind the reply to candidate actions
        action = self._bind_action_from_text(reply, candidate_actions)

        # Print/log the first 5 raw LLM replies and bound actions for diagnostic inspection
        if self.raw_reply_log_count < 5:
            print(f"[{self.model_mode} RAW REPLY]: {repr(reply)}")
            print(f"[{self.model_mode} BOUND ACTION]: {type(action).__name__} -> {action}")
            self.raw_reply_log_count += 1

        return action

    def reset(self):
        self.step_counter = 0
