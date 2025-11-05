"""
Lightweight client abstractions for strategy-generation language models.
"""
from __future__ import annotations

import abc
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


class LLMClient(abc.ABC):
    """Abstract LLM client interface used by the evolution loops."""

    @abc.abstractmethod
    def build_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Transform the user prompt into the model specific message payload."""

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 320,
        temperature: float = 0.7,
    ) -> str:
        """Return the raw model output."""


def extract_code_blocks(text: str) -> Optional[str]:
    """
    Extract the first fenced Python code block from *text*.
    """
    import re

    match = re.findall(r"```(?:python)?\s*(.+?)```", text, flags=re.S | re.I)
    if match:
        return match[0].strip()
    if "class Strat" in text:
        return text.strip()
    return None


@dataclass
class LocalHFClient(LLMClient):
    """
    Thin wrapper around a local HuggingFace causal language model.
    """

    tokenizer: Any
    model: Any
    system_prompt: str

    def build_messages(self, prompt: str, *, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        system = textwrap.dedent(system_prompt or self.system_prompt).strip()
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 320,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> str:
        messages = self.build_messages(prompt, system_prompt=system_prompt)
        chat_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        encoded = self.tokenizer(chat_text, return_tensors="pt")
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        eos_token = self.tokenizer.eos_token_id
        output = self.model.generate(
            **encoded,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            eos_token_id=eos_token,
            pad_token_id=eos_token,
        )
        new_tokens = output[0, encoded["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
