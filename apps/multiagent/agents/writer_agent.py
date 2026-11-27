"""
Writer agent for long-form story chapter generation.
"""
from typing import Dict, List, Optional

from agents.base_agent import BaseAgent
from config import Config
from rag.memory_system import MemorySystem
from rag.retriever import Retriever
from rag.static_knowledge_base import StaticKnowledgeBase
from utils.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Writer agent: generate narrative prose for the current chapter."""

    def __init__(
        self,
        config: Config,
        llm_client: LLMClient,
        retriever: Optional[Retriever] = None,
        memory_system: Optional[MemorySystem] = None,
        static_kb: Optional[StaticKnowledgeBase] = None,
    ):
        system_prompt = (
            "You are an expert long-form fiction writer.\n"
            "Hard rules:\n"
            "1) Follow the given premise/outline and remain consistent.\n"
            "2) Keep continuity with prior content; avoid jumps and repetition.\n"
            "3) Output narrative prose only (no outlines/bullets).\n"
            "4) If context is incomplete, extend from existing clues instead of unsupported fabrication."
        )
        name = "Writer Agent"

        super().__init__(
            name=name,
            role="writer",
            system_prompt=system_prompt,
            config=config,
            llm_client=llm_client,
            retriever=retriever,
            memory_system=memory_system,
            static_kb=static_kb,
        )

    def _length_hint(
        self,
        target_length: Optional[int],
        target_min_chars: Optional[int] = None,
        target_max_chars: Optional[int] = None,
    ) -> str:
        if target_min_chars and target_max_chars and target_max_chars > target_min_chars:
            return (
                f" Hard constraint: output {target_min_chars}-{target_max_chars} words. "
                "Do not exceed the upper bound. If you reach it, stop immediately."
            )

        if not target_length or target_length <= 0:
            return " Aim for about 1000-1800 words."

        low = max(800, int(target_length * 0.9))
        high = max(low + 200, int(target_length * 1.12))
        return (
            f" Hard constraint: output {low}-{high} words. "
            "Do not exceed the upper bound."
        )

    def _estimate_max_tokens(
        self,
        target_length: Optional[int],
        target_max_chars: Optional[int] = None,
    ) -> Optional[int]:
        hard_cap = target_max_chars if target_max_chars and target_max_chars > 0 else target_length
        if not hard_cap or hard_cap <= 0:
            return None
        # Estimate a token budget from the requested English word count.
        est = int(hard_cap * 1.5)
        est = max(512, est)
        return min(est, int(getattr(self.config, "max_tokens", est)))

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        topic: Optional[str] = None,
        genre: Optional[str] = None,
        style_tags: Optional[List[str]] = None,
        target_length: Optional[int] = None,
        target_min_chars: Optional[int] = None,
        target_max_chars: Optional[int] = None,
    ) -> Dict:
        """Generate prose for the current chapter."""
        hint = self._length_hint(
            target_length,
            target_min_chars=target_min_chars,
            target_max_chars=target_max_chars,
        )
        max_tokens = self._estimate_max_tokens(target_length, target_max_chars=target_max_chars)

        messages = self._build_messages(
            prompt + hint,
            context,
            use_rag=True,
            topic=topic,
            use_memory=True,
            use_static_kb=bool(getattr(self.config, "enable_static_kb", False)),
            genre=genre,
            style_tags=style_tags,
        )

        response = self.llm_client.chat(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=max_tokens,
        )

        self.add_to_history("user", prompt)
        self.add_to_history("assistant", response)

        return {
            "agent": self.name,
            "role": self.role,
            "content": response,
            "type": "writer",
        }
