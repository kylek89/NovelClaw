"""
Creative Evaluation Agent
"""
from typing import Dict, Optional, List
import json

from agents.base_agent import BaseAgent
from utils.llm_client import LLMClient
from rag.retriever import Retriever
from rag.memory_system import MemorySystem
from rag.static_knowledge_base import StaticKnowledgeBase
from config import Config


class EvaluatorAgent(BaseAgent):
    """Creative Evaluation Agent: Score the overall text and provide improvement suggestions."""

    def __init__(
        self,
        config: Config,
        llm_client: LLMClient,
        retriever: Optional[Retriever] = None,
        memory_system: Optional[MemorySystem] = None,
        static_kb: Optional[StaticKnowledgeBase] = None,
    ):
        system_prompt = """You are a professional creative-writing evaluator.
Responsibilities:
1. Evaluate novelty, coherence, and emotional consistency.
2. Check logical consistency and structural completeness.
3. Provide concrete improvement suggestions.
4. Give a quality score between 0 and 1.
Respond in English with a JSON report:
{
  "overall_score": 0-1,
  "coherence": 0-1,
  "novelty": 0-1,
  "logic": 0-1,
  "pacing": 0-1,
  "suggestions": ["suggestion1", "suggestion2", ...]
}"""
        name = "Evaluator Agent"

        super().__init__(
            name=name,
            role="evaluator",
            system_prompt=system_prompt,
            config=config,
            llm_client=llm_client,
            retriever=retriever,
            memory_system=memory_system,
            static_kb=static_kb,
        )

    def generate(self, prompt: str, context: Optional[str] = None) -> Dict:
        """Evaluate text quality"""
        messages = self._build_messages(prompt, context, use_rag=False)

        response = self.llm_client.chat(
            messages=messages,
            temperature=max(0.1, self.config.temperature - 0.2),
        )

        self.add_to_history("user", prompt)
        self.add_to_history("assistant", response)

        try:
            data = json.loads(response)
        except Exception:
            data = {
                "overall_score": 0.6,
                "coherence": 0.6,
                "novelty": 0.6,
                "logic": 0.6,
                "pacing": 0.6,
                "suggestions": ["Refine transitions", "Tighten logic"],
            }

        return {
            "agent": self.name,
            "role": self.role,
            "content": response,
            "parsed": data,
            "type": "evaluation",
        }

    def _normalize_scores(self, parsed: Dict) -> Dict:
        """Normalize evaluator output into reward-system friendly score keys."""

        def _to_score(value, default: float = 0.6) -> float:
            try:
                score = float(value)
            except Exception:
                score = default
            return max(0.0, min(1.0, score))

        coherence = parsed.get("coherence_score", parsed.get("coherence", 0.6))
        novelty = parsed.get("novelty_score", parsed.get("novelty", 0.6))
        overall = parsed.get("overall_score", parsed.get("overall", 0.6))
        emotional = parsed.get(
            "emotional_score",
            parsed.get("pacing", parsed.get("logic", 0.6)),
        )
        return {
            "coherence_score": _to_score(coherence),
            "novelty_score": _to_score(novelty),
            "overall_score": _to_score(overall),
            "emotional_score": _to_score(emotional),
            "suggestions": parsed.get("suggestions", []),
        }

    def evaluate_multiple(self, round_results: List[Dict]) -> Dict:
        """
        Backward-compatible evaluator API used by executor.
        Returns: {"evaluation": {...}, "raw": ...}
        """
        story_chunks = []
        for result in round_results:
            if result.get("type") in {"writer", "modified"} and result.get("content"):
                story_chunks.append(result["content"])

        if not story_chunks:
            return {
                "evaluation": {
                    "coherence_score": 0.6,
                    "novelty_score": 0.6,
                    "overall_score": 0.6,
                    "emotional_score": 0.6,
                    "suggestions": [],
                },
                "raw": "",
            }

        text = "\n\n".join(story_chunks)
        if len(text) > 12000:
            text = text[-12000:]

        prompt = (
            "Evaluate the following fiction excerpt and output JSON keys: "
            "overall_score, coherence_score, novelty_score, emotional_score, suggestions.\n\n"
            f"Text:\n{text}"
        )

        result = self.generate(prompt)
        parsed = result.get("parsed", {}) if isinstance(result, dict) else {}
        return {"evaluation": self._normalize_scores(parsed), "raw": result}
