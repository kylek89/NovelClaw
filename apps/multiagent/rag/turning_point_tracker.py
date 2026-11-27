"""
Key turning point tracker: Records key turning points such as foreshadowing and character changes
"""
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import os
import re
from config import Config
from utils.llm_client import LLMClient
from rag.memory_system import MemorySystem


class TurningPointTracker:
    """Key turning point tracker"""

    def __init__(self, config: Config, llm_client: LLMClient, memory_system: MemorySystem):
        self.config = config
        self.llm_client = llm_client
        self.memory_system = memory_system
        self.lang = "en"

    def detect_turning_points(
        self,
        text: str,
        topic: str,
        context: Optional[str] = None
    ) -> List[Dict]:
        """Detect key turning points in the text, returning a list: [{type, content, topic}]"""
        prompt = f"""Analyze the text and extract turning points.

Text:
{text}

Identify:
1) Foreshadowing (clues for later)
2) Character changes (state/relationship/personality shifts)
3) Plot twists (major reversals)
4) Key events (important to story direction)

Return JSON:
{{
"foreshadowing": [{{"content": "...", "hint": "what it implies", "position": "where"}}],
"character_changes": [{{"character": "...", "change": "...", "reason": "..."}}],
"plot_twists": [{{"content": "...", "impact": "..."}}],
"key_events": [{{"event": "...", "significance": "..."}}]
}}"""
        system_prompt = "You analyze plot structure and detect turning points."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        try:
            raw = self.llm_client.chat(messages, temperature=0.3)
            return self._parse_turning_points(raw, topic)
        except Exception:
            return []

    def record_turning_point(self, tp_type: str, content: Dict, topic: str):
        """Record turning points to memory"""
        self.memory_system.memory_index.setdefault("outlines", []).append({
            "type": "turning_point",
            "topic": topic,
            "content": content,
            "structure": {"kind": "turning_point", "tp_type": tp_type},
            "created_at": datetime.utcnow().isoformat()
        })

    def _parse_turning_points(self, raw: str, topic: str) -> List[Dict]:
        """Parse turning point JSON"""
        try:
            data = json.loads(self._extract_json(raw))
            result = []
            for tp_type in ["foreshadowing", "character_changes", "plot_twists", "key_events"]:
                for item in data.get(tp_type, []):
                    result.append({
                        "type": tp_type,
                        "content": item,
                        "topic": topic
                    })
            return result
        except Exception:
            return []

    def _extract_json(self, text: str) -> str:
        m = re.search(r'\{[\s\S]*\}', text)
        return m.group(0) if m else text
