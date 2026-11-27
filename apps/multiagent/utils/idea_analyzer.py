"""
Idea Analyzer: Automatically infer style, genre, etc. based on user's idea
"""
from typing import Dict, List, Optional
from utils.llm_client import LLMClient
from config import Config
import json
import re


class IdeaAnalyzer:
    """Idea Analyzer: Extract style, genre, and other information from user's input idea"""

    def __init__(self, config: Config, llm_client: LLMClient):
        self.config = config
        self.llm_client = llm_client

    def analyze_idea(
        self,
        idea: str,
        language: str = "auto"
    ) -> Dict:
        """
        Analyze user's idea to automatically infer style, genre, etc.
        """
        lang = str(language).lower()

        en_prompt = f"""Analyze the creative idea below and extract key information.

Idea:
{idea}

Extract:
1. Story genre (e.g., sci-fi, fantasy, mystery, historical, contemporary).
2. Style tags (e.g., delicate, epic, suspenseful, humorous, serious, romantic; max 5).
3. Text type: creative.
4. Keywords: 3-5 keywords.
5. Target audience: e.g., teen, adult.
6. Language style: e.g., modern, classical, colloquial.

Return JSON:
{{
    "genre": "sci-fi",
    "style_tags": ["delicate", "epic", "suspenseful"],
    "text_type": "creative",
    "keywords": ["AI", "future", "awakening"],
    "target_audience": "adult",
    "language_style": "modern",
    "suggested_length": 5000,
    "complexity": "medium"
}}"""

        prompt = en_prompt
        system_prompt = (
            "You are a creative-analysis expert who extracts genre, style, theme, and related info."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.llm_client.chat(messages, temperature=0.3)
            result = self._parse_analysis(response)

            if not result:
                result = self._default_result(lang)

            # Ensure required fields
            result.setdefault("genre", "general")
            result.setdefault("style_tags", ["delicate"])
            result.setdefault("text_type", "creative")
            result.setdefault("keywords", [])

            return result

        except Exception as e:
            print(f"Analyzing idea failed: {e}")
            return self._default_result(lang)

    def _default_result(self, lang: str) -> Dict:
        return {
            "genre": "general",
            "style_tags": ["immersive", "tight-pacing"],
            "text_type": "creative",
            "keywords": [],
            "target_audience": "adult",
            "language_style": "modern",
            "suggested_length": 5000,
            "complexity": "medium"
        }

    def _parse_analysis(self, response: str) -> Dict:
        """Parse analysis results"""
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        return {}

    def extract_topic_from_idea(self, idea: str) -> str:
        """
        Extract theme from idea (for generation)
        """
        if len(idea) < 100:
            return idea

        prompt = f"""Extract a concise topic (<=50 words) from the idea below.

{idea}

Output only the topic, nothing else."""
        system = "You are an extraction expert."

        messages = [
            {
                "role": "system",
                "content": system
            },
            {"role": "user", "content": prompt}
        ]

        try:
            topic = self.llm_client.chat(messages, temperature=0.3)
            return topic.strip()[:100]  # Limit length
        except:
            return idea[:100]
