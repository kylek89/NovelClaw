"""
Task Understanding Module (Analyzer)
"""
from typing import Dict, List
from utils.llm_client import LLMClient
from config import Config
import json
import re


class Analyzer:
    """Task Understanding Module: Analyze task requirements and complexity"""

    def __init__(self, config: Config, llm_client: LLMClient):
        self.config = config
        self.llm_client = llm_client

    def analyze_task(
        self,
        topic: str,
        text_type: str = "creative",
        target_length: int = 5000
    ) -> Dict:
        """
        Analyze task requirements
        """
        prompt = f"""Analyze the following text-generation task and assess complexity.

Topic: {topic}
Text type: {text_type}
Target length: {target_length} words

Provide:
1. Plot depth requirement (1-10)
2. Character diversity requirement (1-10)
3. Background info requirement (1-10)
4. Creativity requirement (1-10)
5. Logic rigor requirement (1-10)
6. Task complexity (simple/medium/complex)

Output JSON:
{{
"plot_depth": 5,
"character_diversity": 5,
"background_info": 5,
"creativity": 5,
"logic_rigor": 5,
"complexity": "medium",
"recommended_agents": ["plot", "character", "world"],
"rag_required": true,
"estimated_iterations": 5
}}"""
        system_prompt = "You are a task analysis expert for long-form text generation."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.chat(messages, temperature=0.3)
        analysis_result = self._parse_analysis(response)

        if not analysis_result:
            analysis_result = {
                "plot_depth": 7,
                "character_diversity": 6,
                "background_info": 7,
                "creativity": 8,
                "logic_rigor": 6,
                "complexity": "medium",
                "recommended_agents": ["plot", "character", "world", "writer", "retrieval"],
                "rag_required": True,
                "estimated_iterations": 5
            }

        rec = analysis_result.get("recommended_agents", [])
        if "writer" not in rec:
            rec.append("writer")
        if "evaluator" not in rec:
            rec.append("evaluator")
        analysis_result["recommended_agents"] = rec

        return analysis_result

    def _parse_analysis(self, response: str) -> Dict:
        """Parse the JSON from the analysis results"""
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass

        return {}
