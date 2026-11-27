"""
Consistency check module: detects and verifies the consistency of generated content
"""
from typing import Dict, List, Optional, Tuple
from utils.llm_client import LLMClient
from rag.memory_system import MemorySystem
from config import Config
import re
import json


class ConsistencyChecker:
    """Consistency checker: verifies the consistency of generated content with historical memory"""

    def __init__(self, config: Config, llm_client: LLMClient, memory_system: MemorySystem):
        self.config = config
        self.llm_client = llm_client
        self.memory_system = memory_system
        self.lang = "en"

    def check_character_consistency(
        self,
        generated_text: str,
        topic: str,
        character_name: Optional[str] = None
    ) -> Dict:
        """Check character consistency"""
        if character_name:
            memories = self.memory_system.retrieve_memories(
                f"{character_name} character traits",
                memory_types=["character"],
                topic=topic,
                top_k=3
            )
        else:
            character_names = self._extract_character_names(generated_text)
            if character_names:
                memories = []
                for name in character_names[:3]:
                    mems = self.memory_system.retrieve_memories(
                        f"{name} character traits",
                        memory_types=["character"],
                        topic=topic,
                        top_k=2
                    )
                    memories.extend(mems)
            else:
                memories = []

        if not memories:
            return {
                "consistent": True,
                "confidence": 0.5,
                "issues": [],
                "suggestions": []
            }

        character_info = "\n\n".join([m["text"] for m in memories[:5]])

        prompt = f"""Check whether character depictions in the generated text match existing character settings.

Existing character settings:
{character_info}

Generated text:
{generated_text}

Check:
1) Personality consistency
2) Background consistency
3) Behavior fits established traits
4) Any contradictions/conflicts

Return JSON:
{{
"consistent": true/false,
"confidence": 0.0-1.0,
"issues": ["issue1", "issue2"],
"suggestions": ["suggestion1", "suggestion2"]
}}"""
        system_prompt = "You are a consistency checker for fiction. Respond in English."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.llm_client.chat(messages, temperature=0.3)
            result = self._parse_consistency_result(response)
            return result
        except:
            return {
                "consistent": True,
                "confidence": 0.5,
                "issues": [],
                "suggestions": []
            }

    def check_plot_consistency(
        self,
        generated_text: str,
        topic: str,
        baseline_outline: Optional[str] = None
    ) -> Dict:
        """Check plot consistency"""
        if baseline_outline:
            plot_info = baseline_outline
        else:
            memories = self.memory_system.retrieve_memories(
                generated_text[:200],
                memory_types=["plot_point", "outline"],
                topic=topic,
                top_k=5
            )

            if not memories:
                return {
                    "consistent": True,
                    "confidence": 0.5,
                    "issues": [],
                    "suggestions": []
                }

            plot_info = "\n\n".join([m["text"] for m in memories])

        prompt = f"""Check if the plot in the generated text aligns with the outline and plot points.

Existing outline/plot points:
{plot_info}

Generated text:
{generated_text}

Check:
1) Plot progression matches outline
2) Logical contradictions
3) Timeline errors
4) Other inconsistencies

Return JSON:
{{
"consistent": true/false,
"confidence": 0.0-1.0,
"issues": ["issue1", "issue2"],
"suggestions": ["suggestion1", "suggestion2"]
}}"""
        system_prompt = "You are a plot consistency checker. Respond in English."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.llm_client.chat(messages, temperature=0.3)
            result = self._parse_consistency_result(response)
            return result
        except:
            return {
                "consistent": True,
                "confidence": 0.5,
                "issues": [],
                "suggestions": []
            }

    def check_world_consistency(
        self,
        generated_text: str,
        topic: str
    ) -> Dict:
        """Check world consistency"""
        memories = self.memory_system.retrieve_memories(
            "World setting",
            memory_types=["world_setting"],
            topic=topic,
            top_k=5
        )

        if not memories:
            return {
                "consistent": True,
                "confidence": 0.5,
                "issues": [],
                "suggestions": []
            }

        world_info = "\n\n".join([m["text"] for m in memories])

        prompt = f"""Check whether the generated text follows the established world/setting rules.

Existing world info:
{world_info}

Generated text:
{generated_text}

Check:
1) Conforms to rules/constraints of the world
2) Any violations
3) Logical contradictions

Return JSON:
{{
"consistent": true/false,
"confidence": 0.0-1.0,
"issues": ["issue1", "issue2"],
"suggestions": ["suggestion1", "suggestion2"]
}}"""
        system_prompt = "You are a worldbuilding consistency checker. Respond in English."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.llm_client.chat(messages, temperature=0.3)
            result = self._parse_consistency_result(response)
            return result
        except:
            return {
                "consistent": True,
                "confidence": 0.5,
                "issues": [],
                "suggestions": []
            }

    def comprehensive_check(
        self,
        generated_text: str,
        topic: str,
        baseline_outline: Optional[str] = None
    ) -> Dict:
        """
        Comprehensive consistency check
        """
        results = {
            "character": self.check_character_consistency(generated_text, topic),
            "plot": self.check_plot_consistency(generated_text, topic, baseline_outline=baseline_outline),
            "world": self.check_world_consistency(generated_text, topic)
        }

        all_consistent = all(r["consistent"] for r in results.values())
        avg_confidence = sum(r["confidence"] for r in results.values()) / len(results)

        all_issues = []
        all_suggestions = []
        for r in results.values():
            all_issues.extend(r.get("issues", []))
            all_suggestions.extend(r.get("suggestions", []))

        return {
            "overall_consistent": all_consistent,
            "overall_confidence": avg_confidence,
            "details": results,
            "all_issues": all_issues,
            "all_suggestions": all_suggestions
        }

    def _extract_character_names(self, text: str) -> List[str]:
        """Extract person names from text"""
        patterns = [
            r"\b([A-Z][\w'-]*(?: [A-Z][\w'-]*)?)\s+(?:said|asked|replied|whispered)\b",
            r"(?i:character|name)\s*:\s*([A-Z][\w'-]*(?: [A-Z][\w'-]*)?)",
        ]

        names = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            names.update(matches)

        return list(names)[:5]

    def _parse_consistency_result(self, response: str) -> Dict:
        """Parse consistency check results"""
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass

        return {
            "consistent": True,
            "confidence": 0.5,
            "issues": [],
            "suggestions": []
        }
