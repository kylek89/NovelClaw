"""
Real-time modification system: Detect and adjust related content to maintain overall consistency
"""
from typing import Dict, List, Optional, Tuple
import json
import re
from utils.llm_client import LLMClient
from rag.memory_system import MemorySystem
from rag.turning_point_tracker import TurningPointTracker
from config import Config


class RealtimeEditor:
    """Real-time modification system: Detect parts that need modification and automatically adjust them"""

    def __init__(
        self,
        config: Config,
        llm_client: LLMClient,
        memory_system: MemorySystem,
        turning_point_tracker: TurningPointTracker
    ):
        self.config = config
        self.llm_client = llm_client
        self.memory_system = memory_system
        self.turning_point_tracker = turning_point_tracker
        self.lang = "en"

    def detect_modification_needs(
        self,
        text: str,
        topic: str,
        consistency_result: Optional[Dict] = None,
        turning_point_notes: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Detect parts that need modification

        Args:
            text: Generated text
            topic: Topic
            consistency_result: Consistency check result

        Returns:
            List of parts that need modification
        """
        modifications = []

        # 1. Based on consistency check results
        if consistency_result:
            issues = consistency_result.get("all_issues", [])
            for issue in issues:
                modifications.append({
                    "type": "consistency_issue",
                    "issue": issue,
                    "priority": "high",
                    "suggestion": self._generate_modification_suggestion(issue, text)
                })

        # 2. Refer to transition prompts/issues, requiring the main text to be revised to align with the outline
        if turning_point_notes:
            for note in turning_point_notes:
                modifications.append({
                    "type": "turning_point_issue",
                    "issue": note,
                    "priority": "high",
                    "suggestion": self._generate_modification_suggestion(note, text)
                })

        return modifications

    def apply_modifications(
        self,
        text: str,
        modifications: List[Dict],
        topic: str
    ) -> str:
        """
        Apply modifications

        Args:
            text: Original text
            modifications: list of modifications
            topic: Topic

        Returns:
            Revised text
        """
        if not modifications:
            return text

        # Sort by priority
        modifications.sort(key=lambda x: {
            "high": 0,
            "medium": 1,
            "low": 2
        }.get(x.get("priority", "low"), 2))

        # Retrieve relevant context
        context = self.memory_system.get_relevant_context(
            text[:200],
            topic,
            include_types=["character", "outline", "plot_point"]
        )

        # Construct modification prompts
        mod_descriptions = []
        for i, mod in enumerate(modifications[:5], 1):  # Process up to 5 items
            mod_descriptions.append(
                f"{i}. [{mod['type']}] {mod.get('issue', mod.get('suggestion', ''))}"
            )

        prompt = f"""Adjust the text based on the modification suggestions:

Original text:
{text}

Relevant context:
{context}

Needed changes:
{chr(10).join(mod_descriptions)}

Please:
1) Keep overall style and structure.
2) Ensure consistency with outline/characters.
3) Smooth turning points; avoid abruptness.
4) Maintain coherence.

Output only the revised full text."""
        system_prompt = "You are a professional fiction editor; apply the requested changes and keep coherence."

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {"role": "user", "content": prompt}
        ]

        try:
            modified_text = self.llm_client.chat(messages, temperature=0.5)
            return modified_text
        except:
            return text

    def update_related_parts(
        self,
        modified_text: str,
        topic: str,
        modification_type: str
    ):
        """
        Update relevant sections (when key content is modified)

        Args:
            modified_text: revised text
            topic: Topic
            modification_type: type of modification
        """
        # If characters are modified, update character information
        if "character" in modification_type.lower():
            # Re-extract character information and update
            characters = self._extract_characters(modified_text)
            for char_name, char_info in characters.items():
                self.memory_system.store_character(
                    char_name, char_info, topic
                )

        # If plot is modified, update plot points
        if "plot" in modification_type.lower() or "turning_point" in modification_type.lower():
            # Re-detect turning points
            turning_points = self.turning_point_tracker.detect_turning_points(
                modified_text, topic
            )
            for tp in turning_points:
                self.turning_point_tracker.record_turning_point(
                    tp.get("type", "key_events"),
                    tp.get("content", {}),
                    topic
                )

    def _generate_modification_suggestion(self, issue: str, text: str) -> str:
        """Generate modification suggestions"""
        # Simple heuristic, can be improved
        issue_type = issue.casefold()
        if "character" in issue_type or "cast" in issue_type:
            return "Check that the character's actions fit their established traits."
        elif "plot" in issue_type or "logic" in issue_type:
            return "Check that the events follow logically from what came before."
        elif "world" in issue_type:
            return "Check the passage against the established world rules."
        else:
            return f"Revise the passage to address this issue:{issue}"

    def _check_if_abrupt(
        self,
        turning_point: Dict,
        text: str,
        topic: str
    ) -> bool:
        """Check if turning points are abrupt"""
        # Use LLM to judge
        prompt = f"""Decide whether this turning point feels abrupt:

Passage:
{text}

Turning point:
{json.dumps(turning_point, ensure_ascii=False)}

Check whether the turn:
1. Lacks setup
2. Happens too suddenly
3. Conflicts with the preceding events

Answer yes or no and briefly explain why."""

        messages = [
            {
                "role": "system",
                "content": "You analyze story structure."
            },
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.llm_client.chat(messages, temperature=0.3)
            return bool(re.match(r"^\s*yes\b", response, re.IGNORECASE))
        except:
            return False

    def _smooth_turning_point(
        self,
        turning_point: Dict,
        text: str,
        topic: str
    ) -> str:
        """Smooth out turning points"""
        return f"Add enough setup for this turning point to feel earned. Turning point:{turning_point.get('content', {})}"

    def _is_character_change_reasonable(
        self,
        change: Dict,
        text: str,
        topic: str
    ) -> bool:
        """Check if character changes are reasonable"""
        # Retrieve character history
        character_name = change.get("content", {}).get("character", "Unknown")
        characters = self.memory_system.get_characters_by_topic(topic)

        # Check if there is sufficient foreshadowing
        # This can be improved with more complex logic
        return True

    def _adjust_character_change(
        self,
        change: Dict,
        text: str,
        topic: str
    ) -> str:
        """Adjust character changes"""
        return f"Give this character change a clear cause and setup. Change:{change.get('content', {})}"

    def _extract_characters(self, text: str) -> Dict[str, str]:
        """Extract character information from text"""
        # Simple extraction method, can be improved
        characters = {}

        # Find character descriptions
        patterns = [
            r"\b([A-Z][\w'-]*(?: [A-Z][\w'-]*)?)\s+(?:is|was|seems|feels)\b",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match not in characters:
                    characters[match] = ""

        return characters

