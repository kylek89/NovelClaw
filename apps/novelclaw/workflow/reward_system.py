"""
Reinforcement learning reward system.
"""
from typing import Dict, List, Optional
import re
from utils.text_metrics import text_length


class RewardSystem:
    """Reward system: evaluate the quality of the generated text and provide a reward score."""

    def __init__(self, config):
        self.config = config

    def calculate_reward(
        self,
        generated_text: str,
        evaluation_result: Dict,
        consistency_result: Optional[Dict] = None,
        context: Optional[Dict] = None,
        target_length: Optional[int] = None
    ) -> Dict:
        """
        Calculate the reward score.

        Args:
            generated_text: The generated text.
            evaluation_result: The evaluation result.
            context: Contextual information.
            consistency_result: Consistency check result

        Returns:
            Reward score and details.
        """
        # Retrieve the score from the evaluation result (novelty is zeroed out and does not affect the total score).
        novelty_score = 0.0
        coherence_score = evaluation_result.get("coherence_score", 0.5)
        emotional_score = evaluation_result.get("emotional_score", 0.5)
        overall_score = evaluation_result.get("overall_score", 0.5)
        # Consistency (from consistency check).
        consistency_conf = 0.5
        consistency_issues = 0
        if consistency_result:
            consistency_conf = consistency_result.get("overall_confidence", 0.5)
            consistency_issues = len(consistency_result.get("all_issues", []))

        # Calculate the text length reward (encourages reaching the target length, allowing slight overruns).
        measured_length = text_length(generated_text, getattr(self.config, "language", "en"))
        target = max(target_length or 1000, 1)
        length_score = min(1.2, measured_length / target)

        # Calculate the structural integrity reward.
        structure_score = self._calculate_structure_score(generated_text)

        # Consistency score: confidence minus a slight penalty for the number of issues.
        consistency_score = max(0.0, min(1.0, consistency_conf - 0.03 * consistency_issues))

        # Comprehensive reward (excluding novelty, emphasizing coherence, overall quality, consistency, length, and structure).
        reward = (
            coherence_score * 0.30 +
            overall_score * 0.35 +
            consistency_score * 0.20 +
            length_score * 0.10 +
            structure_score * 0.05
        )

        return {
            "total_reward": reward,
            "novelty_score": novelty_score,
            "coherence_score": coherence_score,
            "emotional_score": emotional_score,
            "overall_score": overall_score,
            "length_score": length_score,
            "structure_score": structure_score,
            "consistency_score": consistency_score,
            "consistency_issues": consistency_issues,
            "breakdown": {
                "coherence": coherence_score * 0.30,
                "overall": overall_score * 0.35,
                "consistency": consistency_score * 0.20,
                "length": length_score * 0.10,
                "structure": structure_score * 0.05
            }
        }

    def _calculate_structure_score(self, text: str) -> float:
        """
        Calculate the structural integrity score.

        Args:
            text: Text content

        Returns:
            Structure score (0-1).
        """
        score = 0.5  # Base score.

        # Check for paragraph separators.
        if "\n\n" in text or "\n" in text:
            score += 0.2

        # Check for title or section markers
        if re.search(r'^chapter\s+\d+\b', text, re.I | re.M) or re.search(r'^#{1,3}\s', text, re.MULTILINE):
            score += 0.2

        # Check if text length is reasonable
        if 500 <= len(text) <= 10000:
            score += 0.1

        return min(1.0, score)

    def should_continue(
        self,
        reward: Dict,
        iteration: int,
        target_length: Optional[int] = None,
        current_length: Optional[int] = None
    ) -> bool:
        """
        Determine whether to continue iterating

        Args:
            rReward: reward score
            iteration: Current iteration count

        Returns:
            Whether to continue
        """
        total_reward = reward.get("total_reward", 0)

        # Stop if the reward score is high enough and the length is basically satisfactory
        if total_reward >= self.config.min_quality_score:
            if target_length and current_length:
                if current_length >= target_length * 0.8:
                    return False
            else:
                if current_length is None or current_length >= 800:
                    return False

        # End early if the target length has been reached
        if target_length and current_length and current_length >= target_length:
            return False

        # Stop if the number of iterations is too high
        max_iters = min(self.config.max_iterations, getattr(self.config, 'max_total_iterations', self.config.max_iterations))
        if iteration >= max_iters:
            return False

        return True



