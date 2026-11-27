"""
Execute the plan scheduling module (Adaptive Organizer).
"""
from typing import Dict, List, Optional
from utils.llm_client import LLMClient
from config import Config


class AdaptiveOrganizer:
    """Execution Plan Scheduling Module: Formulate execution plans based on task requirements"""

    def __init__(self, config: Config, llm_client: LLMClient):
        self.config = config
        self.llm_client = llm_client

    def create_execution_plan(
        self,
        task_analysis: Dict,
        available_agents: List[str]
    ) -> Dict:
        """
        Create execution plan

        Args:
            task_analysis: Task analysis results
            available_agents: List of available Agents

        Returns:
            Execution plan
        """
        complexity = task_analysis.get("complexity", "medium")
        recommended_agents = task_analysis.get("recommended_agents", [])
        rag_required = task_analysis.get("rag_required", True)
        estimated_iterations = task_analysis.get("estimated_iterations", 5)

        # Filter out evaluation Agents, keeping only those capable of generation/retrieval
        agent_sequence = [
            agent for agent in recommended_agents
            if agent in available_agents and agent != "evaluator"
        ]

        # Determine execution strategy based on complexity
        if complexity == "simple":
            strategy = "direct_generation"
            use_rag = rag_required
        elif complexity == "complex":
            strategy = "collaborative_generation"
            use_rag = True
        else:  # medium
            strategy = "hybrid_generation"
            use_rag = rag_required

        # Ensure retrieval steps are included when RAG is required
        if use_rag and "retrieval" not in agent_sequence and "retrieval" in available_agents:
            agent_sequence.insert(0, "retrieval")

        # Prioritize having the Writing Agent conclude the generation at the end
        if "writer" in available_agents and "writer" not in agent_sequence:
            agent_sequence.append("writer")

        # Automatically build chapter plan (total chapters and target length per chapter)
        chapter_plan = self._build_chapter_plan(task_analysis)
        # Use chapter count to guide iteration rounds (if more reasonable than analysis results)
        if chapter_plan.get("total_chapters"):
            estimated_iterations = max(estimated_iterations, chapter_plan["total_chapters"])

        execution_plan = {
            "strategy": strategy,
            "agent_sequence": agent_sequence,
            "use_rag": use_rag,
            "estimated_iterations": estimated_iterations,
            "chapter_plan": chapter_plan,
            "workflow_steps": self._generate_workflow_steps(
                strategy, agent_sequence, use_rag
            )
        }

        return execution_plan

    def _generate_workflow_steps(
        self,
        strategy: str,
        agent_sequence: List[str],
        use_rag: bool
    ) -> List[Dict]:
        """
        Generate workflow steps

        Args:
            strategy: Execution strategy
            agent_sequence: Agent sequence
            use_rag: Whether to use RAG

        Returns:
            Workflow step list
        """
        steps = []

        # Initial retrieval step (if needed)
        if use_rag:
            steps.append({
                "step": 1,
                "action": "retrieval",
                "agent": "retrieval",
                "description": "Retrieve relevant background"
            })

        # Agent execution step
        step_num = len(steps) + 1
        for agent in agent_sequence:
            steps.append({
                "step": step_num,
                "action": "generate",
                "agent": agent,
                "description": f"{agent} generates content"
            })
            step_num += 1

        # Evaluation step (disabled by default; enable via config.enable_evaluator)
        if getattr(self.config, "enable_evaluator", False):
            steps.append({
                "step": step_num,
                "action": "evaluate",
                "agent": "evaluator",
                "description": "Evaluate the generated content"
            })

        return steps

    def adjust_plan(
        self,
        current_plan: Dict,
        feedback: Dict,
        iteration: int
    ) -> Dict:
        """
        Adjust execution plan based on feedback

        Args:
            current_plan: Current execution plan
            feedback: Feedback information (including evaluation results, etc.)
            iteration: Current iteration count

        Returns:
            Adjusted execution plan
        """
        adjusted_plan = current_plan.copy()

        # If the quality score is low, increase iterations or adjust the strategy.
        evaluation = feedback.get("evaluation", {})
        overall_score = evaluation.get("overall_score", 0.5)

        if overall_score < self.config.min_quality_score:
            # Increase the number of iterations.
            adjusted_plan["estimated_iterations"] = min(
                adjusted_plan["estimated_iterations"] + 1,
                self.config.max_iterations
            )

            # If the score is very low, switch to collaborative mode.
            if overall_score < 0.5 and adjusted_plan["strategy"] != "collaborative_generation":
                adjusted_plan["strategy"] = "collaborative_generation"
                adjusted_plan["use_rag"] = True

        # Adjust the Agent sequence based on suggestions.
        suggestions = evaluation.get("suggestions", [])
        if suggestions:
            # If suggestions involve specific aspects, ensure the relevant Agents are in the sequence.
            if any("plot" in s.lower() for s in suggestions):
                if "plot" not in adjusted_plan["agent_sequence"]:
                    adjusted_plan["agent_sequence"].insert(0, "plot")
            if any("character" in s.lower() for s in suggestions):
                if "character" not in adjusted_plan["agent_sequence"]:
                    adjusted_plan["agent_sequence"].insert(1, "character")

        return adjusted_plan

    def _build_chapter_plan(self, task_analysis: Dict) -> Dict:
        """
        Infer the number of chapters and target length per chapter based on the target length/complexity (adaptive, no user configuration required).
        """
        target_length = task_analysis.get("target_length", 20000) or 20000
        complexity = task_analysis.get("complexity", "medium")

        # Automatic estimation: soft target of 3200-3800 per chapter, deriving the number of chapters from the total length.
        base_per_chapter = 3500
        if complexity == "simple":
            base_per_chapter = 3700
        elif complexity == "complex":
            base_per_chapter = 3300

        base_per_chapter = max(self.config.min_chapter_chars, min(3800, base_per_chapter))
        total_chapters = max(4, min(80, round(target_length / base_per_chapter)))
        per_chapter_target = max(self.config.min_chapter_chars, min(4000, round(target_length / max(total_chapters, 1))))

        return {
            "total_chapters": total_chapters,
            "per_chapter_target": per_chapter_target
        }
