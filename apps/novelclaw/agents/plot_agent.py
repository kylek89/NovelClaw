"""
Plot Construction Agent
"""
from typing import Dict, Optional, List
from agents.base_agent import BaseAgent
from utils.llm_client import LLMClient
from rag.retriever import Retriever
from rag.memory_system import MemorySystem
from rag.static_knowledge_base import StaticKnowledgeBase
from config import Config


class PlotAgent(BaseAgent):
    """Plot Construction Agent: Responsible for designing the main story arc and plot branches"""

    def __init__(
        self,
        config: Config,
        llm_client: LLMClient,
        retriever: Optional[Retriever] = None,
        memory_system: Optional[MemorySystem] = None,
        static_kb: Optional[StaticKnowledgeBase] = None,
    ):
        system_prompt = """You are a professional plot designer.
Responsibilities:
1. Design compelling main plotlines and branches.
2. Ensure logical coherence and callbacks.
3. Create dramatic conflict and climaxes.
4. Keep the plot consistent yet inventive.
Respond in English with detailed, creative plot design."""
        name = "Plot Agent"

        super().__init__(
            name=name,
            role="plot",
            system_prompt=system_prompt,
            config=config,
            llm_client=llm_client,
            retriever=retriever,
            memory_system=memory_system,
            static_kb=static_kb,
        )

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        topic: Optional[str] = None,
        genre: Optional[str] = None,
        style_tags: Optional[List[str]] = None,
    ) -> Dict:
        """Generate plot content"""
        messages = self._build_messages(
            prompt,
            context,
            use_rag=True,
            topic=topic,
            use_memory=True,
            use_static_kb=True,
            genre=genre,
            style_tags=style_tags,
        )

        response = self.llm_client.chat(
            messages=messages,
            temperature=self.config.temperature + 0.1,  # Plot needs more creativity
        )

        self.add_to_history("user", prompt)
        self.add_to_history("assistant", response)

        return {
            "agent": self.name,
            "role": self.role,
            "content": response,
            "type": "plot",
        }
