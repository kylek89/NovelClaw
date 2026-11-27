"""
Agent base class
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from utils.llm_client import LLMClient
from rag.retriever import Retriever
from rag.memory_system import MemorySystem
from rag.static_knowledge_base import StaticKnowledgeBase
from config import Config


class BaseAgent(ABC):
    """Agent base class"""

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        config: Config,
        llm_client: LLMClient,
        retriever: Optional[Retriever] = None,
        memory_system: Optional[MemorySystem] = None,
        static_kb: Optional[StaticKnowledgeBase] = None
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.config = config
        self.llm_client = llm_client
        self.retriever = retriever
        self.memory_system = memory_system
        self.static_kb = static_kb  # External static knowledge base
        self.conversation_history: List[Dict[str, str]] = []
        self.runtime_use_rag_override: Optional[bool] = None
        self.runtime_use_static_kb_override: Optional[bool] = None

    def _build_messages(
        self,
        prompt: str,
        context: Optional[str] = None,
        use_rag: bool = True,
        topic: Optional[str] = None,
        use_memory: bool = True,
        use_static_kb: bool = True,
        genre: Optional[str] = None,
        style_tags: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Build message list

        Args:
            prompt: user prompt
            context: additional context
            use_rag: whether to use RAG retrieval
            topic: topic (used for memory retrieval)
            use_memory: whether to use the memory system

        Returns:
            message list
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        rag_enabled = bool(use_rag and self.retriever and getattr(self.config, "enable_rag", True))
        static_kb_enabled = bool(use_static_kb and self.static_kb and getattr(self.config, "enable_static_kb", True))
        if self.runtime_use_rag_override is not None:
            rag_enabled = rag_enabled and bool(self.runtime_use_rag_override)
        if self.runtime_use_static_kb_override is not None:
            static_kb_enabled = static_kb_enabled and bool(self.runtime_use_static_kb_override)

        # Add memory system context (priority)
        memory_context = ""
        if use_memory and self.memory_system and topic:
            memory_context = self.memory_system.get_relevant_context(
                query=f"{prompt}\n\n{context or ''}",
                topic=topic,
                include_types=self._get_relevant_memory_types()
            )
            if memory_context:
                messages.append({
                    "role": "system",
                    "content": f"[Important]Historical memory (keep these details consistent):\n{memory_context}\n\nKeep generated content consistent with the history above and avoid contradictions."
                })

        # Add style reference from the external static knowledge base (to reduce hallucinations and adjust style)
        if static_kb_enabled:
            # First provide 'plot reference' (only helpful for plot/retrieval/writer)
            if self.role in {"plot", "retrieval", "writer"}:
                plot_context = self.static_kb.get_plot_context(
                    query=prompt,
                    genre=genre,
                    style_tags=style_tags,
                    top_k=2
                )
                if plot_context:
                    messages.append({"role": "system", "content": plot_context})

            style_context = self.static_kb.get_style_context(
                query=prompt,
                genre=genre,
                style_tags=style_tags,
                top_k=3
            )
            if style_context:
                messages.append({
                    "role": "system",
                    "content": style_context
                })

        # Add retrieved context (dynamic knowledge base)
        if rag_enabled:
            retrieved_context = self.retriever.retrieve_with_context(
                query=prompt,
                context=context,
                top_k=3
            )
            if retrieved_context:
                messages.append({
                    "role": "system",
                    "content": f"Relevant background:\n{retrieved_context}"
                })

        # Add conversation history
        messages.extend(self.conversation_history[-5:])  # Keep only the most recent 5 turns of conversation

        # Add current prompt
        if context and not rag_enabled:
            messages.append({
                "role": "user",
                "content": f"{context}\n\n{prompt}"
            })
        else:
            messages.append({"role": "user", "content": prompt})

        return messages

    def _get_relevant_memory_types(self) -> List[str]:
        """Get relevant memory types (subclasses can override)"""
        return ["character", "outline", "plot_point", "world_setting", "fact_card"]

    @abstractmethod
    def generate(self, prompt: str, context: Optional[str] = None) -> Dict:
        """
        Generate content (subclasses must implement)

        Args:
            prompt: prompt
            context: context

        Returns:
            generation result dictionary
        """
        pass

    def reset(self):
        """Reset conversation history"""
        self.conversation_history = []

    def add_to_history(self, role: str, content: str):
        """Add to conversation history"""
        self.conversation_history.append({"role": role, "content": content})
