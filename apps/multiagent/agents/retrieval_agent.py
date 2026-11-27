"""
Data Retrieval Agent
"""
from typing import Dict, Optional, List
from agents.base_agent import BaseAgent
from utils.llm_client import LLMClient
from rag.retriever import Retriever
from rag.memory_system import MemorySystem
from rag.static_knowledge_base import StaticKnowledgeBase
from config import Config


class RetrievalAgent(BaseAgent):
    """Data Retrieval Agent: Retrieve relevant historical, cultural, and technical background information"""

    def __init__(
        self,
        config: Config,
        llm_client: LLMClient,
        retriever: Retriever,
        memory_system: Optional[MemorySystem] = None,
        static_kb: Optional[StaticKnowledgeBase] = None,
    ):
        system_prompt = """You are an information retrieval and synthesis expert.
Responsibilities:
1. Retrieve relevant historical/cultural/technical background based on a query.
2. Integrate retrieved information into accurate, useful background knowledge.
3. Ensure accuracy and relevance; support other agents with concise summaries.
Respond in English with clear, well-structured integration."""
        name = "Retrieval Agent"

        super().__init__(
            name=name,
            role="retrieval",
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
        """Retrieve and integrate relevant information"""
        if not self.retriever:
            return {
                "agent": self.name,
                "role": self.role,
                "content": "RAG is disabled for this run.",
                "retrieved_docs": [],
                "type": "retrieval",
            }

        retrieved_docs = self.retriever.retrieve(prompt, top_k=5)

        if not retrieved_docs:
            return {
                "agent": self.name,
                "role": self.role,
                "content": "No relevant information found.",
                "retrieved_docs": [],
                "type": "retrieval",
            }

        retrieved_text = "\n\n".join(
            [f"[Source {i+1}]: {doc['text']}" for i, doc in enumerate(retrieved_docs)]
        )

        integration_prompt = f"""Based on the retrieved information below, synthesize and summarize what is relevant.

Retrieved information:
{retrieved_text}

Query: {prompt}

Please provide a clear, structured summary in English."""

        messages = self._build_messages(integration_prompt, context, use_rag=False)

        response = self.llm_client.chat(
            messages=messages,
            temperature=0.5,
        )

        self.add_to_history("user", prompt)
        self.add_to_history("assistant", response)

        return {
            "agent": self.name,
            "role": self.role,
            "content": response,
            "retrieved_docs": retrieved_docs,
            "type": "retrieval",
        }

    def add_knowledge(self, text: str, metadata: Optional[Dict] = None):
        """Add knowledge to the knowledge base"""
        if self.retriever:
            self.retriever.add_knowledge(text, metadata)
