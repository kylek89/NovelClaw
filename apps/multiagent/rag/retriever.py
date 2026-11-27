"""
Retriever module: integrates vector database and document processing
"""
from typing import List, Dict, Optional
from config import Config


class Retriever:
    """RAG Retriever"""

    def __init__(self, config: Config, collection_name: str = "documents"):
        from rag.vector_store import VectorStore
        from rag.document_processor import DocumentProcessor
        self.config = config
        db_path = getattr(config, "knowledge_vector_db_path", None) or config.vector_db_path
        self.vector_store = VectorStore(config, collection_name, db_path=db_path)
        self.document_processor = DocumentProcessor(config)

    def add_knowledge(self, text: str, metadata: Optional[Dict] = None):
        """
        Add knowledge to the knowledge base

        Args:
            text: Knowledge text
            metadata: Metadata
        """
        processed_docs = self.document_processor.process_document(text)

        # Add metadata
        if metadata:
            for doc in processed_docs:
                doc["metadata"].update(metadata)

        self.vector_store.add_documents(processed_docs)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents

        Args:
            query: Query text
            top_k: Number of returned documents
            filter_metadata: Metadata filter conditions

        Returns:
            List of relevant documents
        """
        # Get embedding vectors for the query
        query_embedding = self.document_processor.get_embeddings([query])[0]

        # Search in the vector database
        results = self.vector_store.search(
            query_embedding.tolist(),
            top_k=top_k,
            filter_metadata=filter_metadata
        )

        return results

    def retrieve_with_context(
        self,
        query: str,
        context: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> str:
        """
        Retrieve and combine context

        Args:
            query: Query text
            context: additional context
            top_k: Number of returned documents

        Returns:
            Combined context text
        """
        retrieved_docs = self.retrieve(query, top_k=top_k)

        # Combine retrieved documents
        context_parts = []
        if context:
            context_parts.append(f"Additional context:\n{context}\n")

        if retrieved_docs:
            context_parts.append("Retrieved information:\n")
            for i, doc in enumerate(retrieved_docs, 1):
                context_parts.append(f"[{i}] {doc['text']}\n")

        return "\n".join(context_parts)

    def clear_knowledge_base(self):
        from rag.vector_store import VectorStore
        """Clear the knowledge base"""
        self.vector_store.delete_collection()
        db_path = getattr(self.config, "knowledge_vector_db_path", None) or self.config.vector_db_path
        self.vector_store = VectorStore(self.config, self.vector_store.collection_name, db_path=db_path)

