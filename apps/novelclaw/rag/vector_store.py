"""
Vector database module: Uses ChromaDB to store and retrieve documents
"""
from typing import List, Dict, Optional
import os
from config import Config


class VectorStore:
    """Vector database manager"""

    def __init__(self, config: Config, collection_name: str = "documents", db_path: Optional[str] = None):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb is required for vector storage. "
                "Install it with: pip install chromadb>=0.4.0"
            )
        self.config = config
        self.collection_name = collection_name
        self.db_path = db_path or config.vector_db_path

        # Create ChromaDB client
        os.makedirs(self.db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[Dict]):
        """
        Add documents to the vector database

        Args:
            documents: List of documents, each containing text and embedding
        """
        if not documents:
            return

        ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        embeddings = [doc["embedding"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]

        payload = dict(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        # Prefer upsert to handle repeated imports of the same data by overwriting or skipping, rather than causing errors
        upsert = getattr(self.collection, "upsert", None)
        if callable(upsert):
            upsert(**payload)
        else:
            self.collection.add(**payload)

    def search(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents

        Args:
            query_embedding: Query vector
            top_k: Number of returned documents
            filter_metadata: Metadata filter conditions

        Returns:
            List of similar documents, including text, metadata, and distance
        """
        top_k = top_k or self.config.top_k

        # Chroma requires the top-level 'where' clause to accept only one operator; if multiple fields are involved, combine them using $and
        where = None
        if filter_metadata:
            if "$and" in filter_metadata or "$or" in filter_metadata:
                where = filter_metadata
            elif len(filter_metadata.keys()) > 1:
                where = {"$and": [{k: v} for k, v in filter_metadata.items()]}
            else:
                where = filter_metadata

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )

        # Format the results
        formatted_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None
                })

        return formatted_results

    def delete_collection(self):
        """Delete the collection"""
        self.client.delete_collection(name=self.collection_name)

    def get_collection_size(self) -> int:
        """Get the number of documents in the collection"""
        return self.collection.count()

