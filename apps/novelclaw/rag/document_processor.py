"""
Document processing module: document chunking, embedding, etc.
"""
import os
from typing import List, Dict, Optional
import uuid
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    # Use simple text splitting if langchain is unavailable
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=500, chunk_overlap=100, length_function=len):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.length_function = length_function

        def split_text(self, text: str):
            chunks = []
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunks.append(text[start:end])
                start = end - self.chunk_overlap
            return chunks
# Avoid conflicts caused by transformers automatically importing TensorFlow in certain environments
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")


def _embedding_downloads_disabled() -> bool:
    raw = str(os.getenv("DISABLE_EMBEDDING_DOWNLOADS", "1") or "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


if _embedding_downloads_disabled():
    # Make accidental HuggingFace embedding initialization stay offline by default.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np


class DocumentProcessor:
    """Document processor"""

    def __init__(self, config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
        )
        embed_name = (config.embedding_model or "").strip().lower()
        if _embedding_downloads_disabled():
            self.embedding_model = None
        elif embed_name in {"", "none", "noop", "disable", "disabled"}:
            self.embedding_model = None
        else:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(config.embedding_model)
            except Exception:
                self.embedding_model = None

    def split_documents(self, text: str) -> List[str]:
        """
        Split documents into chunks

        Args:
            text: Original text

        Returns:
            List of text chunks
        """
        chunks = self.text_splitter.split_text(text)
        return chunks

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embedding vectors for text

        Args:
            texts: List of texts

        Returns:
            Embedding vector array
        """
        if self.embedding_model is None:
            # return a small dummy vector to avoid heavy model load when embedding is disabled
            return np.zeros((len(texts), 4), dtype=float)
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings

    def process_document(self, text: str, doc_id: Optional[str] = None) -> List[Dict]:
        """
        Process documents: chunk and generate embeddings

        Args:
            text: Original document
            doc_id: Unique document ID (optional, used for deduplication)

        Returns:
            List of dictionaries containing text chunks and embeddings
        """
        # doc_id is used for stable deduplication; it is recommended to pass stable values (e.g., file_path hash / jsonl line number)
        prefix = doc_id or uuid.uuid4().hex
        chunks = self.split_documents(text)
        embeddings = self.get_embeddings(chunks)

        result = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            result.append({
                "id": f"{prefix}_chunk_{i}",
                "text": chunk,
                "embedding": embedding.tolist(),
                "metadata": {
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })

        return result

