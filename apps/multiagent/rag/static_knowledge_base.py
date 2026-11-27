"""
External static knowledge base RAG: Stores large amounts of novel/creative text data
Used for style reference and reducing hallucinations
"""
from typing import Dict, List, Optional
from rag.retriever import Retriever
from config import Config
import os
import json


class StaticKnowledgeBase:
    """External static knowledge base: Stores large amounts of novel and creative text data"""

    def __init__(self, config: Config):
        from rag.vector_store import VectorStore
        from rag.document_processor import DocumentProcessor
        self.config = config

        # Create a specialized external knowledge base vector store
        self.knowledge_store = VectorStore(
            config,
            collection_name="static_knowledge_base",
            db_path=getattr(config, "static_vector_db_path", None)
        )
        self.document_processor = DocumentProcessor(config)

        # Knowledge base index file
        self.kb_index_path = os.path.join(
            getattr(config, "static_vector_db_path", config.vector_db_path),
            "static_kb_index.json"
        )
        self.kb_index = self._load_kb_index()

    def _load_kb_index(self) -> Dict:
        """Load knowledge base index"""
        if os.path.exists(self.kb_index_path):
            try:
                with open(self.kb_index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {
                    "novels": [],
                    "creative_texts": [],
                    "styles": [],
                    "plot_references": [],
                    "total_documents": 0
                }
        return {
            "novels": [],
            "creative_texts": [],
            "styles": [],
            "plot_references": [],
            "total_documents": 0
        }

    def _save_kb_index(self):
        """Save knowledge base index"""
        os.makedirs(os.path.dirname(self.kb_index_path), exist_ok=True)
        with open(self.kb_index_path, "w", encoding="utf-8") as f:
            json.dump(self.kb_index, f, ensure_ascii=False, indent=2)

    def add_novel(
        self,
        novel_text: str,
        title: str,
        author: Optional[str] = None,
        genre: Optional[str] = None,
        style_tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None,
    ):
        """
        Add novel to knowledge base

        Args:
            novel_text: Novel text
            title: Novel title
            author: Author
            genre: Genre (e.g., sci-fi, fantasy)
            style_tags: Style tags (e.g., detailed, epic, suspenseful)
            metadata: Additional metadata
        """
        # Process document
        processed_docs = self.document_processor.process_document(novel_text, doc_id=doc_id)
        max_chunks = getattr(self.config, "static_max_chunks_per_doc", 0) or 0
        if max_chunks > 0:
            processed_docs = processed_docs[:max_chunks]

        # Add metadata
        base_meta = self._sanitize_metadata(metadata)
        for doc in processed_docs:
            doc["metadata"].update({
                "type": "novel",
                "title": title,
                "author": author or "unknown",
                "genre": genre or "unknown",
                "style_tags": ", ".join(style_tags) if style_tags else "",
                **base_meta
            })

        self.knowledge_store.add_documents(processed_docs)

        # Update index
        self.kb_index["novels"].append({
            "title": title,
            "author": author,
            "genre": genre,
            "style_tags": style_tags or [],
            "chunks": len(processed_docs)
        })
        self.kb_index["total_documents"] += len(processed_docs)
        self._save_kb_index()

    def _sanitize_metadata(self, meta: Optional[Dict]) -> Dict:
        """Ensure metadata values are scalar types supported by Chroma"""
        safe = {}
        if not meta:
            return safe
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                safe[k] = v
            elif isinstance(v, list):
                try:
                    safe[k] = ", ".join(str(x) for x in v)[:500]
                except Exception:
                    safe[k] = str(v)[:500]
            elif isinstance(v, dict):
                safe[k] = str(v)[:500]
            else:
                try:
                    safe[k] = str(v)[:500]
                except Exception:
                    pass
        return safe

    def add_creative_text(
        self,
        text: str,
        title: str,
        text_type: str = "creative",
        style_tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None,
    ):
        """
        Add creative text to knowledge base

        Args:
            text: Text content
            title: Title
            text_type: Text type (e.g., creative, academic)
            style_tags: Style tags
            metadata: Additional metadata
        """
        processed_docs = self.document_processor.process_document(text, doc_id=doc_id)
        max_chunks = getattr(self.config, "static_max_chunks_per_doc", 0) or 0
        if max_chunks > 0:
            processed_docs = processed_docs[:max_chunks]

        for doc in processed_docs:
            doc["metadata"].update({
                "type": "creative_text",
                "title": title,
                "text_type": text_type,
                "style_tags": ", ".join(style_tags) if style_tags else "",
                **self._sanitize_metadata(metadata)
            })

        self.knowledge_store.add_documents(processed_docs)

        self.kb_index["creative_texts"].append({
            "title": title,
            "text_type": text_type,
            "style_tags": style_tags or [],
            "chunks": len(processed_docs)
        })
        self.kb_index["total_documents"] += len(processed_docs)
        self._save_kb_index()

    def add_plot_reference(
        self,
        plot_text: str,
        title: str,
        genre: Optional[str] = None,
        style_tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None,
    ):
        """
        Add 'plot reference' to knowledge base (retrieved separately from style reference).
        """
        processed_docs = self.document_processor.process_document(plot_text, doc_id=doc_id)
        max_chunks = getattr(self.config, "static_max_chunks_per_doc", 0) or 0
        if max_chunks > 0:
            processed_docs = processed_docs[:max_chunks]

        base_meta = self._sanitize_metadata(metadata)
        for doc in processed_docs:
            doc["metadata"].update({
                "type": "plot_reference",
                "title": title,
                "genre": genre or "unknown",
                "style_tags": ", ".join(style_tags) if style_tags else "",
                **base_meta,
            })

        self.knowledge_store.add_documents(processed_docs)
        self.kb_index.setdefault("plot_references", [])
        self.kb_index["plot_references"].append({
            "title": title,
            "genre": genre,
            "style_tags": style_tags or [],
            "chunks": len(processed_docs),
        })
        self.kb_index["total_documents"] += len(processed_docs)
        self._save_kb_index()

    def load_from_directory(self, directory_path: str, file_pattern: str = "*.txt"):
        """
        Batch load text files from directory

        Args:
            directory_path: Directory path
            file_pattern: File matching pattern
        """
        import glob

        files = glob.glob(os.path.join(directory_path, file_pattern))

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                title = os.path.basename(file_path).replace(".txt", "")
                self.add_creative_text(text, title)
                print(f"Loaded: {title}")
            except Exception as e:
                print(f"Could not load file {file_path}: {e}")

    def retrieve_style_reference(
        self,
        query: str,
        genre: Optional[str] = None,
        style_tags: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve style reference text

        Args:
            query: Query text
            genre: Genre filter
            style_tags: Style tag filtering
            top_k: Number of results to return

        Returns:
            List of relevant texts
        """
        # Build filter conditions (style_tags are stored as strings to avoid using $in in the WHERE clause)
        filter_metadata = {"type": "novel"}
        if genre:
            filter_metadata["genre"] = genre

        # Retrieve the query vector
        query_embedding = self.document_processor.get_embeddings([query])[0]

        # Retrieve
        try:
            results = self.knowledge_store.search(
                query_embedding.tolist(),
                top_k=max(top_k, 10),
                filter_metadata=filter_metadata if filter_metadata else None
            )
            if style_tags:
                wanted = set(style_tags)
                filtered = []
                for r in results:
                    tags_str = (r.get("metadata", {}) or {}).get("style_tags", "") or ""
                    if any(t in tags_str for t in wanted):
                        filtered.append(r)
                return filtered[:top_k] if filtered else results[:top_k]
            return results[:top_k]
        except Exception as e:
            # Prevent full pipeline failure due to static library index corruption; fixable by rebuilding the vector library
            print(f"[StaticKB] ⚠️ Skipping style references after a retrieval error: {e}")
            return []

    def retrieve_plot_reference(
        self,
        query: str,
        genre: Optional[str] = None,
        style_tags: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Retrieve 'plot references' (typically extracted or summarized plot rhythms and tropes from the corpus).
        """
        filter_metadata = {"type": "plot_reference"}
        if genre:
            filter_metadata["genre"] = genre

        query_embedding = self.document_processor.get_embeddings([query])[0]
        try:
            results = self.knowledge_store.search(
                query_embedding.tolist(),
                top_k=max(top_k, 10),
                filter_metadata=filter_metadata,
            )
            if style_tags:
                wanted = set(style_tags)
                filtered = []
                for r in results:
                    tags_str = (r.get("metadata", {}) or {}).get("style_tags", "") or ""
                    if any(t in tags_str for t in wanted):
                        filtered.append(r)
                return filtered[:top_k] if filtered else results[:top_k]
            return results[:top_k]
        except Exception as e:
            print(f"[StaticKB] ?? Skipping plot references after a retrieval error: {e}")
            return []

    def get_style_context(
        self,
        query: str,
        genre: Optional[str] = None,
        style_tags: Optional[List[str]] = None,
        top_k: int = 5
    ) -> str:
        """
        Retrieve style context (used as reference during generation)

        Args:
            query: Query text
            genre: Genre
            style_tags: Style tags
            top_k: Number of results to return

        Returns:
            Combined style context
        """
        results = self.retrieve_style_reference(query, genre, style_tags, top_k)

        if not results:
            return ""

        context_parts = ["=== Style references (external knowledge base)==="]

        for i, result in enumerate(results, 1):
            metadata = result.get("metadata", {})
            title = metadata.get("title", "Unknown")
            style_tags_val = metadata.get("style_tags", "")
            if isinstance(style_tags_val, str):
                style_tags_show = style_tags_val
            else:
                style_tags_show = ", ".join(style_tags_val) if style_tags_val else ""
            snippet = result.get("text", "")
            # Limit the length of each reference to avoid overflowing the context window
            if len(snippet) > 900:
                snippet = snippet[:900] + "…"

            context_parts.append(
                f"\n[Reference {i}] {title}"
                f"{' | Style: ' + style_tags_show if style_tags_show else ''}\n"
                f"{snippet}"
            )

        context_parts.append(
            "\nUse these excerpts to study style, expression, and narrative technique. Use original wording."
        )

        return "\n".join(context_parts)

    def get_plot_context(
        self,
        query: str,
        genre: Optional[str] = None,
        style_tags: Optional[List[str]] = None,
        top_k: int = 2,
    ) -> str:
        """
        Retrieve 'plot reference context' (used by plot/writer to reference rhythm and tropes).
        """
        results = self.retrieve_plot_reference(query, genre, style_tags, top_k=top_k)
        if not results:
            return ""

        parts = ["=== Plot references (external excerpts and summaries)==="]
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {}) or {}
            title = meta.get("title", "Unknown")
            snippet = r.get("text", "")
            if len(snippet) > 700:
                snippet = snippet[:700] + "…"
            parts.append(f"\n[Reference {i}] {title}\n{snippet}")

        parts.append("\nUse the pacing, escalation, and transitions as references. Use original names, settings, and wording.")
        return "\n".join(parts)

    def get_statistics(self) -> Dict:
        """Retrieve knowledge base statistics"""
        return {
            "total_novels": len(self.kb_index["novels"]),
            "total_creative_texts": len(self.kb_index["creative_texts"]),
            "total_plot_references": len(self.kb_index.get("plot_references", [])),
            "total_documents": self.kb_index["total_documents"],
            "collection_size": self.knowledge_store.get_collection_size()
        }

