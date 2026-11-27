"""
Configuration Management Module
"""
import os
from dotenv import load_dotenv, dotenv_values
from typing import Optional
import os.path

from capability_registry import enabled_capability_slugs_from_env


def _env_flag(name: str, default: str = "0") -> bool:
    val = os.getenv(name, default)
    return str(val).strip() in {"1", "true", "True", "yes", "YES", "y", "Y", "on", "ON"}

# Always load .env from the current file's directory; fall back to the project root or current working directory if missing
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DOTENV_CANDIDATES = [
    os.path.join(_BASE_DIR, ".env"),
    os.path.join(os.path.dirname(_BASE_DIR), ".env"),
    os.path.join(os.getcwd(), ".env"),
]
loaded = False
for _p in _DOTENV_CANDIDATES:
    if os.path.exists(_p):
        load_dotenv(dotenv_path=_p, encoding="utf-8-sig", override=False)
        loaded = True
        break
# Fallback: If the key is still not loaded, manually read from candidate paths and write to environment variables
if not os.getenv("DEEPSEEK_API_KEY"):
    for _p in _DOTENV_CANDIDATES:
        if os.path.exists(_p):
            vals = dotenv_values(_p, encoding="utf-8-sig")
            for k, v in vals.items():
                if k and v and k not in os.environ:
                    os.environ[k] = v
            break
# Handle keys that may have a BOM prefix
for k in list(os.environ.keys()):
    if k.startswith("\ufeff"):
        clean_k = k.lstrip("\ufeff")
        if clean_k and clean_k not in os.environ:
            os.environ[clean_k] = os.environ[k]


class Config:
    """System configuration class"""

    def __init__(self, require_api_key: bool = True):
        # Base language/provider configuration
        # Writing and interface language.
        self.language: str = "en"
        # Workflow mode: chaptered (chapter-based) / unfixed (free-form whole text, recommended for English)
        self.workflow_mode: str = os.getenv("WORKFLOW_MODE", "chaptered").lower()
        # Execution mode: workflow (fixed process) / claw (OpenClaw-style dynamic management loop)
        self.execution_mode: str = os.getenv("EXECUTION_MODE", "claw").lower()
        self.claw_max_steps: int = int(os.getenv("CLAW_MAX_STEPS", "20"))  # safety limit; LLM exits via finalize tool
        # English sub-modes: novel (long-form segmented writing), premise_story (single piece based on premise), extensible
        self.en_mode: str = os.getenv("EN_MODE", "novel").lower()
        # Whether to completely skip the outline (available for free English generation)
        self.outline_free: bool = os.getenv("OUTLINE_FREE", "0") in {"1", "true", "True", "yes", "YES"}
        # English segmentation control (prompt length for unfixed/novel segmentation)
        self.en_segment_words: int = int(os.getenv("EN_SEGMENT_WORDS", "1200"))
        self.en_segments: int = int(os.getenv("EN_SEGMENTS", "3"))
        self.llm_provider: str = os.getenv("LLM_PROVIDER", "deepseek").lower()

        # API Key and Base URL (prioritize using generic variables, compatible with any provider)
        generic_api_key = os.getenv("LLM_API_KEY", "").strip()
        generic_base_url = os.getenv("LLM_BASE_URL", "").strip()
        if self.llm_provider == "deepseek":
            fallback_key = (
                os.getenv("DEEPSEEK_API_KEY")
                or os.getenv("CODEX_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or ""
            )
            fallback_base = (
                os.getenv("DEEPSEEK_BASE_URL")
                or "https://api.deepseek.com/v1"
            )
        elif self.llm_provider in {"codex", "packycode"}:
            fallback_key = (
                os.getenv("CODEX_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("DEEPSEEK_API_KEY", "")
            )
            fallback_base = (
                os.getenv("CODEX_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or "https://codex-api.packycode.com/v1"
            )
        else:
            fallback_key = (
                os.getenv("OPENAI_API_KEY")
                or os.getenv("CODEX_API_KEY")
                or os.getenv("DEEPSEEK_API_KEY", "")
            )
            fallback_base = (
                os.getenv("OPENAI_BASE_URL")
                or os.getenv("CODEX_BASE_URL")
                or os.getenv("DEEPSEEK_BASE_URL")
                or "https://api.openai.com/v1"
            )

        self.api_key: str = generic_api_key or fallback_key
        self.api_base_url: str = generic_base_url or fallback_base
        # Fields maintained for backward compatibility
        self.deepseek_api_key: str = self.api_key if self.llm_provider == "deepseek" else os.getenv("DEEPSEEK_API_KEY", "")
        self.deepseek_base_url: str = self.api_base_url or "https://api.deepseek.com/v1"

        # RAG configuration
        # Used as the 'base directory'; specific collections are recommended to be stored in subdirectories to prevent global impact from a single database corruption
        self.vector_db_path: str = (os.getenv("VECTOR_DB_PATH", "").strip() or "./vector_db")

        # Individual vector database subpaths (isolated to reduce collateral impact when Chroma index corruption occurs)
        self.memory_vector_db_path: str = os.getenv(
            "MEMORY_VECTOR_DB_PATH",
            os.path.join(self.vector_db_path, "memory")
        )
        self.knowledge_vector_db_path: str = os.getenv(
            "KNOWLEDGE_VECTOR_DB_PATH",
            os.path.join(self.vector_db_path, "knowledge")
        )
        # Static knowledge base uses an independent vector database path to avoid global impact from index corruption when sharing with other collections
        self.static_vector_db_path: str = os.getenv(
            "STATIC_VECTOR_DB_PATH",
            os.path.join(self.vector_db_path, "static_kb")
        )
        self.embedding_model: str = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
        self.top_k: int = int(os.getenv("TOP_K", "5"))  # Number of documents returned by retrieval
        # Static knowledge base: maximum number of chunks written per original text entry (reduces disk usage and index corruption probability)
        self.static_max_chunks_per_doc: int = int(os.getenv("STATIC_MAX_CHUNKS_PER_DOC", "12"))

        # Agent configuration
        self.max_iterations: int = int(os.getenv("MAX_ITERATIONS", "60"))
        self.temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
        # Maximum output tokens, default 8000 (can be overridden via environment variables, but will be truncated again on the client side if it exceeds the interface limit)
        self.max_tokens: int = int(os.getenv("MAX_TOKENS", "8000"))
        # LLM call timeout/retry controls. Set LLM_TIMEOUT_SECONDS=0 to disable request timeout.
        self.llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "0"))
        self.model_reasoning_effort: str = os.getenv("MODEL_REASONING_EFFORT", "").strip()
        self.llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "1"))
        default_model = "deepseek-chat" if self.llm_provider == "deepseek" else "gpt-5.2"
        self.model_name: str = os.getenv("MODEL_NAME", default_model)
        self.model_kind = os.getenv("NOVELCLAW_MODEL_KIND", "custom")
        self.wire_api: str = os.getenv("WIRE_API", "chat").lower()
        self.enabled_capabilities = enabled_capability_slugs_from_env(os.getenv("ENABLED_CAPABILITIES", ""))

        # Workflow configuration
        self.min_quality_score: float = 0.7  # Minimum quality score
        self.max_retry_rounds: int = 3  # Maximum retry rounds

        # Context control (prevents prompt explosion during excessively long generation)
        # Default is moderately relaxed to allow prompts to carry more outline/summary information
        self.context_max_chars: int = int(os.getenv("CONTEXT_MAX_CHARS", "12000"))
        self.recent_context_items: int = int(os.getenv("RECENT_CONTEXT_ITEMS", "4"))
        # per-run memory isolation
        self.memory_reset_each_run: bool = os.getenv("MEMORY_RESET_EACH_RUN", "1") in {"1", "true", "True", "yes", "YES"}
        # store full chapter text in vector memory
        self.store_full_text_in_memory: bool = os.getenv("STORE_FULL_TEXT_IN_MEMORY", "0") in {"1", "true", "True", "yes", "YES"}
        # Force total length to reach target (default is not forced; the target is a soft constraint)
        self.force_target_length: bool = os.getenv("FORCE_TARGET_LENGTH", "0") in {"1", "true", "True", "yes", "YES"}
        # minimum chapter length
        # Default minimum of 3000 words per chapter (can be overridden via environment variables)
        self.min_chapter_chars: int = int(os.getenv("MIN_CHAPTER_CHARS", "3000"))
        # Hard cap per chapter (0 disables the hard cap)
        self.max_chapter_chars: int = int(os.getenv("MAX_CHAPTER_CHARS", "0"))
        # When no hard cap is set, use this ratio of the target chapter length as the limit
        self.chapter_max_ratio: float = float(os.getenv("CHAPTER_MAX_RATIO", "1.15"))
        # chapter target ratio
        self.chapter_target_ratio: float = float(os.getenv("CHAPTER_TARGET_RATIO", "1.0"))
        # max subrounds per chapter
        self.max_chapter_subrounds: int = int(os.getenv("MAX_CHAPTER_SUBROUNDS", "3"))
        # max total iterations
        self.max_total_iterations: int = int(os.getenv("MAX_TOTAL_ITERATIONS", "120"))
        # analysis interval (chapters)
        self.analysis_interval: int = int(os.getenv("ANALYSIS_INTERVAL", "5"))
        # run output
        self.runs_dir: str = os.getenv("RUNS_DIR", "runs")
        self.run_id: str = os.getenv("RUN_ID", "")
        # Number of chapters to write per outer iteration (can be increased for efficiency; default is 1 chapter/round)
        self.chapters_per_iter: int = int(os.getenv("CHAPTERS_PER_ITER", "1"))

        # Fast generation mode: uses only the core writing Agent to reduce the number of turns
        self.fast_mode: bool = os.getenv("FAST_MODE", "0") in {"1", "true", "True", "yes", "YES"}
        # Chapter cycle: run a full multi-Agent process every N chapters/rounds; otherwise, only writing is performed (accelerates long-form generation)
        self.full_cycle_interval: int = int(os.getenv("FULL_CYCLE_INTERVAL", "0"))  # 0 means always run the full process
        # Whether the static library is used only in the first round (reduces retrieval overhead in subsequent rounds)
        self.static_kb_first_only: bool = os.getenv("STATIC_KB_FIRST_ONLY", "1") in {"1", "true", "True", "yes", "YES"}
        # RAG controls (for local memory-only mode and server deployment flexibility)
        # Default to dynamic memory only: disables static RAG and static knowledge base
        self.memory_only_mode: bool = _env_flag("MEMORY_ONLY_MODE", "1")
        self.enable_rag: bool = _env_flag("ENABLE_RAG", "0") and not _env_flag("DISABLE_RAG", "0")
        self.enable_static_kb: bool = _env_flag("ENABLE_STATIC_KB", "0") and not _env_flag("DISABLE_STATIC_KB", "0")
        if self.memory_only_mode:
            self.enable_rag = False
            self.enable_static_kb = False
        # Static KB is part of RAG; if RAG is off, static KB must be off as well.
        if not self.enable_rag:
            self.enable_static_kb = False
            self.embedding_model = "none"
        # Evaluation/scoring interval (granularity: iteration rounds), reducing the overhead of scoring every round
        self.eval_interval: int = int(os.getenv("EVAL_INTERVAL", "4"))
        # Whether to enable the evaluation Agent (disabled by default to avoid impacting the latency of the main writing workflow)
        self.enable_evaluator: bool = _env_flag("ENABLE_EVALUATOR", "0") and not _env_flag("DISABLE_EVALUATOR", "0")
        # Whether to print logs for each LLM call's duration (for terminal progress awareness)
        self.llm_call_log: bool = _env_flag("LLM_CALL_LOG", "1")
        # Whether to enable turning point detection (enabled by default; set TURNING_POINT_ENABLED=0 to disable)
        self.turning_point_enabled: bool = os.getenv("TURNING_POINT_ENABLED", "1") in {"1", "true", "True", "yes", "YES"}
        # Chapter planning control (now automatically estimated; this default value is retained but not dependent on the environment variable)
        self.target_chapters_min: int = 5
        self.target_chapters_max: int = 30
        self.per_chapter_target_hint: int = 3300

        # Configuration validation
        if require_api_key and not self.api_key:
            raise ValueError("Set LLM_API_KEY, or set CODEX_API_KEY/OPENAI_API_KEY/DEEPSEEK_API_KEY")

    def __repr__(self):
        return f"Config(model={self.model_name}, db_path={self.vector_db_path})"
