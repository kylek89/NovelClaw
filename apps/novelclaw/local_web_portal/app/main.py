from __future__ import annotations

import json
import os
import re
import secrets
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import quote_plus, urlsplit, urlunsplit

from fastapi import Depends, FastAPI, Form, Header, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from config import Config
from rag.memory_system import MemorySystem
from utils.llm_client import LLMClient
from utils.file_io import atomic_write_text, resolve_run_directory, tail_text, save_text_version, ContentConflictError
from utils.artifacts import chapter_outputs, latest_chapter_files
from utils.memory_store import MemoryIndex, load_memory_index, save_memory_index

from agents.idea_copilot_agent import (
    IdeaCopilotAgent,
    append_assistant_turn,
    append_user_reply,
    build_generation_idea,
    dump_state,
    latest_assistant_turn,
    load_state,
    merge_generation_preferences,
    normalize_generation_preferences,
    requested_questions,
)
from . import legacy_fields
from .db import SessionLocal, engine, get_db, open_auth_db
from .book_activity import book_activity, start_book_thread
from filelock import FileLock
from . import model_connections, model_accounting
from . import scene_manuscript, prose_style
from . import book_library
from .book_paths import book_memory_dir
from .book_workspace import bind_book, current_book, new_book, upgrade_books, workspace_path
from .job_runner import run_generation_job
from .downloads import output_download_response, chapters_archive_response
from .models import ApiCredential, Base, Book, CapabilityPreference, GenerationJob, IdeaCopilotSession, ProviderConfig, User
from .provider_registry import (
    ProviderSpec,
    get_provider_specs,
    is_valid_slug,
    normalize_slug,
    normalize_wire_api,
)
from .security import SESSION_SECRET, encrypt_api_key, hash_password, verify_password
from .settings import BASE_DIR, RUNS_DIR, settings
from capability_registry import CAPABILITY_REGISTRY, capability_map, default_enabled_capability_slugs, normalize_capability_slugs

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

EVENT_LABELS = {
    "global_outline": "Claw builds global outline",
    "chapter_outline_ready": "Claw prepares chapter outline set",
    "chapter_plan": "Claw chapter plan",
    "chapter_outline": "Claw chapter outline",
    "chapter_length_plan": "Chapter length plan",
    "chapter_length_warning": "Chapter length warning",
    "character_setting": "Character setting",
    "world_setting": "World setting",
    "memory_snapshot": "Memory snapshot",
}


def _ui_language(request: Request) -> str:
    return "en"


def _console_texts(lang: str) -> Dict[str, str]:
    return {
        "brand_subtitle": "Creative Writing Console",
        "nav_create": "Create",
        "nav_chat": "Writing Chat",
        "nav_storyboard": "Storyboard",
        "nav_style": "Style Guide",
        "nav_assets": "Assets",
        "nav_manuscripts": "Drafts & Memory",
        "nav_characters": "Characters",
        "nav_world": "World",
        "nav_capabilities": "Capabilities",
        "nav_skills": "Skills",
        "nav_mcp": "MCP",
        "nav_agents": "Agent Config",
        "nav_control": "Control",
        "nav_sessions": "Sessions",
        "nav_runs": "Runs",
        "nav_status": "Status",
        "nav_setup": "Setup",
        "nav_models": "Models",
        "nav_environment": "Environment",
        "runtime_running": "NovelClaw running",
        "runtime_mode": "mode",
        "runtime_max_steps": "max_steps",
        "models_page_title": "Models",
        "models_page_desc": "Configure the providers, models, and API keys used by NovelClaw.",
        "models_heading_providers": "Providers",
        "models_api_key_label": "API key:",
        "models_saved": "Saved",
        "models_not_saved": "Not saved",
        "models_no_providers": "No providers yet.",
        "models_save_api_key": "Save API Key",
        "models_select_provider": "Select a provider",
        "models_enter_api_key": "Enter API key",
        "models_api_key_note": "Keys are encrypted before being stored in the local database.",
        "models_provider_form": "Add or update provider",
        "models_slug_placeholder": "Unique slug, e.g. openrouter_custom",
        "models_label_placeholder": "Display label",
        "models_base_url_placeholder": "Base URL",
        "models_model_placeholder": "Model name",
        "models_save_provider": "Save Provider",
        "models_delete_provider": "Delete",
    }


def _modelless_mode_enabled() -> bool:
    return bool(getattr(settings, "modelless_mode", False))


def _modelless_notice(lang: str) -> str:
    return "This deployment is running in model-free workspace mode. The portal can start, log in, browse the workspace, and manage local state without triggering any provider, API key, or model-dependent flow. If you later want generation, set WEB_MODELLESS_MODE=0 in local_web_portal/.env."


def _job_language(job: Optional[GenerationJob]) -> str:
    return "en"


def _event_label(event_name: str, language: str) -> str:
    labels = EVENT_LABELS
    return labels.get(event_name, event_name.replace("_", " "))

MEMORY_COUNT_KEYS = (
    "texts",
    "outlines",
    "characters",
    "world_settings",
    "plot_points",
    "fact_cards",
)

app = FastAPI(title=settings.app_name, dependencies=[Depends(bind_book)])
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie=settings.session_cookie_name,
    same_site="lax",
    https_only=settings.https_only,
    domain=settings.session_cookie_domain or None,
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


SHARED_PORTAL_PATHS = {
    "/login",
    "/register",
    "/verify-email",
    "/forgot-password",
    "/reset-password",
    "/account/password",
    "/select-mode",
    "/mode-a",
    "/mode-b",
    "/healthz",
}


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)
    upgrade_books(engine)
    # Recover jobs left in non-terminal state after a server/process restart.
    # Note: web jobs run in daemon threads; after restart those threads are gone.
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        running_cutoff = now - timedelta(seconds=max(60, settings.startup_recovery_seconds))
        queued_cutoff = now - timedelta(seconds=max(60, settings.stale_queued_seconds))

        stale_running_jobs = db.execute(
            select(GenerationJob).where(
                GenerationJob.status == "running",
                GenerationJob.updated_at < running_cutoff,
            )
        ).scalars().all()
        stale_queued_jobs = db.execute(
            select(GenerationJob).where(
                GenerationJob.status == "queued",
                GenerationJob.created_at < queued_cutoff,
            )
        ).scalars().all()

        for job in stale_running_jobs:
            job.status = "failed"
            prefix = (job.error_message + "\n") if job.error_message else ""
            job.error_message = prefix + "[system] marked as failed after app restart (worker thread was lost)."
            job.finished_at = now
            job.updated_at = now

        for job in stale_queued_jobs:
            job.status = "failed"
            prefix = (job.error_message + "\n") if job.error_message else ""
            job.error_message = prefix + "[system] queued too long without active worker; marked failed on startup."
            job.finished_at = now
            job.updated_at = now

        if stale_running_jobs or stale_queued_jobs:
            db.commit()


def _current_user(request: Request, db: Session) -> Optional[User]:
    uid = request.session.get("uid")
    if not uid:
        return None
    try:
        clean_uid = int(uid)
    except (TypeError, ValueError):
        return None

    user = db.get(User, clean_uid)
    if user:
        return user
    return _sync_user_from_auth_db(db, clean_uid)


def _sync_user_from_auth_db(db: Session, user_id: int) -> Optional[User]:
    auth_db = open_auth_db()
    if auth_db is None:
        return None
    try:
        auth_user = auth_db.get(User, user_id)
        if not auth_user:
            return None

        existing = db.get(User, user_id)
        if existing:
            return existing

        shadow_user = User(
            id=int(auth_user.id),
            email=str(auth_user.email or "").strip(),
            password_hash=str(auth_user.password_hash or ""),
            created_at=getattr(auth_user, "created_at", None) or datetime.now(timezone.utc),
        )
        db.add(shadow_user)
        db.commit()
        return db.get(User, user_id)
    except Exception:
        db.rollback()
        return None
    finally:
        auth_db.close()


def _require_user(request: Request, db: Session) -> User:
    user = _current_user(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return user


def _is_shared_portal_path(path: str) -> bool:
    return any(path == item or path.startswith(f"{item}/") for item in SHARED_PORTAL_PATHS)


def _shared_path(path: str) -> str:
    normalized = (path or "").strip() or "/"
    normalized = normalized if normalized.startswith("/") else f"/{normalized}"
    if _is_shared_portal_path(normalized) and settings.shared_portal_url:
        return f"{settings.shared_portal_url}{normalized}"
    return normalized


def _external_host(request: Request) -> str:
    host = request.url.hostname or "localhost"
    if host in {"0.0.0.0", "::"}:
        host = "localhost"
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _shared_portal_url(request: Request, path: str) -> str:
    normalized = (path or "").strip() or "/select-mode"
    normalized = normalized if normalized.startswith("/") else f"/{normalized}"
    if settings.shared_portal_url:
        return f"{settings.shared_portal_url}{normalized}"
    return urlunsplit((request.url.scheme, f"{_external_host(request)}:{settings.shared_portal_port}", normalized, "", ""))


def _app_path(path: str) -> str:
    book = current_book.get()
    if book and workspace_path(path):
        path = f"/books/{book.id}{path}"
    normalized = _shared_path(path)
    if _is_shared_portal_path(normalized) and settings.shared_portal_url:
        return f"{settings.shared_portal_url}{normalized}"
    if not settings.base_path:
        return normalized
    if normalized == settings.base_path or normalized.startswith(f"{settings.base_path}/"):
        return normalized
    if _is_shared_portal_path(normalized):
        return normalized
    return f"{settings.base_path}{normalized}"


def _static_asset_path(path: str) -> str:
    asset = (path or "").strip().lstrip("/")
    return _app_path(f"/static/{asset}")


def _current_public_path(request: Request) -> str:
    query = f"?{request.url.query}" if request.url.query else ""
    return f"{_app_path(request.url.path)}{query}"


def _redirect(path: str) -> RedirectResponse:
    parts = urlsplit(path)
    target = urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            _app_path(parts.path),
            parts.query,
            parts.fragment,
        )
    )
    return RedirectResponse(target, status_code=status.HTTP_303_SEE_OTHER)


def _safe_next_path(target: str, default: str = "/console/chat") -> str:
    normalized = (target or "").strip()
    if not normalized or not normalized.startswith("/") or normalized.startswith("//"):
        return _app_path(default)
    return normalized


def _redirect_with_notice(path: str, *, message: str = "", error: str = "") -> RedirectResponse:
    query_parts = []
    if message:
        query_parts.append(f"message={quote_plus(message)}")
    if error:
        query_parts.append(f"error={quote_plus(error)}")
    if query_parts:
        parts = urlsplit(path)
        existing_query = parts.query
        merged_query = "&".join([item for item in [existing_query, *query_parts] if item])
        path = urlunsplit((parts.scheme, parts.netloc, parts.path, merged_query, parts.fragment))
    return _redirect(path)


def _reject_when_modelless(
    request: Request,
    *,
    path: str = "/console/chat",
    fetch_status: int = 409,
) -> RedirectResponse | JSONResponse:
    message = _modelless_notice(_ui_language(request))
    if request.headers.get("x-requested-with") == "fetch":
        return JSONResponse({"ok": False, "error": message}, status_code=fetch_status)
    return _redirect_with_notice(path, error=message)


def _mask_hint(raw: str) -> str:
    if len(raw) < 8:
        return "********"
    return f"{raw[:4]}...{raw[-4:]}"


templates.env.globals["app_path"] = _app_path
templates.env.globals["static_asset_path"] = _static_asset_path
templates.env.globals["shared_path"] = _shared_path
templates.env.globals["current_public_path"] = _current_public_path
templates.env.globals["app_base_path"] = settings.base_path


@app.get("/healthz")
def healthz():
    return {"ok": True}


def _tail_text(path: Path, max_chars: int = 12000) -> str:
    return ANSI_ESCAPE_RE.sub("", tail_text(path, max_chars))


def _resolve_run_dir(run_id: str) -> Path:
    try:
        return resolve_run_directory(run_id, RUNS_DIR, BASE_DIR / "runs", BASE_DIR.parent / "runs")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc


def _capability_preferences_for_user(db: Session, user_id: int) -> Dict[str, CapabilityPreference]:
    rows = (
        db.execute(
            select(CapabilityPreference).where(CapabilityPreference.user_id == user_id)
        )
        .scalars()
        .all()
    )
    return {str(row.slug or "").strip().lower(): row for row in rows if row.slug}


def _enabled_capability_slugs_for_user(db: Session, user_id: int) -> set[str]:
    pref_map = _capability_preferences_for_user(db, user_id)
    enabled = set(default_enabled_capability_slugs())
    for slug, row in pref_map.items():
        if getattr(row, "enabled", False):
            enabled.add(slug)
        else:
            enabled.discard(slug)
    return normalize_capability_slugs(enabled)


def _build_capability_catalog(db: Session, user_id: int, language: str) -> List[Dict[str, object]]:
    pref_map = _capability_preferences_for_user(db, user_id)
    enabled = _enabled_capability_slugs_for_user(db, user_id)
    items: List[Dict[str, object]] = []
    for spec in sorted(CAPABILITY_REGISTRY, key=lambda item: (item.order, item.slug)):
        items.append(
            {
                "slug": spec.slug,
                "name": spec.name_en,
                "category": spec.category_en,
                "description": spec.description_en,
                "enabled": bool(spec.always_enabled or spec.slug in enabled),
                "always_enabled": bool(spec.always_enabled),
                "manager_action": spec.manager_action or "",
                "is_configured": spec.slug in pref_map,
            }
        )
    return items


def _list_custom_providers(db: Session, user_id: int) -> List[ProviderConfig]:
    return (
        db.execute(
            select(ProviderConfig)
            .where(ProviderConfig.user_id == user_id)
            .order_by(ProviderConfig.slug.asc())
        )
        .scalars()
        .all()
    )


def _provider_specs_for_user(db, user_id):
    if _modelless_mode_enabled():
        return {}
    return model_connections.connections(db, user_id)


def _all_provider_specs_for_user(db, user_id):
    return model_connections.connections(db, user_id)


def _provider_api_key(db, user_id, provider):
    spec = model_connections.connections(db, user_id).get(provider)
    if not spec:
        raise ValueError("Provider is no longer available")
    return model_connections.credential(db, user_id, spec)


def _chosen(spec, model="", snapshot=None):
    try:
        return model_connections.select_model(spec, model, snapshot)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


def _load_console_jobs(db: Session, user_id: int, book_id=None, limit: int = 20) -> List[GenerationJob]:
    return (
        db.execute(
            select(GenerationJob)
            .where(GenerationJob.user_id == user_id, GenerationJob.book_id == book_id)
            .order_by(GenerationJob.created_at.desc(), GenerationJob.id.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )


def _load_console_sessions(db: Session, user_id: int, book_id=None, limit: int = 20) -> List[IdeaCopilotSession]:
    return (
        db.execute(
            select(IdeaCopilotSession)
            .where(IdeaCopilotSession.user_id == user_id, IdeaCopilotSession.book_id == book_id)
            .order_by(IdeaCopilotSession.updated_at.desc(), IdeaCopilotSession.id.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )


def _memory_index_candidates(run_id: str) -> List[Path]:
    base_root = BASE_DIR.parent
    clean_run_id = str(run_id or "").strip()
    if not clean_run_id:
        return []
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", clean_run_id):
        return []
    vector_root = Path(os.getenv("VECTOR_DB_PATH", "").strip() or base_root / "vector_db").expanduser()
    if not vector_root.is_absolute():
        vector_root = base_root / vector_root
    return [(vector_root / "memory" / f"run_{clean_run_id}" / "memory_index.json").resolve()]


def _load_memory_index_for_run(run_id: str) -> Dict:
    candidates = _memory_index_candidates(run_id)
    for path in candidates:
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return {}


def _empty_memory_index() -> Dict[str, object]:
    return {
        "schema_version": 2,
        "texts": [],
        "outlines": [],
        "characters": [],
        "world_settings": [],
        "plot_points": [],
        "fact_cards": [],
        "claw": {bank: [] for bank in MemorySystem.CLAW_BANKS},
    }


def _ensure_memory_index_shape(index: Dict) -> Dict:
    value = index if isinstance(index, dict) else {}
    value.setdefault("schema_version", 2)
    for key in ("texts", "outlines", "characters", "world_settings", "plot_points", "fact_cards"):
        bucket = value.get(key)
        value[key] = bucket if isinstance(bucket, list) else []
    claw = value.get("claw")
    if not isinstance(claw, dict):
        claw = {}
        value["claw"] = claw
    for bank in MemorySystem.CLAW_BANKS:
        entries = claw.get(bank)
        claw[bank] = entries if isinstance(entries, list) else []
    return value


def _load_editable_memory_index_for_run(run_id: str) -> MemoryIndex:
    path_value = _memory_index_path_for_run(run_id)
    if not path_value:
        raise HTTPException(status_code=404, detail="Run not found")
    try:
        loaded = load_memory_index(Path(path_value))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=409, detail="Memory index could not be read; existing data has been preserved") from exc
    return _ensure_memory_index_shape(loaded)


def _save_memory_index_for_run(run_id: str, index: MemoryIndex) -> None:
    path_value = _memory_index_path_for_run(run_id)
    if not path_value:
        raise ValueError("Memory index path is unavailable")
    save_memory_index(Path(path_value), index)


def _job_for_run_id(db: Session, user_id: int, run_id: str) -> Optional[GenerationJob]:
    clean_run_id = str(run_id or "").strip()
    if not clean_run_id:
        return None
    return db.execute(
        select(GenerationJob)
        .where(
            GenerationJob.user_id == user_id,
            GenerationJob.run_id == clean_run_id,
        )
        .order_by(GenerationJob.updated_at.desc(), GenerationJob.id.desc())
    ).scalars().first()


def _memory_topic_hint(index: Dict, fallback: str = "global") -> str:
    claw = index.get("claw", {}) if isinstance(index, dict) else {}
    if isinstance(claw, dict):
        for bank in MemorySystem.CLAW_BANKS:
            entries = claw.get(bank) or []
            if entries:
                topic = str((entries[-1] or {}).get("topic") or "").strip()
                if topic:
                    return topic
    for bucket in ("texts", "outlines", "characters", "world_settings", "plot_points", "fact_cards"):
        items = index.get(bucket) or []
        if items:
            topic = str((items[-1] or {}).get("topic") or "").strip()
            if topic:
                return topic
    return fallback


def _portal_memory_system(book_key=None) -> MemorySystem:
    # Each writer owns a snapshot; saves merge with other writers under a file lock.
    cfg = Config(require_api_key=False)
    cfg.language = "en"
    cfg.memory_only_mode = True
    cfg.enable_rag = False
    cfg.enable_static_kb = False
    cfg.embedding_model = "none"
    if book_key:
        cfg.memory_vector_db_path = str(book_memory_dir(cfg, book_key))
        cfg.book_scoped = True
    return MemorySystem(cfg)


def _book_notes(book_id: int | None) -> str:
    if not book_id:
        return ""
    with SessionLocal() as db:
        book = db.get(Book, book_id)
        if not book:
            raise ValueError("Book no longer exists")
        index = _load_memory_index_for_run(book.memory_key)
    notes = []
    for label, items in [("Characters", index.get("characters", [])), ("World", index.get("world_settings", [])),
                         *((bank, rows) for bank, rows in index.get("claw", {}).items()
                           if bank in {"story_premise", "task_briefs", "style_guide", "decision_log", "continuity_facts", "entity_state", "relationship_state"})]:
        for item in items[-5:]:
            content = str(item.get("content") or "").strip()
            if content:
                notes.append(f"{label}: {content[:480]}")
    return "\n".join(notes)[:8000]


def _remember_session_turn(
    session: IdeaCopilotSession,
    state: Dict,
    *,
    user_reply: str = "",
    turn: Optional[Dict] = None,
    confirmed: bool = False,
    final_topic: str = "",
) -> None:
    try:
        with SessionLocal() as db:
            book = db.get(Book, session.book_id) if session.book_id else None
            memory = _portal_memory_system(book.memory_key if book else None)
        topic = str(state.get("refined_idea") or session.refined_idea or session.original_idea or final_topic or f"session:{session.id}").strip()
        if not topic:
            topic = f"session:{session.id}"
        preferred_language = "en"
        source_language = str(state.get("source_language") or preferred_language or "en").lower()
        memory.store_claw_memory(
            "session_profile",
            (
                f"session_id={session.id}\n"
                f"provider={session.provider}\n"
                f"status={session.status}\n"
                f"round={int(state.get('round', 0) or 0)}"
            ),
            topic,
            metadata={"session_id": session.id},
            store_vector=False,
        )
        memory.store_claw_memory(
            "language_profile",
            (
                f"preferred_language={preferred_language}\n"
                f"source_language={source_language}\n"
                f"translation_mode={state.get('translation_mode', 'english')}"
            ),
            topic,
            metadata={"session_id": session.id},
            store_vector=False,
        )
        refined = str(state.get("refined_idea") or session.refined_idea or session.original_idea or "").strip()
        if refined:
            memory.store_claw_memory(
                "task_briefs",
                refined,
                topic,
                metadata={"session_id": session.id},
                store_vector=True,
            )
            memory.store_claw_memory(
                "story_premise",
                refined,
                topic,
                metadata={"session_id": session.id},
                store_vector=False,
            )
        if user_reply.strip():
            memory.store_claw_memory(
                "user_preferences",
                user_reply.strip(),
                topic,
                metadata={"session_id": session.id},
                store_vector=True,
            )
        if turn:
            analysis = str(turn.get("analysis") or "").strip()
            if analysis:
                memory.store_claw_memory(
                    "working_set",
                    analysis,
                    topic,
                    metadata={"session_id": session.id, "kind": "analysis"},
                    store_vector=True,
                )
            style_targets = turn.get("style_targets") or []
            if isinstance(style_targets, list) and style_targets:
                memory.store_claw_memory(
                    "style_guide",
                    "\n".join(str(item) for item in style_targets[:8]),
                    topic,
                    metadata={"session_id": session.id},
                    store_vector=False,
                )
            questions = turn.get("questions") or []
            if isinstance(questions, list) and questions:
                memory.store_claw_memory(
                    "decision_log",
                    "next_questions=\n" + "\n".join(str(q) for q in questions[:3]),
                    topic,
                    metadata={"session_id": session.id},
                    store_vector=False,
                )
        if confirmed:
            memory.store_claw_memory(
                "decision_log",
                "session_confirmed_for_generation",
                topic,
                metadata={"session_id": session.id},
                store_vector=False,
            )
    except Exception:
        pass
def _build_workspace_files(run_id: str) -> List[Dict[str, str]]:
    if not run_id:
        return []
    run_dir = _resolve_run_dir(run_id)
    if not run_dir.exists():
        return []
    files: List[Dict[str, str]] = []
    for name in ["output.txt", "result.json", "round_results.json", "metadata.json", "worker.log", "progress.log"]:
        path = run_dir / name
        if not path.exists():
            continue
        files.append(
            {
                "name": name,
                "path": str(path),
                "size": f"{max(1, int(path.stat().st_size / 1024))} KB",
                "preview": _tail_text(path, max_chars=300),
            }
        )
    workspace_dir = run_dir / "workspace"
    if workspace_dir.exists():
        for path in sorted([item for item in workspace_dir.rglob("*") if item.is_file()])[:20]:
            files.append(
                {
                    "name": str(path.relative_to(run_dir)).replace("\\", "/"),
                    "path": str(path),
                    "size": f"{max(1, int(path.stat().st_size / 1024))} KB",
                    "preview": _tail_text(path, max_chars=300),
                }
            )
    return files


def _build_memory_overview(run_id: str) -> Dict:
    index = _load_memory_index_for_run(run_id)
    claw = index.get("claw", {}) if isinstance(index, dict) else {}
    counts = {
        "texts": len(index.get("texts", [])) if isinstance(index, dict) else 0,
        "outlines": len(index.get("outlines", [])) if isinstance(index, dict) else 0,
        "characters": len(index.get("characters", [])) if isinstance(index, dict) else 0,
        "world_settings": len(index.get("world_settings", [])) if isinstance(index, dict) else 0,
        "plot_points": len(index.get("plot_points", [])) if isinstance(index, dict) else 0,
        "fact_cards": len(index.get("fact_cards", [])) if isinstance(index, dict) else 0,
    }
    claw_counts = {key: len(value or []) for key, value in claw.items()} if isinstance(claw, dict) else {}
    latest_claw = {}
    if isinstance(claw, dict):
        for key, value in claw.items():
            latest_claw[key] = value[-1] if value else None
    return {
        "index": index,
        "counts": counts,
        "claw_counts": claw_counts,
        "latest_claw": latest_claw,
        "index_path": _memory_index_path_for_run(run_id),
    }


def _tail_items(items: List[Dict], limit: int = 8) -> List[Dict]:
    return list(items[-limit:]) if items else []


def _build_agent_catalog() -> List[Dict[str, str]]:
    return [
        {"slug": "openclaw_manager", "name": "OpenClaw Manager", "role": "dynamic orchestration", "desc": "Chooses the next writing action from live signals instead of following a fixed workflow."},
        {"slug": "idea_analyzer", "name": "IdeaAnalyzer", "role": "idea shaping", "desc": "Turns raw ideas into a stable writing brief with genre, conflict, tone, cast, and constraints."},
        {"slug": "analyzer", "name": "Analyzer", "role": "task understanding", "desc": "Decides whether the current turn needs planning, drafting, rewriting, worldbuilding, or cleanup."},
        {"slug": "plot", "name": "Plot Agent", "role": "plot progression", "desc": "Builds outlines, chapter briefs, conflict escalations, and turning-point candidates."},
        {"slug": "character", "name": "Character Agent", "role": "character arc", "desc": "Maintains character setup, motivation, relationships, and behavior boundaries."},
        {"slug": "world", "name": "World Agent", "role": "world constraints", "desc": "Stores world rules, setting boundaries, and reusable canon facts for later writing."},
        {"slug": "retrieval", "name": "Retrieval Agent", "role": "memory retrieval", "desc": "Pulls relevant context back from memory, chapters, facts, and prior drafts."},
        {"slug": "writer", "name": "Writer Agent", "role": "draft generation", "desc": "Drafts, continues, and rewrites prose. This is the core writing engine inside NovelClaw."},
        {"slug": "evaluator", "name": "Evaluator Agent", "role": "quality review", "desc": "Scores coherence, pacing, emotion, structure, and task-fit of candidate drafts."},
        {"slug": "judge", "name": "Judge Agent", "role": "candidate selection", "desc": "Chooses between close draft candidates when the loop needs a final decision."},
        {"slug": "turning_point_tracker", "name": "TurningPointTracker", "role": "turning-point tracking", "desc": "Tracks story beats and whether the chapter is meaningfully advancing."},
        {"slug": "consistency_checker", "name": "ConsistencyChecker", "role": "continuity checking", "desc": "Checks character, world, timeline, and fact consistency across outputs."},
        {"slug": "realtime_editor", "name": "RealtimeEditor", "role": "live editing", "desc": "Finds weak spans and applies targeted revision passes during iteration."},
    ]


def _build_story_assets(memory_overview: Dict, active_session: Optional[IdeaCopilotSession], latest_job: Optional[GenerationJob]) -> Dict:
    index = memory_overview.get("index", {}) if isinstance(memory_overview, dict) else {}
    outlines = list(index.get("outlines", []) or [])
    plot_points = list(index.get("plot_points", []) or [])
    characters = list(index.get("characters", []) or [])
    world_settings = list(index.get("world_settings", []) or [])
    fact_cards = list(index.get("fact_cards", []) or [])
    texts = list(index.get("texts", []) or [])
    refined_idea = active_session.refined_idea if active_session and active_session.refined_idea else (active_session.original_idea if active_session else "")
    run_id = latest_job.run_id if latest_job and latest_job.run_id else ""
    run_dir = _resolve_run_dir(run_id) if run_id else None
    chapter_outputs = _load_chapter_outputs(run_dir) if run_dir else []
    latest_chapter = chapter_outputs[-1] if chapter_outputs else None
    latest_text = texts[-1] if texts else None
    project_name = refined_idea or "Untitled Project"
    if len(project_name) > 28:
        project_name = project_name[:28] + "..."
    return {
        "project_name": project_name,
        "refined_idea": refined_idea,
        "latest_run_id": run_id,
        "outlines": _tail_items(outlines, 10),
        "plot_points": _tail_items(plot_points, 10),
        "characters": _tail_items(characters, 12),
        "world_settings": _tail_items(world_settings, 8),
        "fact_cards": _tail_items(fact_cards, 12),
        "texts": _tail_items(texts, 6),
        "latest_text": latest_text,
        "chapter_outputs": chapter_outputs,
        "latest_chapter": latest_chapter,
    }


def _empty_story_assets() -> Dict[str, object]:
    return {
        "project_name": "",
        "refined_idea": "",
        "latest_run_id": "",
        "outlines": [],
        "plot_points": [],
        "characters": [],
        "world_settings": [],
        "fact_cards": [],
        "texts": [],
        "latest_text": None,
        "chapter_outputs": [],
        "latest_chapter": None,
    }


def _build_workspace_bootstrap(
    story_assets: Dict,
    memory_overview: Dict,
    memory_bank_groups: List[Dict[str, object]],
    language: str,
    *,
    download_job_id: int = 0,
) -> Dict[str, object]:
    index = memory_overview.get("index", {}) if isinstance(memory_overview, dict) else {}
    claw = index.get("claw", {}) if isinstance(index, dict) else {}
    outlines = []
    for item in list(story_assets.get("outlines") or [])[-24:]:
        structure = item.get("structure") if isinstance(item.get("structure"), dict) else {}
        kind = str(structure.get("kind") or "outline").strip() or "outline"
        title = str(structure.get("title") or "").strip()
        chapter = structure.get("chapter")
        if not title:
            if kind == "global_outline":
                title = "Global Outline"
            elif kind == "chapter_outline" and chapter:
                title = f"Chapter {chapter} Outline"
            else:
                title = kind.replace("_", " ").strip().title()
        outlines.append(
            {
                "id": str(item.get("id") or ""),
                "kind": kind,
                "chapter": chapter,
                "title": title,
                "content": str(item.get("content") or "").strip(),
                "timestamp": str(item.get("timestamp") or ""),
                "source": str(structure.get("source") or ""),
            }
        )

    tactical_banks = {}
    for bank in [
        "story_premise",
        "task_briefs",
        "style_guide",
        "chapter_briefs",
        "scene_cards",
        "working_set",
        "decision_log",
        "entity_state",
        "relationship_state",
        "world_state",
        "continuity_facts",
    ]:
        entries = list(claw.get(bank, []) or []) if isinstance(claw, dict) else []
        tactical_banks[bank] = [_serialize_claw_entry(item) for item in entries[-10:]][::-1]

    chapters = []
    for item in story_assets.get("chapter_outputs") or []:
        chapters.append(
            {
                "chapter": item.get("chapter"),
                "iteration": item.get("iteration"),
                "filename": item.get("filename"),
                "content": str(item.get("content") or "").strip(),
            }
        )

    plot_points = []
    for item in story_assets.get("plot_points") or []:
        plot_points.append(
            {
                "position": str(item.get("position") or "").strip(),
                "content": str(item.get("content") or "").strip(),
                "timestamp": str(item.get("timestamp") or ""),
            }
        )

    return {
        "project_name": str(story_assets.get("project_name") or ""),
        "run_id": str(story_assets.get("latest_run_id") or ""),
        "download_job_id": int(download_job_id or 0),
        "chapters": chapters,
        "outlines": outlines,
        "plot_points": plot_points,
        "tactical_banks": tactical_banks,
        "memory_groups": memory_bank_groups,
    }


def _stringify_memory_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " / ".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, dict):
        return " / ".join(f"{key}: {val}" for key, val in value.items() if str(val).strip())
    return str(value).strip()



def _compact_preview(text: str, limit: int = 180) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _extract_generation_brief_summary(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    lines = [line.rstrip() for line in raw.splitlines()]
    header = lines[0].strip() if lines else ""
    if header not in {"[NovelClaw generation brief]", legacy_fields.LEGACY_BRIEF_HEADER}:
        return raw

    summary_lines: List[str] = []
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            if summary_lines and summary_lines[-1] != "":
                summary_lines.append("")
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            break
        summary_lines.append(line)
    return "\n".join(summary_lines).strip() or raw


def _extract_markdown_section(text: str, heading: str) -> str:
    raw = str(text or "").strip()
    if not raw or heading not in raw:
        return ""
    lines = raw.splitlines()
    capture = False
    kept: List[str] = []
    heading_level = heading.count("#")
    for line in lines:
        stripped = line.strip()
        if stripped == heading:
            capture = True
            continue
        if not capture:
            continue
        if stripped.startswith("#") and stripped.count("#") <= heading_level:
            break
        kept.append(line)
    return "\n".join(kept).strip()


def _clean_chapter_brief_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""

    planning_section = _extract_markdown_section(raw, "## Unified Planning Packet")
    if planning_section:
        return planning_section

    cleaned_lines: List[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        if stripped in {"## Support Signals", "## Claw History", "## Candidate Excerpt"}:
            break
        if re.match(r"^#\s+", stripped):
            continue
        if re.match(r"^-\s*(topic|chapter|reward|issues|length)\s*:", stripped, re.IGNORECASE):
            continue
        if re.match(r"^(topic|chapter|reward|issues|length)\s*=", stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip() or raw


def _display_claw_entry_text(entry: Dict[str, object]) -> Tuple[str, str]:
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    topic = _extract_generation_brief_summary(entry.get("topic") or "")
    content = str(entry.get("content") or "").strip()
    bank = str(entry.get("bank") or "").strip()
    tool = str(metadata.get("tool") or metadata.get("source_type") or "").strip()

    if bank in {"task_briefs", "story_premise"}:
        content = _extract_generation_brief_summary(content)
        topic = _compact_preview(_extract_generation_brief_summary(topic or content), 100)
    elif bank == "chapter_briefs" or tool in {"sync_storyboard", "storyboard_sync"} or "## Unified Planning Packet" in content:
        content = _clean_chapter_brief_text(content)
        if not topic:
            first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
            topic = _compact_preview(first_line, 100)
    else:
        topic = _compact_preview(topic, 100)

    return topic, content


def _session_status_label(language: str, value: object) -> str:
    normalized = str(value or "").strip().lower()
    mapping = {
        "active": "Active",
        "confirmed": "Confirmed",
        "queued": "Queued",
        "running": "Running",
        "succeeded": "Succeeded",
        "failed": "Failed",
        "canceled": "Canceled",
        "deleted": "Deleted",
        "new": "New",
    }
    return mapping.get(normalized, str(value or "").strip())


def _find_memory_bucket_item(index: Dict, bucket: str, item_id: str) -> Optional[Dict[str, object]]:
    items = index.get(bucket) if isinstance(index, dict) else None
    if not isinstance(items, list):
        return None
    target_id = str(item_id or "").strip()
    if not target_id:
        return None
    for item in items:
        if isinstance(item, dict) and str(item.get("id") or "").strip() == target_id:
            return item
    return None



def _nonempty_lines(text: str) -> List[str]:
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]



def _strip_list_prefix(text: str) -> str:
    return re.sub(r"^[\-?*#\d\s\.\)\(]+", "", str(text or "").strip())



def _parse_text_pairs(text: str) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    for line in _nonempty_lines(text):
        cleaned = _strip_list_prefix(line)
        for sep in ("?", ":", "\uff1a"):
            if sep not in cleaned:
                continue
            key, value = cleaned.split(sep, 1)
            key = legacy_fields.field_name(key)
            value = value.strip()
            if key and value and len(key) <= 24:
                pairs[key] = value
                break
    return pairs



def _field_from_aliases(attrs: Dict[str, object], pairs: Dict[str, str], aliases: List[str]) -> str:
    for alias in aliases:
        alias_key = alias.lower()
        if alias_key in attrs:
            value = _stringify_memory_value(attrs.get(alias_key))
            if value:
                return value
        if alias_key in pairs:
            value = _stringify_memory_value(pairs.get(alias_key))
            if value:
                return value
    return ""


def _normalize_entity_name(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "").strip()).lower()


def _character_entry_chapter(item: Dict[str, object], attrs: Dict[str, object], pairs: Dict[str, str]) -> Optional[int]:
    for source in (attrs, pairs):
        for key in ("chapter", "chapter_no", "current_chapter"):
            raw = source.get(key) if isinstance(source, dict) else None
            if raw in (None, ""):
                continue
            match = re.search(r"\d+", str(raw))
            if match:
                return int(match.group(0))
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    raw_meta = metadata.get("chapter") if isinstance(metadata, dict) else None
    if raw_meta not in (None, ""):
        match = re.search(r"\d+", str(raw_meta))
        if match:
            return int(match.group(0))
    return None


def _is_character_record(name: str, raw_text: str, attrs: Dict[str, object]) -> bool:
    normalized = _normalize_entity_name(name)
    if not normalized:
        return False
    if normalized.endswith("_world_setting"):
        return False
    blocked_tokens = (*legacy_fields.WORLD_NAME_MARKERS, "world_setting", "world state", "rule", "canon")
    lowered_name = str(name or "").lower()
    if any(token in lowered_name or token in str(name or "") for token in blocked_tokens):
        return False
    if _stringify_memory_value(attrs.get("role")):
        return True
    lowered_raw = str(raw_text or "").lower()
    if any(token in lowered_raw for token in ("world setting", "world state", "continuity fact", "canon fact")):
        return False
    if any(token in str(raw_text or "") for token in legacy_fields.WORLD_TEXT_MARKERS):
        return False
    return True


def _character_change_summary(name: str, raw_text: str, attrs: Dict[str, object], pairs: Dict[str, str]) -> str:
    for aliases in (
        ["summary"],
        ["change", "growth", "arc"],
        ["goal", "purpose", "want"],
        ["motivation", "desire"],
        ["conflict", "risk"],
    ):
        value = _field_from_aliases(attrs, pairs, aliases)
        if value:
            return _compact_preview(value, 220)
    lines = [line for line in _nonempty_lines(raw_text) if name not in line]
    return _compact_preview(lines[0] if lines else raw_text, 220)


def _memory_index_path_for_run(run_id: str) -> str:
    candidates = _memory_index_candidates(run_id)
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0]) if candidates else ""



def _build_storyboard_view(story_assets: Dict, language: str) -> Dict[str, object]:
    outline_cards = []
    spotlight = None
    for idx, item in enumerate(story_assets.get("outlines", []) or [], start=1):
        structure = item.get("structure") if isinstance(item.get("structure"), dict) else {}
        kind = str(structure.get("kind") or "").strip() or "outline"
        chapter = structure.get("chapter")
        title = str(structure.get("title") or "").strip()
        if not title:
            if kind == "global_outline":
                title = "Global Outline"
            elif kind == "chapter_outline" and chapter:
                title = f"Chapter {chapter} Outline"
            else:
                title = kind.replace("_", " ").strip().title()
        bucket = "global" if kind == "global_outline" or not chapter else "chapter"
        card = {
            "id": str(item.get("id") or ""),
            "title": title,
            "kind": kind,
            "bucket": bucket,
            "chapter": chapter,
            "summary": _compact_preview(item.get("content") or "", 420 if bucket == "global" else 280),
            "detail": str(item.get("content") or "").strip(),
            "meta": [
                segment
                for segment in [
                    f"Chapter {chapter}" if chapter else "Global",
                    structure.get("phase"),
                    item.get("timestamp"),
                ]
                if segment
            ],
            "chip": "Master" if bucket == "global" else "Chapter",
        }
        outline_cards.append(card)
        if spotlight is None or bucket == "global":
            spotlight = card

    plot_points = []
    for idx, item in enumerate(story_assets.get("plot_points", []) or [], start=1):
        plot_points.append(
            {
                "index": idx,
                "title": item.get("position") or f"Beat {idx}",
                "summary": _compact_preview(item.get("content") or "", 240),
                "detail": str(item.get("content") or "").strip(),
                "timestamp": str(item.get("timestamp") or ""),
            }
        )

    chapter_outputs = []
    for item in story_assets.get("chapter_outputs", []) or []:
        chapter_no = item.get("chapter")
        title = (
            f"Chapter {chapter_no}"
            if chapter_no
            else (item.get("filename") or "Chapter file")
        )
        chapter_outputs.append(
            {
                "chapter": chapter_no,
                "iteration": item.get("iteration"),
                "title": title,
                "filename": item.get("filename") or "",
                "summary": _compact_preview(item.get("content") or "", 260),
                "detail": str(item.get("content") or "").strip(),
            }
        )

    return {
        "spotlight_outline": spotlight,
        "outlines": outline_cards,
        "plot_points": plot_points,
        "chapter_outputs": chapter_outputs,
        "latest_chapter": story_assets.get("latest_chapter"),
    }



def _build_character_cards(story_assets: Dict, language: str) -> List[Dict[str, object]]:
    field_specs = [
        ("Role", ["role", "identity", "position"]),
        ("Goal", ["goal", "purpose", "want"]),
        ("Motivation", ["motivation", "desire"]),
        ("Conflict", ["conflict", "risk"]),
        ("Relationships", ["relationships", "relation", "bond"]),
        ("Boundaries", ["boundary", "boundaries"]),
    ]
    groups: Dict[str, Dict[str, object]] = {}
    for idx, item in enumerate(story_assets.get("characters", []) or [], start=1):
        attrs_raw = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
        attrs = {legacy_fields.field_name(str(key)): value for key, value in attrs_raw.items()} if isinstance(attrs_raw, dict) else {}
        raw_text = str(item.get("character_info") or item.get("content") or "").strip()
        pairs = _parse_text_pairs(raw_text)
        name = (
            item.get("character_name")
            or item.get("name")
            or _field_from_aliases(attrs, pairs, ["name", "character_name"])
            or f"Character {idx}"
        )
        display_name = str(name).strip()
        if not _is_character_record(display_name, raw_text, attrs):
            continue
        summary = _character_change_summary(display_name, raw_text, attrs, pairs)
        fields = [{"label": label, "value": _field_from_aliases(attrs, pairs, aliases)} for label, aliases in field_specs]
        tags = []
        for key, value in attrs.items():
            rendered = _stringify_memory_value(value)
            if not rendered:
                continue
            if any(rendered == field["value"] for field in fields if field["value"]):
                continue
            tags.append({"label": key, "value": rendered})
            if len(tags) >= 6:
                break
        group_key = _normalize_entity_name(display_name)
        timestamp = str(item.get("timestamp") or "")
        chapter = _character_entry_chapter(item, attrs, pairs)
        group = groups.setdefault(
            group_key,
            {
                "name": display_name,
                "summary": "",
                "raw_text": "",
                "timestamp": "",
                "fields_map": {},
                "tags_map": {},
                "changes_map": {},
            },
        )
        if timestamp >= str(group.get("timestamp") or ""):
            group["name"] = display_name
            group["summary"] = summary
            group["raw_text"] = raw_text
            group["timestamp"] = timestamp
        for field in fields:
            if field["value"]:
                group["fields_map"][field["label"]] = field["value"]
        for tag in tags:
            group["tags_map"][tag["label"]] = tag["value"]
        change_key = f"chapter:{chapter}" if chapter else f"entry:{item.get('id') or timestamp or idx}"
        current_change = group["changes_map"].get(change_key)
        next_change = {"chapter": chapter, "timestamp": timestamp, "summary": summary, "raw_text": raw_text}
        if not current_change or timestamp >= str(current_change.get("timestamp") or ""):
            group["changes_map"][change_key] = next_change
    cards: List[Dict[str, object]] = []
    for group in groups.values():
        changes = sorted(
            list(group["changes_map"].values()),
            key=lambda item: (str(item.get("timestamp") or ""), int(item.get("chapter") or 0)),
            reverse=True,
        )
        latest_chapter = next((item.get("chapter") for item in changes if item.get("chapter")), None)
        cards.append(
            {
                "name": str(group["name"]).strip(),
                "summary": _compact_preview(str(group["summary"] or ""), 220),
                "fields": [
                    {"label": label, "value": group["fields_map"].get(label)}
                    for label, _aliases in field_specs
                    if group["fields_map"].get(label)
                ],
                "tags": [{"label": key, "value": value} for key, value in list(group["tags_map"].items())[:6]],
                "raw_text": str(group["raw_text"] or "").strip(),
                "timestamp": str(group["timestamp"] or ""),
                "changes": changes[:8],
                "change_count": len(changes),
                "latest_chapter": latest_chapter,
            }
        )
    cards.sort(key=lambda item: (str(item.get("timestamp") or ""), str(item.get("name") or "")), reverse=True)
    return cards



def _build_world_cards(story_assets: Dict, language: str) -> Dict[str, List[Dict[str, object]]]:
    settings: List[Dict[str, object]] = []
    for idx, item in enumerate(story_assets.get("world_settings", []) or [], start=1):
        raw_text = str(item.get("setting_info") or item.get("content") or "").strip()
        pairs = _parse_text_pairs(raw_text)
        title = item.get("name") or pairs.get("name") or f"Setting {idx}"
        fields = [
            {"label": "Category", "value": pairs.get("category", "")},
            {"label": "Rule", "value": pairs.get("rule", "")},
            {"label": "Boundary", "value": pairs.get("boundary", "")},
            {"label": "Impact", "value": pairs.get("impact", "")},
        ]
        lines = [line for line in _nonempty_lines(raw_text) if str(title) not in line]
        summary = lines[0] if lines else raw_text
        settings.append(
            {
                "title": str(title).strip(),
                "summary": _compact_preview(summary, 220),
                "fields": [field for field in fields if field["value"]],
                "raw_text": raw_text,
                "timestamp": item.get("timestamp") or "",
            }
        )
    facts: List[Dict[str, object]] = []
    for idx, item in enumerate(story_assets.get("fact_cards", []) or [], start=1):
        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        title = meta.get("title") or item.get("card_type") or f"{'Fact Card'} {idx}"
        facts.append(
            {
                "title": str(title).strip(),
                "type": item.get("card_type") or "General",
                "summary": _compact_preview(item.get("content") or "", 220),
                "raw_text": str(item.get("content") or "").strip(),
                "timestamp": item.get("timestamp") or "",
            }
        )
    return {"settings": settings, "facts": facts}



def _memory_bank_meta(language: str) -> Dict[str, Dict[str, str]]:
    return {
        "session_profile": {"name": "Session Profile", "desc": "Stores session state, round count, provider, and confirmation status."},
        "language_profile": {"name": "Language Profile", "desc": "Records the language used for a writing run."},
        "user_preferences": {"name": "User Preferences", "desc": "Captures your explicit style, taboo, and workflow requests."},
        "task_briefs": {"name": "Task Briefs", "desc": "Keeps the active writing brief available to later loop steps."},
        "story_premise": {"name": "Story Premise", "desc": "Stores the core premise and genre setup."},
        "style_guide": {"name": "Style Guide", "desc": "Accumulates voice, pacing, POV, and style requirements."},
        "chapter_briefs": {"name": "Chapter Briefs", "desc": "Stores chapter-level goals and task breakdowns."},
        "scene_cards": {"name": "Scene Cards", "desc": "Stores scene-level events, conflicts, and purposes."},
        "entity_state": {"name": "Entity State", "desc": "Tracks state changes for characters, factions, and objects."},
        "relationship_state": {"name": "Relationship State", "desc": "Tracks relationships and emotional tension."},
        "world_state": {"name": "World State", "desc": "Stores world rules, boundaries, and changes."},
        "continuity_facts": {"name": "Continuity Facts", "desc": "Stores canon facts that later chapters must respect."},
        "tool_observations": {"name": "Tool Observations", "desc": "Keeps observations emitted by local tool actions."},
        "decision_log": {"name": "Decision Log", "desc": "Preserves loop decisions, questions, and confirmation checkpoints."},
        "revision_notes": {"name": "Revision Notes", "desc": "Stores issues and targets for revision passes."},
        "working_set": {"name": "Working Set", "desc": "Keeps the most immediately relevant working analysis in play."},
    }

def _memory_bank_group_specs(language: str) -> List[Dict[str, object]]:
    return [
        {
            "slug": "premise",
            "name": "Story Premise",
            "description": "Core premise, genre direction, and the central story proposition.",
            "banks": ["story_premise"],
        },
        {
            "slug": "author_brief",
            "name": "Author Briefing",
            "description": "Task brief, style targets, user preferences, and writing preferences.",
            "banks": ["task_briefs", "style_guide", "user_preferences", "language_profile"],
        },
        {
            "slug": "chapter_planning",
            "name": "Chapter Planning",
            "description": "Chapter goals, scene decomposition, and progression routes.",
            "banks": ["chapter_briefs", "scene_cards"],
        },
        {
            "slug": "revision_loop",
            "name": "Revision Loop",
            "description": "Revision targets, loop decisions, and active working memory.",
            "banks": ["revision_notes", "working_set", "decision_log"],
        },
        {
            "slug": "canon_state",
            "name": "Canon State",
            "description": "Characters, relationships, world state, and continuity facts.",
            "banks": ["entity_state", "relationship_state", "world_state", "continuity_facts"],
        },
        {
            "slug": "runtime",
            "name": "Runtime Trace",
            "description": "Session state and tool observations that explain what the system is doing right now.",
            "banks": ["session_profile", "tool_observations"],
        },
    ]


def _serialize_claw_entry(entry: Dict) -> Dict[str, object]:
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    display_topic, display_content = _display_claw_entry_text(entry)
    return {
        "id": str(entry.get("id") or ""),
        "topic": display_topic,
        "content": display_content,
        "timestamp": str(entry.get("timestamp") or ""),
        "chapter": metadata.get("chapter"),
        "kind": str(metadata.get("kind") or metadata.get("type") or ""),
        "source": str(metadata.get("source") or ""),
        "metadata": metadata,
    }


def _build_memory_bank_groups_from_index(index: Dict, language: str, *, entry_limit: int = 12) -> List[Dict[str, object]]:
    meta = _memory_bank_meta(language)
    claw = index.get("claw", {}) if isinstance(index, dict) else {}
    groups: List[Dict[str, object]] = []
    for spec in _memory_bank_group_specs(language):
        banks: List[Dict[str, object]] = []
        total_count = 0
        for bank in spec["banks"]:
            entries = list(claw.get(bank, []) or []) if isinstance(claw, dict) else []
            serialized_entries = [_serialize_claw_entry(item) for item in entries[-entry_limit:]]
            serialized_entries.reverse()
            latest = serialized_entries[0] if serialized_entries else {}
            total_count += len(entries)
            banks.append(
                {
                    "slug": bank,
                    "name": meta[bank]["name"],
                    "description": meta[bank]["desc"],
                    "count": len(entries),
                    "latest_preview": _compact_preview(latest.get("content") or "", 160) if latest else "",
                    "latest_topic": _compact_preview(latest.get("topic") or "", 80) if latest else "",
                    "entries": serialized_entries,
                }
            )
        groups.append(
            {
                "slug": spec["slug"],
                "name": spec["name"],
                "description": spec["description"],
                "count": total_count,
                "banks": banks,
            }
        )
    return groups


def _build_memory_bank_cards(memory_overview: Dict, language: str) -> List[Dict[str, object]]:
    claw_counts = memory_overview.get("claw_counts", {}) if isinstance(memory_overview, dict) else {}
    latest_claw = memory_overview.get("latest_claw", {}) if isinstance(memory_overview, dict) else {}
    meta = _memory_bank_meta(language)
    cards: List[Dict[str, object]] = []
    for slug, spec in meta.items():
        latest = latest_claw.get(slug) if isinstance(latest_claw, dict) else None
        latest_content = latest.get("content") if isinstance(latest, dict) else ""
        cards.append(
            {
                "slug": slug,
                "name": spec["name"],
                "description": spec["desc"],
                "count": int(claw_counts.get(slug, 0) or 0),
                "latest_preview": _compact_preview(latest_content or "", 180),
                "latest_topic": _compact_preview(latest.get("topic") or "", 100) if isinstance(latest, dict) else "",
                "timestamp": latest.get("timestamp") or "" if isinstance(latest, dict) else "",
            }
        )
    cards.sort(key=lambda item: (item["count"], item["name"]), reverse=True)
    return cards



def _build_agent_runtime(agent_catalog: List[Dict[str, str]], capability_catalog: List[Dict[str, object]], story_assets: Dict, memory_overview: Dict, language: str) -> List[Dict[str, object]]:
    enabled_actions = {str(item.get("manager_action") or "") for item in capability_catalog if item.get("enabled") and item.get("manager_action")}
    claw_counts = memory_overview.get("claw_counts", {}) if isinstance(memory_overview, dict) else {}
    cards: List[Dict[str, object]] = []
    for agent in agent_catalog:
        slug = str(agent.get("slug") or "")
        evidence_count = 0
        if slug == "openclaw_manager":
            evidence_count = len(enabled_actions)
        elif slug == "plot":
            evidence_count = len(story_assets.get("outlines", []) or []) + len(story_assets.get("plot_points", []) or [])
        elif slug == "character":
            evidence_count = len(story_assets.get("characters", []) or [])
        elif slug == "world":
            evidence_count = len(story_assets.get("world_settings", []) or []) + len(story_assets.get("fact_cards", []) or [])
        elif slug == "retrieval":
            evidence_count = sum(int(value or 0) for value in claw_counts.values())
        elif slug == "writer":
            evidence_count = len(story_assets.get("texts", []) or []) + len(story_assets.get("chapter_outputs", []) or [])
        elif slug in {"evaluator", "judge", "consistency_checker", "realtime_editor"}:
            evidence_count = int(claw_counts.get("revision_notes", 0) or 0) + int(claw_counts.get("decision_log", 0) or 0)
        else:
            evidence_count = int(claw_counts.get("working_set", 0) or 0)
        cards.append({
            "slug": slug,
            "name": agent.get("name") or slug,
            "role": agent.get("role") or "",
            "description": agent.get("desc") or "",
            "status": "Live" if evidence_count else "Idle",
            "evidence_count": evidence_count,
        })
    return cards



def _build_mcp_surface(capability_catalog: List[Dict[str, object]], provider_count: int, memory_cards: List[Dict[str, object]], language: str) -> Dict[str, object]:
    local_tools = [item for item in capability_catalog if str(item.get("category") or "") in {"Local Tools", 'Local Tools'}]
    enabled_tools = [item for item in local_tools if item.get("enabled")]
    external_connectors = [
        {"name": "Browser", "status": "Not connected", "detail": "Browser automation is not connected to MCP yet."},
        {"name": "Filesystem", "status": "Handled internally", "detail": "Currently handled by local workspace tools instead of an external MCP server."},
        {"name": "Publish", "status": "Not connected", "detail": "Publishing, export, and platform sync are not wired in yet."},
    ]
    return {
        "provider_count": provider_count,
        "enabled_local_tools": enabled_tools,
        "memory_ready_banks": len([item for item in memory_cards if int(item.get("count", 0) or 0) > 0]),
        "external_connectors": external_connectors,
    }


def _build_style_profile(active_session: Optional[IdeaCopilotSession], latest_turn: Dict) -> Dict[str, List[str] | str]:
    base = (active_session.refined_idea if active_session and active_session.refined_idea else (active_session.original_idea if active_session else "")).strip()
    hints: List[str] = []
    for token in [
        "first person", "third person", "ensemble", "restrained", "lyrical", "dark", "cozy", "suspense",
        "fantasy", "romance", "science fiction", "thriller", "slow burn", "fast pace", "epic", "literary",
    ]:
        if token.lower() in base.lower():
            hints.append(token)
    questions = latest_turn.get("questions") if isinstance(latest_turn, dict) else []
    return {
        "summary": base or "No style summary yet. Start a session and clarify voice, tone, and pacing.",
        "tags": hints[:8],
        "questions": questions[:4] if isinstance(questions, list) else [],
    }
def _build_model_overview(
    provider_specs: Dict[str, ProviderSpec],
    cred_map: Dict[str, ApiCredential],
    custom_providers: List[ProviderConfig],
    language: str,
) -> Dict:
    providers = []
    for slug, spec in provider_specs.items():
        credential = cred_map.get(slug)
        providers.append(
            {
                "slug": slug,
                "label": spec.label or slug,
                "base_url": spec.base_url,
                "model": spec.model,
                "wire_api": spec.wire_api,
                "kind": spec.kind,
                "has_key": bool(credential or model_connections.env_key(slug,spec.kind)),
                "key_hint": credential.key_hint if credential else "Not saved",
                "is_custom": any((row.slug or "") == slug for row in custom_providers),
            }
        )
    return {
        "providers": providers,
        "modelless_mode": _modelless_mode_enabled(),
        "notice": _modelless_notice(language) if _modelless_mode_enabled() else "",
    }



def _build_env_overview() -> List[Dict[str, str]]:
    keys = [
        "WEB_EXECUTION_MODE",
        "WEB_CLAW_MAX_STEPS",
        "WEB_MAX_ITERATIONS",
        "WEB_MAX_TOTAL_ITERATIONS",
        "WEB_FAST_MODE",
        "WEB_ENABLE_EVALUATOR",
        "WEB_ENABLE_RAG",
        "WEB_ENABLE_STATIC_KB",
        "APP_DATABASE_URL",
    ]
    overview = []
    for key in keys:
        overview.append({"key": key, "value": os.getenv(key, "") or "(default)"})
    return overview


def _console_context(request: Request, db: Session, user: User, active_nav: str) -> Dict:
    ui_language = _ui_language(request)


    provider_specs = _provider_specs_for_user(db, user.id)
    provider_list = list(provider_specs.keys())
    custom_providers = _list_custom_providers(db, user.id)
    creds = db.execute(select(ApiCredential).where(ApiCredential.user_id == user.id)).scalars().all()
    cred_map = {c.provider: c for c in creds}
    book = current_book.get()
    book_id = book.id if book else None
    jobs = _load_console_jobs(db, user.id, book_id)
    idea_sessions = _load_console_sessions(db, user.id, book_id)
    compose_new = request.query_params.get("new", "").strip().lower() in {"1", "true", "yes", "on"}
    requested_session_id = request.query_params.get("session_id", "").strip()
    active_session = idea_sessions[0] if book and idea_sessions and not compose_new else None
    if requested_session_id.isdigit():
        active_session = db.scalar(select(IdeaCopilotSession).where(IdeaCopilotSession.id == int(requested_session_id), IdeaCopilotSession.user_id == user.id, IdeaCopilotSession.book_id == book_id))
    chat_session = None if compose_new else active_session
    session_state = load_state(chat_session.conversation_json) if chat_session else {
        "messages": [],
        "refined_idea": "",
        "round": 0,
        "preferred_language": ui_language,
        "source_language": ui_language,
        "ui_language": ui_language,
        "translation_mode": "english",
    }
    if not session_state.get("ui_language"):
        session_state["ui_language"] = ui_language

    latest_turn = latest_assistant_turn(session_state) if chat_session else {
        "role": "assistant",
        "analysis": "Welcome to the Claw console. Tell me your goals, style, and constraints, and I will decide the next step dynamically.",
        "questions": [
            "What writing or research task matters most right now?",
            "Which capabilities should Claw prioritize first?",
        ],
        "readiness": 0,
        "ready_hint": "Start an idea-refinement session first so Claw can begin iterating continuously.",
    }
    session_job = None
    if chat_session and chat_session.final_job_id:
        session_job = _session_job(db, chat_session)
    # An ideation chat can use the last manuscript even before it starts a run.
    if book and (not session_job or not session_job.run_id):
        session_job = db.scalar(select(GenerationJob).where(GenerationJob.user_id == user.id, GenerationJob.book_id == book_id, GenerationJob.run_id != "").order_by(GenerationJob.id.desc()).limit(1))
    manuscript_run = request.query_params.get("manuscript_run") if active_nav == "manuscript_read" else None
    if manuscript_run:
        session_job = _job_for_run_id(db, user.id, manuscript_run)
        if not session_job or not book or session_job.book_id != book.id:
            raise HTTPException(404, "Manuscript not found in this book")
    run_id = session_job.run_id if session_job and session_job.run_id else ""
    memory_key = book.memory_key if book else run_id
    memory_overview = _build_memory_overview(memory_key) if memory_key else {
        "index": _empty_memory_index(),
        "counts": {key: 0 for key in ("texts", "outlines", "characters", "world_settings", "plot_points", "fact_cards")},
        "claw_counts": {bank: 0 for bank in MemorySystem.CLAW_BANKS},
        "latest_claw": {},
        "index_path": "",
    }
    workspace_files = _build_workspace_files(run_id)
    agent_catalog = _build_agent_catalog()
    capability_catalog = _build_capability_catalog(db, user.id, ui_language)
    story_assets = _build_story_assets(memory_overview, chat_session, session_job) if book or chat_session else _empty_story_assets()
    if book:
        story_assets["project_name"] = book.title
    storyboard_view = _build_storyboard_view(story_assets, ui_language)
    character_cards = _build_character_cards(story_assets, ui_language)
    world_view = _build_world_cards(story_assets, ui_language)
    memory_bank_cards = _build_memory_bank_cards(memory_overview, ui_language)
    memory_bank_groups = _build_memory_bank_groups_from_index(memory_overview.get("index", {}), ui_language)
    workspace_bootstrap = _build_workspace_bootstrap(
        story_assets,
        memory_overview,
        memory_bank_groups,
        ui_language,
        download_job_id=session_job.id if session_job else 0,
    )
    workspace_bootstrap["memory_target"] = "book-memory" if book else run_id
    agent_runtime = _build_agent_runtime(_build_agent_catalog(), capability_catalog, story_assets, memory_overview, ui_language)
    style_profile = _build_style_profile(chat_session, latest_turn)
    model_overview = _build_model_overview(provider_specs, cred_map, custom_providers, ui_language)
    mcp_surface = _build_mcp_surface(capability_catalog, len(provider_list), memory_bank_cards, ui_language)
    env_overview = _build_env_overview()
    status_summary = {
        "api_health": (
            "model-free mode"
            if _modelless_mode_enabled()
            else ("healthy" if provider_list else "not configured")
        ),
        "agent_mode": settings.execution_mode,
        "running_jobs": len([job for job in jobs if job.status == "running"]),
        "tool_count": len([key for key, value in memory_overview.get("claw_counts", {}).items() if value]),
        "active_sessions": len([session for session in idea_sessions if session.status not in {"canceled", "deleted"}]),
        "session_label": (
            f"#{chat_session.id}" if chat_session else "None selected"
        ),
        "session_status": _session_status_label(ui_language, chat_session.status) if chat_session else "New",
        "claw_max_steps": settings.claw_max_steps,
        "memory_count": sum(memory_overview.get("counts", {}).values()),
        "chapter_outputs": len(story_assets.get("chapter_outputs") or []),
        "plot_points": len(story_assets.get("plot_points") or []),
        "characters": len(character_cards),
        "world_facts": len(story_assets.get("world_settings") or []),
    }
    current_path = _current_public_path(request)

    return {
        "request": request,
        "user": user,
        "active_nav": active_nav,
        "ui_language": ui_language,
        "current_path": current_path,
        "ui_texts": _console_texts(ui_language),
        "jobs": jobs,
        "idea_sessions": idea_sessions,
        "active_session": active_session,
        "chat_session": chat_session,
        "compose_new": compose_new,
        "session_state": session_state,
        "messages": list(session_state.get("messages") or []),
        "latest_turn": latest_turn,
        "cred_map": cred_map,
        "providers": provider_list,
        "provider_specs": provider_specs,
        "custom_providers": custom_providers,
        "modelless_mode": _modelless_mode_enabled(),
        "modelless_notice": _modelless_notice(ui_language) if _modelless_mode_enabled() else "",
        "default_provider": settings.default_provider if settings.default_provider in provider_specs else (provider_list[0] if provider_list else ""),
        "workspace_files": workspace_files,
        "memory_overview": memory_overview,
        "memory_bank_cards": memory_bank_cards,
        "memory_bank_groups": memory_bank_groups,
        "workspace_bootstrap": workspace_bootstrap,
        "agent_catalog": agent_catalog,
        "agent_runtime": agent_runtime,
        "session_status_text": lambda value: _session_status_label(ui_language, value),
        "capability_catalog": capability_catalog,
        "story_assets": story_assets,
        "storyboard_view": storyboard_view,
        "character_cards": character_cards,
        "world_view": world_view,
        "style_profile": style_profile,
        "model_overview": model_overview,
        "mcp_surface": mcp_surface,
        "env_overview": env_overview,
        "status_summary": status_summary,
        "claw_config": {"execution_mode": settings.execution_mode, "max_steps": settings.claw_max_steps},
        "message": request.query_params.get("message", ""),
        "error": request.query_params.get("error", ""),
    }


def _console_page_context(request: Request, db: Session, user: User, active_nav: str, **extra: object) -> Dict:
    context = _console_context(request, db, user, active_nav=active_nav)
    context.update(extra)
    return context


def _idea_session_payload(session: IdeaCopilotSession, state: Dict, provider_label: str) -> Dict[str, object]:
    latest_turn = latest_assistant_turn(state)
    return {
        "session_id": session.id,
        "status": session.status,
        "final_job_id": session.final_job_id,
        "provider_label": provider_label,
        "round_count": int(session.round_count or 0),
        "readiness_score": int(session.readiness_score or 0),
        "refined_idea": str(state.get("refined_idea") or session.refined_idea or session.original_idea or ""),
        "messages": list(state.get("messages") or []),
        "latest_turn": latest_turn,
        "reply_pending": bool(state.get("reply_pending") or False),
        "reply_error": str(state.get("reply_error") or ""),
        "updated_at": session.updated_at.isoformat() if getattr(session, "updated_at", None) else "",
    }


def _looks_like_generation_intent(message: str) -> bool:
    text = str(message or "").strip().lower()
    if not text:
        return False

    negative_markers = [
        "not yet",
        "don't start",
        "do not start",
        "don't generate",
        "do not generate",
        "wait before generating",
    ]
    if any(marker in text for marker in negative_markers):
        return False

    positive_markers = [
        "start generating",
        "start writing",
        "generate now",
        "go ahead",
        "proceed",
        "begin the job",
        "start the job",
        "start the run",
        "write now",
    ]
    if any(marker in text for marker in positive_markers):
        return True

    positive_patterns = [
        r"go ahead and (write|generate|start)",
        r"finish (the )?(novel|story|draft)",
        r"enter (execution|writing) mode",
        r"you can (start|write|generate) now",
        r"complete (this )?(novel|story|draft)",
    ]
    return any(re.search(pattern, text, re.I) for pattern in positive_patterns)


def _extract_json_object(text: str) -> Optional[str]:
    s = (text or "").strip()
    if not s:
        return None
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(s)):
        ch = s[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : idx + 1]
    return None


def _history_excerpt_for_generation(messages: List[Dict[str, Any]], limit: int = 8) -> str:
    rows: List[str] = []
    for item in messages[-limit:]:
        role = str(item.get("role") or "").strip().lower()
        if role == "user":
            rows.append(f"user: {str(item.get('content') or '').strip()[:300]}")
        elif role == "assistant":
            analysis = str(item.get("analysis") or "").strip()
            ready_hint = str(item.get("ready_hint") or "").strip()
            rows.append(f"assistant: {(analysis or ready_hint)[:300]}")
    return "\n".join(rows) if rows else "(no recent conversation)"


def _extract_requested_chapter_count(message: str) -> int:
    text = str(message or "").strip()
    if not text:
        return 0
    digit_match = re.search(r"(\d+)\s*chapters?\b", text, re.I)
    if digit_match:
        try:
            return max(0, int(digit_match.group(1)))
        except Exception:
            return 0
    en_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    word_match = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+chapters?\b", text, re.I)
    if word_match:
        return en_words.get(word_match.group(1).lower(), 0)
    return 0


def _normalize_generation_intent_payload(payload: Dict[str, Any], latest_user_reply: str) -> Dict[str, Any]:
    preferences = normalize_generation_preferences(
        {
            "generation_scope": payload.get("generation_scope"),
            "requested_chapters": payload.get("requested_chapters"),
            "chapter_pause_mode": payload.get("chapter_pause_mode"),
            "user_request": payload.get("user_request") or latest_user_reply,
        }
    )
    return {
        "should_start_generation": bool(payload.get("should_start_generation") or False),
        **preferences,
    }


def _heuristic_generation_intent(latest_user_reply: str) -> Dict[str, Any]:
    text = str(latest_user_reply or "").strip()
    lower = text.lower()
    requested_chapters = _extract_requested_chapter_count(text)

    negative_markers = [
        "not yet",
        "don't start", "do not start", "don't generate", "do not generate",
    ]
    all_markers = [
        "the whole story", "whole book", "all chapters", "to the end", "run to end",
    ]
    chapter_by_chapter_markers = [
        "chapter by chapter",
        "ask every chapter", "pause every chapter",
    ]
    run_to_end_markers = [
        "run through", "keep going", "continue to the end",
    ]

    should_start = _looks_like_generation_intent(text) or bool(requested_chapters) or any(marker in lower for marker in all_markers)
    if any(marker in lower for marker in negative_markers):
        should_start = False

    payload: Dict[str, Any] = {
        "should_start_generation": should_start,
        "generation_scope": "auto",
        "requested_chapters": requested_chapters,
        "chapter_pause_mode": "manual_each_chapter",
        "user_request": text[:400],
    }
    if any(marker in lower for marker in chapter_by_chapter_markers):
        payload["generation_scope"] = "chapter_by_chapter"
        payload["chapter_pause_mode"] = "manual_each_chapter"
    elif any(marker in lower for marker in all_markers):
        payload["generation_scope"] = "all"
        payload["chapter_pause_mode"] = "run_to_end"
        payload["requested_chapters"] = 0
    elif requested_chapters > 0:
        payload["generation_scope"] = "limited"
        payload["chapter_pause_mode"] = "run_to_end"

    if any(marker in lower for marker in run_to_end_markers):
        payload["chapter_pause_mode"] = "run_to_end"
    return _normalize_generation_intent_payload(payload, text)


def _model_generation_intent(
    *,
    state: Dict[str, Any],
    original_idea: str,
    latest_user_reply: str,
    provider_slug: str,
    provider_base_url: str,
    provider_model: str,
    provider_wire_api: str,
    provider_kind: str = "custom",
    api_key: str,
) -> Optional[Dict[str, Any]]:
    if not api_key:
        return None
    cfg = Config(require_api_key=False)
    cfg.language = "en"
    cfg.llm_provider = provider_slug
    cfg.api_key = api_key
    cfg.api_base_url = provider_base_url
    cfg.model_name = provider_model
    cfg.model_kind = provider_kind
    cfg.wire_api = provider_wire_api or "chat"
    cfg.temperature = 0.1
    cfg.max_tokens = 600
    client = LLMClient(cfg)
    system_prompt = """You detect generation-start intent for a writing copilot.
Return strict JSON only:
{
  "should_start_generation": false,
  "generation_scope": "auto",
  "requested_chapters": 0,
  "chapter_pause_mode": "manual_each_chapter",
  "user_request": ""
}
Rules:
- should_start_generation is true only when the latest user reply means the real writing run should start now.
- generation_scope must be one of auto, all, limited, chapter_by_chapter.
- requested_chapters is 0 unless the user clearly asks for a concrete chapter count.
- chapter_pause_mode must be manual_each_chapter or run_to_end.
- If uncertain, keep should_start_generation=false and generation_scope=auto.
"""
    user_prompt = f"""[original idea]
{original_idea[:1800]}

[current brief]
{str(state.get('refined_idea') or '')[:2400]}

[recent conversation]
{_history_excerpt_for_generation(list(state.get("messages") or []))}

[latest user reply]
{latest_user_reply}
"""
    try:
        raw = client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        block = _extract_json_object(raw or "")
        if not block:
            return None
        parsed = json.loads(block)
        if not isinstance(parsed, dict):
            return None
        return _normalize_generation_intent_payload(parsed, latest_user_reply)
    except Exception:
        return None


def _analyze_generation_intent(
    *,
    state: Dict[str, Any],
    session: IdeaCopilotSession,
    latest_user_reply: str,
    spec: ProviderSpec,
    api_key: str,
) -> Dict[str, Any]:
    messages = state.get("messages") or []
    if requested_questions(state, latest_user_reply) or (messages and messages[-1].get("selected_options")):
        return {"should_start_generation": False}
    model_result = _model_generation_intent(
        state=state,
        original_idea=session.original_idea,
        latest_user_reply=latest_user_reply,
        provider_slug=spec.slug,
        provider_base_url=spec.base_url,
        provider_model=spec.model,
        provider_wire_api=spec.wire_api,
        provider_kind=spec.kind,
        api_key=api_key,
    )
    if model_result:
        return model_result
    return _heuristic_generation_intent(latest_user_reply)


def _generation_pref_summary(preferences: Dict[str, Any], preferred_language: str) -> Dict[str, str]:
    prefs = normalize_generation_preferences(preferences)
    requested = int(prefs.get("requested_chapters") or 0)
    scope = str(prefs.get("generation_scope") or "auto")
    pause_mode = str(prefs.get("chapter_pause_mode") or "manual_each_chapter")
    if scope == "all":
        plan = "I will run through the planned chapters continuously."
    elif scope == "limited" and requested > 0:
        plan = f"I will generate the next {requested} chapters first."
    elif scope == "chapter_by_chapter":
        plan = "I will write chapter by chapter and wait after each chapter."
    else:
        plan = "I will start the writing run from the current brief."
    pause_hint = (
        "The run will keep moving until it reaches your requested scope."
        if pause_mode == "run_to_end"
        else "After each chapter, a checkpoint will appear in the current chat so you can revise direction, add memory, or stop."
    )
    return {
        "analysis": f"Execution mode is starting now. {plan}",
        "ready_hint": pause_hint,
    }


def _session_job(db: Session, session: IdeaCopilotSession) -> Optional[GenerationJob]:
    return db.scalar(select(GenerationJob).where(
        GenerationJob.id == session.final_job_id,
        GenerationJob.user_id == session.user_id,
        GenerationJob.book_id == session.book_id,
    ))


def _start_generation_from_idea_session(db: Session, session: IdeaCopilotSession, user_id: int) -> GenerationJob:
    if session.user_id != user_id:
        raise ValueError("Session belongs to a different user")
    if session.final_job_id:
        existing = _session_job(db, session)
        if existing:
            return existing

    state = load_state(session.conversation_json)
    final_idea = build_generation_idea(session.original_idea, state)
    book_notes = _book_notes(session.book_id)
    if book_notes:
        final_idea += "\n\n[Book notes]\n" + book_notes
    if not final_idea.strip():
        raise ValueError("Final idea is empty")

    # Claim the session before allocating a job, in the same transaction.
    claimed = db.execute(update(IdeaCopilotSession).where(
        IdeaCopilotSession.id == session.id,
        IdeaCopilotSession.status == "active",
        IdeaCopilotSession.final_job_id.is_(None),
        IdeaCopilotSession.conversation_json == session.conversation_json,
    ).values(status="confirmed")).rowcount
    if not claimed:
        db.rollback()
        db.refresh(session)
        if session.final_job_id:
            return _session_job(db, session)
        raise ValueError("Session changed; refresh before confirming generation")

    job = _create_generation_job(db, user_id, session.provider, final_idea, commit=False, book_id=session.book_id, snapshot=session.model_selection)
    session.status = "confirmed"
    session.final_job_id = job.id
    session.refined_idea = str(state.get("refined_idea") or session.refined_idea or session.original_idea)
    session.round_count = int(state.get("round", 0) or 0)
    session.finished_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(job)
    _remember_session_turn(session, state, confirmed=True, final_topic=final_idea)

    _start_generation_worker(job.id)
    return job


def _cancel_generation_job(job: Optional[GenerationJob], reason: str = "[user] canceled manually.") -> bool:
    if not job or job.status not in {"queued", "running"}:
        return False

    run_id = str(job.run_id or "").strip()
    if run_id:
        run_dir = _resolve_run_dir(run_id)
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "cancel.flag").write_text("1", encoding="utf-8")
        except Exception:
            pass

    job.status = "canceled"
    prefix = (job.error_message + "\n") if job.error_message else ""
    job.error_message = prefix + reason
    now = datetime.now(timezone.utc)
    job.finished_at = now
    job.updated_at = now
    return True


def _detach_job_from_sessions(db: Session, job: GenerationJob) -> None:
    related_sessions = (
        db.execute(
            select(IdeaCopilotSession).where(IdeaCopilotSession.final_job_id == job.id, IdeaCopilotSession.book_id == job.book_id, IdeaCopilotSession.user_id == job.user_id)
        )
        .scalars()
        .all()
    )
    for session in related_sessions:
        session.final_job_id = None


def _save_reply_if_current(db: Session, session: IdeaCopilotSession, original_json: str, state: Dict, readiness: int) -> bool:
    changed = db.execute(update(IdeaCopilotSession).where(
        IdeaCopilotSession.id == session.id,
        IdeaCopilotSession.status == "active",
        IdeaCopilotSession.conversation_json == original_json,
    ).values(
        conversation_json=dump_state(state),
        refined_idea=str(state.get("refined_idea") or session.original_idea),
        round_count=int(state.get("round", 0) or 0),
        readiness_score=readiness,
    )).rowcount
    db.commit()
    if changed:
        db.refresh(session)
    return bool(changed)


def _run_idea_copilot_reply_worker(session_id: int) -> None:
    with SessionLocal() as db:
        session = db.get(IdeaCopilotSession, session_id)
        if not session or session.status != "active":
            return
        user = db.get(User, session.user_id)
        if not user:
            return

        state = load_state(session.conversation_json)
        original_json = session.conversation_json
        if not state.get("reply_pending"):
            return
        messages = list(state.get("messages") or [])
        latest_user_reply = ""
        for item in reversed(messages):
            if str(item.get("role") or "") == "user":
                latest_user_reply = str(item.get("content") or "")
                break

        provider_specs = _provider_specs_for_user(db, user.id)
        spec = provider_specs.get(session.provider)
        if not spec:
            state["reply_pending"] = False
            state["reply_error"] = "Provider is no longer available"
            _save_reply_if_current(db, session, original_json, state, session.readiness_score)
            return

        try:
            spec = model_connections.select_model(spec, snapshot=session.model_selection)
            api_key = ""
            try:
                api_key = _provider_api_key(db, user.id, session.provider)
            except Exception:
                api_key = ""

            with model_accounting.recording(session.book_id, f"chat {session.id} intent"):
                generation_intent = _analyze_generation_intent(
                    state=state,
                    session=session,
                    latest_user_reply=latest_user_reply,
                    spec=spec,
                    api_key=api_key,
                )
            state["generation_preferences"] = merge_generation_preferences(state, generation_intent)

            if generation_intent.get("should_start_generation"):
                preferred_language = "en"
                summary = _generation_pref_summary(state.get("generation_preferences") or {}, preferred_language)
                turn = {
                    "role": "assistant",
                    "analysis": summary["analysis"],
                    "refined_idea": str(state.get("refined_idea") or session.refined_idea or session.original_idea or ""),
                    "questions": [],
                    "readiness": 100,
                    "ready_hint": summary["ready_hint"],
                    "language": preferred_language,
                    "style_targets": [],
                    "memory_targets": [],
                }
                state = append_assistant_turn(state, turn)
                state["reply_pending"] = False
                state["reply_error"] = ""
                if not _save_reply_if_current(db, session, original_json, state, 100):
                    return
                _remember_session_turn(session, state, turn=turn)
                _start_generation_from_idea_session(db, session, user.id)
                return

            api_key = api_key or _provider_api_key(db, user.id, session.provider)
            agent = IdeaCopilotAgent(spec, api_key)
            with model_accounting.recording(session.book_id, f"chat {session.id}"):
                turn = agent.generate_turn(
                    original_idea=session.original_idea,
                    state={**state, "book_notes": _book_notes(session.book_id)},
                    latest_user_reply=latest_user_reply,
                )
            state = append_assistant_turn(state, turn)
            state["reply_pending"] = False
            state["reply_error"] = ""
            if not _save_reply_if_current(db, session, original_json, state, int(turn.get("readiness", 0) or 0)):
                return
            _remember_session_turn(session, state, turn=turn)
        except Exception as exc:
            db.rollback()
            state = load_state(original_json)
            state["reply_pending"] = False
            state["reply_error"] = str(exc)
            _save_reply_if_current(db, session, original_json, state, session.readiness_score)


def _start_idea_copilot_reply_worker(session_id: int) -> None:
    with SessionLocal() as db:
        book_id = db.scalar(select(IdeaCopilotSession.book_id).where(IdeaCopilotSession.id == session_id))
    start_book_thread(book_id, _run_idea_copilot_reply_worker, session_id)



def _create_generation_job(
    db: Session,
    user_id: int,
    provider: str,
    idea: str,
    *,
    commit: bool = True,
    model: str = "",
    snapshot: str | None = None,
    book_id: int | None = None,
) -> GenerationJob:
    spec = _chosen(_all_provider_specs_for_user(db,user_id).get(provider), model, snapshot)
    job = GenerationJob(
        model_selection=model_connections.snapshot(spec),
        user_id=user_id,
        book_id=book_id,
        provider=provider,
        idea=idea,
        status="queued",
    )
    db.add(job)
    if commit:
        db.commit()
        db.refresh(job)
    else:
        db.flush()
    return job


def _start_generation_worker(job_id: int) -> None:
    with SessionLocal() as db:
        book_id = db.scalar(select(GenerationJob.book_id).where(GenerationJob.id == job_id))
    start_book_thread(book_id, run_generation_job, job_id)


def _extract_bearer_token(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.lower().startswith("bearer "):
        return raw[7:].strip()
    return ""


def _require_agent_token(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    expected = str(settings.agent_api_key or "").strip()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent API is disabled. Set APP_AGENT_API_KEY to enable /api/v1 endpoints.",
        )
    supplied = str(x_api_key or "").strip() or _extract_bearer_token(authorization)
    if not supplied or not secrets.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid agent API key")


def _agent_user(db: Session) -> User:
    email = settings.agent_user_email
    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if user:
        return user
    user = User(email=email, password_hash="agent-api")
    db.add(user)
    try:
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        user = db.execute(select(User).where(User.email == email)).scalar_one()
    return user


async def _json_payload(request: Request) -> Dict[str, object]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Expected JSON body") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Expected JSON object")
    return payload


def _payload_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _payload_text(payload: Dict[str, object], *names: str) -> str:
    for name in names:
        value = payload.get(name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _select_agent_provider(db: Session, user_id: int, raw_provider: str) -> str:
    provider_specs = _all_provider_specs_for_user(db, user_id)
    provider = normalize_slug(raw_provider or settings.default_provider)
    if not provider and provider_specs:
        provider = next(iter(provider_specs.keys()))
    if provider not in provider_specs:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported provider: {provider or raw_provider}")
    return provider


def _build_agent_idea(payload: Dict[str, object]) -> str:
    idea = _payload_text(payload, "premise", "brief", "idea", "story", "summary")
    if not idea:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing story premise. Use premise, brief, or idea.")

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    metadata_lines: List[str] = []
    for key in (
        "title",
        "genre",
        "tone",
        "audience",
        "style",
        "constraints",
        "preferred_language",
        "source_language",
        "requested_chapters",
        "chapter_pause_mode",
        "generation_scope",
    ):
        value = payload.get(key, metadata.get(key))
        if value is None:
            continue
        text = str(value).strip()
        if text:
            metadata_lines.append(f"{key}: {text}")

    if not metadata_lines:
        return idea
    return f"{idea}\n\nAgent API metadata:\n" + "\n".join(metadata_lines)


def _upsert_api_key(db: Session, user_id: int, provider: str, api_key: str) -> ApiCredential:
    key = str(api_key or "").strip()
    if not key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="api_key cannot be empty")

    cred = db.execute(
        select(ApiCredential).where(
            ApiCredential.user_id == user_id,
            ApiCredential.provider == provider,
        )
    ).scalar_one_or_none()

    encrypted = encrypt_api_key(key)
    if cred:
        cred.encrypted_key = encrypted
        cred.key_hint = _mask_hint(key)
    else:
        cred = ApiCredential(
            user_id=user_id,
            provider=provider,
            encrypted_key=encrypted,
            key_hint=_mask_hint(key),
        )
        db.add(cred)
    db.commit()
    db.refresh(cred)
    return cred


def _agent_job(db: Session, user_id: int, story_id: int) -> GenerationJob:
    job = db.get(GenerationJob, story_id)
    if not job or job.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Story not found")
    return job


def _start_agent_job(db: Session, user_id: int, job: GenerationJob) -> GenerationJob:
    if _modelless_mode_enabled():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Generation is disabled because WEB_MODELLESS_MODE=1. Set WEB_MODELLESS_MODE=0 before starting agent runs.",
        )
    provider_specs = _provider_specs_for_user(db, user_id)
    if job.provider not in provider_specs:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported provider: {job.provider}")
    try:
        _provider_api_key(db, user_id, job.provider)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    if job.status == "queued":
        _start_generation_worker(job.id)
    elif job.status in {"running", "succeeded"}:
        return job
    else:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Cannot start story with status {job.status}")
    db.refresh(job)
    return job


def _job_progress_payload(job: GenerationJob) -> Dict[str, object]:
    worker_log_text = ""
    progress_log_text = ""
    run_dir: Optional[Path] = None
    chapters: List[Dict] = []
    if job.run_id:
        run_dir = _resolve_run_dir(job.run_id)
        worker_log_text = _tail_text(run_dir / "worker.log")
        progress_log_text = _tail_text(run_dir / "progress.log")
        chapters = _load_chapter_outputs(run_dir, include_content=False)
    try:
        progress_snapshot = _build_progress_snapshot(job, run_dir, worker_log_text, progress_log_text)
    except Exception:
        progress_snapshot = _default_progress_snapshot(_job_language(job))
    return {
        "progress_snapshot": progress_snapshot,
        "chapter_count": len(chapters),
        "worker_log_tail": worker_log_text,
        "progress_log_tail": progress_log_text,
    }


def _api_job_payload(job: GenerationJob, *, include_logs: bool = False) -> Dict[str, object]:
    progress = _job_progress_payload(job)
    payload = {
        "story_id": job.id,
        "job_id": job.id,
        "provider": job.provider,
        "model": json.loads(job.model_selection)["model"] if job.model_selection else None,
        "status": job.status,
        "run_id": job.run_id or "",
        "premise": job.idea,
        "result_excerpt": job.result_excerpt or "",
        "error_message": job.error_message or "",
        "output_path": job.output_path or "",
        "created_at": job.created_at.isoformat() if job.created_at else "",
        "updated_at": job.updated_at.isoformat() if job.updated_at else "",
        "finished_at": job.finished_at.isoformat() if job.finished_at else "",
        "chapter_count": progress["chapter_count"],
        "progress_snapshot": progress["progress_snapshot"],
    }
    if include_logs:
        payload["worker_log_tail"] = progress["worker_log_tail"]
        payload["progress_log_tail"] = progress["progress_log_tail"]
    return payload


def _api_chapters_payload(job: GenerationJob, include_content: bool = True) -> List[Dict[str, object]]:
    if not job.run_id:
        return []
    chapters = _load_chapter_outputs(_resolve_run_dir(job.run_id), include_content=include_content)
    if include_content:
        return chapters
    return [{k: v for k, v in item.items() if k != "content"} for item in chapters]


@app.get("/api/v1/health")
def api_v1_health(_: None = Depends(_require_agent_token)):
    return {"ok": True, "modelless_mode": _modelless_mode_enabled()}


@app.get("/api/v1/providers")
def api_v1_providers(_: None = Depends(_require_agent_token), db: Session = Depends(get_db)):
    user = _agent_user(db)
    provider_specs = _all_provider_specs_for_user(db, user.id)
    rows = db.execute(select(ApiCredential).where(ApiCredential.user_id == user.id)).scalars().all()
    cred_map = {row.provider: row for row in rows}
    providers = []
    for slug, spec in provider_specs.items():
        configured = bool(spec.kind == "local" or slug in cred_map or model_connections.env_key(slug, spec.kind))
        providers.append(
            {
                "slug": slug,
                "label": spec.label,
                "base_url": spec.base_url,
                "model": spec.model,
                "wire_api": spec.wire_api,
                "has_api_key": configured,
                "key_hint": cred_map[slug].key_hint if slug in cred_map else ("env" if model_connections.env_key(slug, spec.kind) else ""),
            }
        )
    return {"modelless_mode": _modelless_mode_enabled(), "default_provider": settings.default_provider, "providers": providers}


def _clear_moved_key(db, row, base_url):
    if row.base_url.rstrip("/") != base_url.rstrip("/"):
        saved = db.scalar(select(ApiCredential).where(ApiCredential.user_id == row.user_id, ApiCredential.provider == row.slug))
        if saved:
            db.delete(saved)


@app.get("/api/v1/providers/{provider}/models")
def api_provider_models(provider: str, refresh: bool = False, _: None = Depends(_require_agent_token), db: Session = Depends(get_db)):
    return _model_catalog(db, _agent_user(db).id, provider, refresh)


@app.post("/api/v1/providers")
async def api_v1_save_provider(request: Request, _: None = Depends(_require_agent_token), db: Session = Depends(get_db)):
    user = _agent_user(db)
    payload = await _json_payload(request)
    slug = normalize_slug(_payload_text(payload, "slug", "provider"))
    base_url = _payload_text(payload, "base_url")
    model = _payload_text(payload, "model")
    label = _payload_text(payload, "label") or slug.upper()
    wire_api = normalize_wire_api(_payload_text(payload, "wire_api", "wire_mode") or "chat")
    if not is_valid_slug(slug):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid provider slug")
    if not base_url or not model:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="base_url and model are required")

    row = db.execute(
        select(ProviderConfig).where(
            ProviderConfig.user_id == user.id,
            ProviderConfig.slug == slug,
        )
    ).scalar_one_or_none()
    kind = _payload_text(payload, "kind") or (row.kind if row else "custom")
    if kind not in {"local", "custom"} or not base_url.startswith(("http://","https://")):
        raise HTTPException(422, "Enter an HTTP(S) base URL and a local or custom connection type.")
    if not row and slug in get_provider_specs(settings):
        raise HTTPException(409, "Provider name is reserved.")
    if row:
        _clear_moved_key(db,row,base_url)
        row.kind = kind
        row.label = label
        row.base_url = base_url
        row.model = model
        row.wire_api = wire_api
    else:
        row = ProviderConfig(user_id=user.id, slug=slug, kind=kind, label=label, base_url=base_url, model=model, wire_api=wire_api)
        db.add(row)
    db.commit()

    api_key = _payload_text(payload, "api_key")
    if api_key:
        _upsert_api_key(db, user.id, slug, api_key)
    return {"ok": True, "provider": {"slug": slug, "label": label, "base_url": base_url, "model": model, "wire_api": wire_api}}


@app.post("/api/v1/providers/{provider}/api-key")
async def api_v1_save_provider_key(provider: str, request: Request, _: None = Depends(_require_agent_token), db: Session = Depends(get_db)):
    user = _agent_user(db)
    provider = _select_agent_provider(db, user.id, provider)
    payload = await _json_payload(request)
    api_key = _payload_text(payload, "api_key", "key", "token")
    cred = _upsert_api_key(db, user.id, provider, api_key)
    return {"ok": True, "provider": provider, "key_hint": cred.key_hint}


@app.post("/api/v1/stories")
async def api_v1_create_story(request: Request, _: None = Depends(_require_agent_token), db: Session = Depends(get_db)):
    user = _agent_user(db)
    payload = await _json_payload(request)
    provider = _select_agent_provider(db, user.id, _payload_text(payload, "provider"))
    idea = _build_agent_idea(payload)
    book = new_book(db, user.id, _payload_text(payload, "title") or idea)
    job = _create_generation_job(db, user.id, provider, idea, book_id=book.id, model=_payload_text(payload, "model"))
    if _payload_bool(payload.get("start"), default=_payload_bool(payload.get("auto_start"), default=False)):
        job = _start_agent_job(db, user.id, job)
    return JSONResponse(_api_job_payload(job), status_code=status.HTTP_201_CREATED)


@app.get("/api/v1/stories/{story_id}")
def api_v1_get_story(
    story_id: int,
    include_logs: bool = False,
    _: None = Depends(_require_agent_token),
    db: Session = Depends(get_db),
):
    user = _agent_user(db)
    job = _agent_job(db, user.id, story_id)
    return _api_job_payload(job, include_logs=include_logs)


@app.post("/api/v1/stories/{story_id}/start")
def api_v1_start_story(story_id: int, _: None = Depends(_require_agent_token), db: Session = Depends(get_db)):
    user = _agent_user(db)
    job = _agent_job(db, user.id, story_id)
    job = _start_agent_job(db, user.id, job)
    return _api_job_payload(job)


@app.post("/api/v1/stories/{story_id}/cancel")
def api_v1_cancel_story(story_id: int, _: None = Depends(_require_agent_token), db: Session = Depends(get_db)):
    user = _agent_user(db)
    job = _agent_job(db, user.id, story_id)
    canceled = _cancel_generation_job(job, reason=f"[agent-api] canceled via /api/v1/stories/{story_id}/cancel.")
    if canceled:
        db.commit()
    return {"ok": True, "canceled": canceled, "story": _api_job_payload(job)}


@app.get("/api/v1/stories/{story_id}/chapters")
def api_v1_story_chapters(
    story_id: int,
    include_content: bool = True,
    _: None = Depends(_require_agent_token),
    db: Session = Depends(get_db),
):
    user = _agent_user(db)
    job = _agent_job(db, user.id, story_id)
    chapters = _api_chapters_payload(job, include_content=include_content)
    return {"story_id": job.id, "status": job.status, "run_id": job.run_id or "", "chapter_count": len(chapters), "chapters": chapters}


@app.get("/api/v1/stories/{story_id}/chapters/{chapter_no}")
def api_v1_story_chapter(story_id: int, chapter_no: int, _: None = Depends(_require_agent_token), db: Session = Depends(get_db)):
    user = _agent_user(db)
    job = _agent_job(db, user.id, story_id)
    if not job.run_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No chapter output available yet")
    chapter_path = _latest_chapter_file(_resolve_run_dir(job.run_id), chapter_no)
    if not chapter_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chapter output not found")
    content = chapter_path.read_text(encoding="utf-8", errors="replace")
    return {"story_id": job.id, "chapter": chapter_no, "filename": chapter_path.name, "content": content}


@app.get("/api/v1/stories/{story_id}/export")
def api_v1_story_export(story_id: int, _: None = Depends(_require_agent_token), db: Session = Depends(get_db)):
    user = _agent_user(db)
    job = _agent_job(db, user.id, story_id)
    filename = f"story_{job.id}_{job.run_id or 'no_run'}_export.txt"
    return output_download_response(_resolve_run_dir(job.run_id) if job.run_id else None, job.output_path, filename)


def _to_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _default_progress_snapshot(language: str = "en") -> Dict:
    return {
        "phase": "running",
        "phase_label": "Running",
        "phase_note": "",
        "elapsed_seconds": 0,
        "current_chapter": 0,
        "planned_total": 0,
        "chapter_words": 0,
        "total_words": 0,
        "percent": 0,
        "last_log_at": "",
        "idle_seconds": -1,
        "stalled": False,
        "stall_reason": "",
        "memory_counts": {
            "texts": 0,
            "outlines": 0,
            "characters": 0,
            "world_settings": 0,
            "plot_points": 0,
            "fact_cards": 0,
        },
    }

def _parse_progress_log(progress_log_text: str, language: str = "en") -> Dict:
    snapshot = {
        "current_chapter": 0,
        "planned_total": 0,
        "chapter_words": 0,
        "total_words": 0,
        "phase_note": "",
        "latest_event": "",
        "latest_event_label": "",
        "memory_counts": {k: 0 for k in MEMORY_COUNT_KEYS},
    }
    if not progress_log_text:
        return snapshot

    for raw in progress_log_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("chapter="):
            try:
                parts = dict(
                    kv.strip().split("=", 1)
                    for kv in line.split(",")
                    if "=" in kv
                )
                snapshot["current_chapter"] = int(parts.get("chapter", snapshot["current_chapter"]))
                snapshot["planned_total"] = int(parts.get("planned_total", snapshot["planned_total"]))
                snapshot["chapter_words"] = int(parts.get("words", snapshot["chapter_words"]))
            except Exception:
                pass
        elif line.startswith("total_words="):
            try:
                snapshot["total_words"] = int(line.split("=", 1)[1].strip())
            except Exception:
                pass
        elif line.startswith("[event]"):
            try:
                parts = [seg.strip() for seg in line.split("|")]
                if len(parts) < 2:
                    continue

                event_name = parts[1]
                detail_parts: List[str] = []
                chapter_no = 0

                for seg in parts[2:]:
                    if seg.startswith("chapter "):
                        m_ch = re.search(r"chapter\s+(\d+)", seg)
                        if m_ch:
                            chapter_no = int(m_ch.group(1))
                    elif seg:
                        detail_parts.append(seg)

                detail = " | ".join(detail_parts)
                if chapter_no > 0:
                    snapshot["current_chapter"] = max(snapshot["current_chapter"], chapter_no)

                if event_name == "chapter_outline_ready" and snapshot["planned_total"] <= 0:
                    m_plan = re.search(r"count\s*=\s*(\d+)", detail)
                    if m_plan:
                        snapshot["planned_total"] = int(m_plan.group(1))

                if event_name == "memory_snapshot":
                    detail_map = {
                        k.strip(): v.strip()
                        for k, v in (
                            item.split("=", 1)
                            for item in detail.split(",")
                            if "=" in item
                        )
                    }
                    for key in MEMORY_COUNT_KEYS:
                        if key in detail_map:
                            try:
                                snapshot["memory_counts"][key] = int(detail_map[key])
                            except Exception:
                                pass

                label = _event_label(event_name, language)
                formatted_detail = _format_progress_detail(detail, language=language)
                short_detail = _compact_preview(formatted_detail, 160) if formatted_detail else ""
                snapshot["latest_event"] = event_name
                snapshot["latest_event_label"] = label
                snapshot["phase_note"] = f"{label}: {short_detail}" if short_detail else label
            except Exception:
                pass
    return snapshot


def _format_progress_detail(detail: str, language: str = "en") -> str:
    detail = str(detail or "").strip()
    if not detail:
        return ""

    key_labels = {
        "count": "Count",
        "mode": "Mode",
        "words": "Words",
        "min": "Min",
        "max": "Max",
        "target": "Target",
        "topic": "Topic",
        "texts": "Texts",
        "outlines": "Outlines",
        "characters": "Characters",
        "world_settings": "World settings",
        "plot_points": "Plot points",
        "fact_cards": "Fact cards",
        "still_below_min": "Still below minimum",
    }
    mode_labels = {
        "shrink_to_range": "Shrink to range",
        "truncate_fallback": "Truncate fallback",
        "expand_to_min": "Expand to minimum",
        "final_truncate": "Final truncate",
    }

    if "=" not in detail:
        return detail

    parts: List[str] = []
    for item in re.split(r"\s*\|\s*|\s*,\s*", detail):
        if not item:
            continue
        if "=" not in item:
            parts.append(item)
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        label = key_labels.get(key, key.replace("_", " "))
        if key == "mode":
            value = mode_labels.get(value, value)
        parts.append(f"{label}: {value}")
    return " | ".join(parts) if parts else detail


def _render_progress_log(progress_log_text: str, language: str = "en") -> str:
    if not progress_log_text:
        return ""

    rendered: List[str] = []
    for raw in progress_log_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("chapter="):
            try:
                parts = dict(
                    kv.strip().split("=", 1)
                    for kv in line.split(",")
                    if "=" in kv
                )
                rendered.append(
                    f"Chapter {parts.get('chapter', '0')} | Chapter words: {parts.get('words', '0')} | Planned total: {parts.get('planned_total', '0')} | Target: {parts.get('target', '0')} | Min: {parts.get('min', '0')} | Max: {parts.get('max', '0')} | Topic: {parts.get('topic', '')}"
                )
            except Exception:
                rendered.append(line)
            continue
        if line.startswith("total_words="):
            try:
                total = line.split("=", 1)[1].strip()
                rendered.append(f"Total words: {total}")
            except Exception:
                rendered.append(line)
            continue
        if line.startswith("[event]"):
            try:
                parts = [seg.strip() for seg in line.split("|")]
                if len(parts) >= 2:
                    event_name = parts[1]
                    timestamp = parts[0].replace("[event]", "").strip()
                    chapter_no = ""
                    detail_parts: List[str] = []
                    for seg in parts[2:]:
                        if seg.startswith("chapter "):
                            chapter_no = seg.split(" ", 1)[1].strip()
                        elif seg:
                            detail_parts.append(seg)
                    label = _event_label(event_name, language)
                    detail = _format_progress_detail(" | ".join(detail_parts), language)
                    chapter_label = (
                        f"Chapter {chapter_no}"
                        if chapter_no else ""
                    )
                    segments = [item for item in [timestamp, label, chapter_label, detail] if item]
                    rendered.append(" | ".join(segments))
                    continue
            except Exception:
                pass
        rendered.append(line)
    return "\n".join(rendered)


def _job_detail_texts(language: str) -> Dict[str, str]:
    return {
        "page_title": "NovelClaw Run",
        "page_subtitle": "This page shows NovelClaw's live execution state, Claw action trace, chapter outputs, and final result.",
        "job_title": "NovelClaw Run",
        "provider": "Provider",
        "status": "Status",
        "status_queued": "Queued",
        "status_running": "Running",
        "status_succeeded": "Succeeded",
        "status_failed": "Failed",
        "status_canceled": "Canceled",
        "status_unknown": "Unknown",
        "cancel_job": "Cancel this run",
        "job_input": "Current writing brief",
        "run_id": "Run ID",
        "output_path": "Output Path",
        "download_output": "Download full output",
        "download_chapters": "Download all chapters (.zip)",
        "execution_status": "Claw execution status",
        "phase": "Phase",
        "elapsed": "Elapsed",
        "idle": "Idle",
        "current_action": "Current action",
        "chapter_progress": "Chapter progress",
        "chapter_words": "Chapter words",
        "total_words": "Total words",
        "memory": "Dynamic memory",
        "outlines": "Outlines",
        "texts": "Texts",
        "characters": "Characters",
        "worlds": "Worlds",
        "facts": "Facts",
        "live_progress": "Live progress",
        "live_progress_hint": "This shows real worker output, not browser polling logs.",
        "no_worker_log": "No worker log yet.",
        "progress_log": "Progress log",
        "no_progress_log": "No progress log yet.",
        "agent_trace": "Claw action trace",
        "agent_trace_hint": "This turns key progress events into a readable step trace so you can see which capabilities Claw is using.",
        "no_agent_trace": "No Claw events yet.",
        "chapter_results": "Chapter results",
        "chapter_results_hint": "Shows the latest finalized content for each chapter and auto-refreshes while the run is active.",
        "chapter": "Chapter",
        "iteration": "Iteration",
        "download_chapter": "Download chapter",
        "no_chapter_output": "No chapter output yet.",
        "error": "Error",
        "generated_text": "Final result",
        "result": "Result",
        "result_pending": "The run is still active or has not produced a final result yet. Refresh in a few seconds.",
        "back_dashboard": "Back to NovelClaw workspace",
        "back_chat": "Back to current chat",
        "stall_fallback": "Possible stall detected, please check worker logs.",
        "trace_openclaw": "OpenClaw mode",
        "trace_openclaw_detail": "NovelClaw has switched to the native Claw loop instead of seeding a fixed workflow first.",
        "trace_loop": "Claw loop",
        "trace_decision": "Action decision",
        "trace_action": "Action",
        "trace_result": "Result",
        "trace_candidate": "Candidate update",
        "trace_ready": "Ready to finalize",
        "trace_complete": "Claw loop complete",
    }

def _read_status_file(run_dir: Optional[Path]) -> Dict:
    if not run_dir:
        return {}
    path = run_dir / "status.json"
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _infer_phase(worker_log_text: str) -> str:
    if not worker_log_text:
        return "waiting_worker"
    lines = [ln.strip() for ln in worker_log_text.splitlines() if ln.strip()]
    if not lines:
        return "waiting_worker"
    checks = [
        ("Traceback", "error"),
        ("[system] worker started", "worker_started"),
        ("[IdeaAnalyzer]", "idea_analyzing"),
        ("[Analyzer]", "task_analyzing"),
        ("[Organizer]", "planning"),
        ("[Outline]", "outline_building"),
        ("[Writer Agent]", "writing"),
        ("[Memory]", "memory_storing"),
        ("[TurningPoint]", "turning_point"),
        ("[Consistency]", "consistency_check"),
        ("[RealtimeEditor]", "editing"),
        ("[Reward]", "reward_calc"),
        ("[Executor] Reached the planned chapter count", "finishing"),
    ]
    for line in reversed(lines[-80:]):
        for marker, phase in checks:
            if marker in line:
                return phase
    return "running"


def _infer_action_from_worker_log(worker_log_text: str) -> str:
    if not worker_log_text:
        return ""
    lines = [ln.strip() for ln in worker_log_text.splitlines() if ln.strip()]
    if not lines:
        return ""

    noise_prefixes = (
        "Loading weights:",
        "Key",
        "|",
        "-",
        "[system]",
    )
    for line in reversed(lines[-120:]):
        if not line:
            continue
        if any(line.startswith(p) for p in noise_prefixes):
            continue
        if "BertModel LOAD REPORT" in line:
            continue
        if line.startswith("UNEXPECTED") or line.startswith("[3m"):
            continue
        return line[:180]
    return ""


def _phase_label(phase: str, language: str = "en") -> str:
    if phase.startswith("agent_") and phase.endswith("_done"):
        return "Agent step done"
    if phase.startswith("agent_"):
        return "Agent running"
    mapping = {
        "queued": "Queued",
        "running": "Running",
        "succeeded": "Succeeded",
        "failed": "Failed",
        "canceled": "Canceled",
        "waiting_worker": "Waiting worker start",
        "worker_started": "Worker started",
        "idea_analyzing": "Analyzing idea",
        "task_analyzing": "Analyzing task",
        "planning": "Planning",
        "outline_building": "Building outline",
        "writing": "Writing chapter",
        "memory_storing": "Storing memory",
        "turning_point": "Checking turning points",
        "consistency_check": "Consistency check",
        "editing": "Realtime editing",
        "reward_calc": "Calculating reward",
        "finishing": "Finalizing",
        "init": "Initializing",
        "idea_analyzed": "Idea analyzed",
        "planning_done": "Planning done",
        "outline_global_done": "Global outline done",
        "outline_chapters_done": "Chapter outlines done",
        "chapter_start": "Chapter started",
        "chapter_done": "Chapter done",
        "realtime_edit": "Realtime edit",
        "finalizing": "Final output",
        "error": "Error",
    }
    return mapping.get(phase, "Running")


def _build_progress_snapshot(job: GenerationJob, run_dir: Optional[Path], worker_log_text: str, progress_log_text: str) -> Dict:
    now = datetime.now(timezone.utc)
    language = _job_language(job)
    snapshot = _default_progress_snapshot(language)
    created_at = _to_utc(job.created_at)
    finished_at = _to_utc(job.finished_at)
    updated_at = _to_utc(job.updated_at)

    terminal = job.status in {"succeeded", "failed", "canceled"}
    end_ref = finished_at or (updated_at if terminal else now)

    elapsed_seconds = 0
    if created_at:
        try:
            elapsed_seconds = max(0, int((end_ref - created_at).total_seconds()))
        except Exception:
            elapsed_seconds = 0

    parsed = _parse_progress_log(progress_log_text, language=language)
    current_chapter = parsed["current_chapter"]
    planned_total = parsed["planned_total"]
    chapter_words = parsed["chapter_words"]
    total_words = parsed["total_words"]

    status_state = _read_status_file(run_dir)
    if status_state:
        current_chapter = int(status_state.get("chapter_no") or current_chapter or 0)
        planned_total = int(status_state.get("planned_total") or planned_total or 0)
        chapter_words = int(status_state.get("chapter_words") or chapter_words or 0)
        total_words = int(status_state.get("total_words") or total_words or 0)

    percent = 0
    if planned_total > 0 and current_chapter > 0:
        percent = int(min(100, max(0, (current_chapter / planned_total) * 100)))

    last_log_at = ""
    idle_seconds: Optional[int] = None
    if run_dir:
        worker_log = run_dir / "worker.log"
        if worker_log.exists():
            try:
                mtime = datetime.fromtimestamp(worker_log.stat().st_mtime, tz=timezone.utc)
                last_log_at = mtime.isoformat()
                anchor = end_ref if terminal else now
                idle_seconds = max(0, int((anchor - mtime).total_seconds()))
            except Exception:
                pass

    latest_event = str(parsed.get("latest_event") or "")
    event_phase_map = {
        "claw_native_loop": "planning",
        "global_outline": "planning",
        "chapter_outline_ready": "planning_done",
        "chapter_plan": "planning",
        "claw_loop_start": "planning",
        "claw_action_start": "running",
        "claw_action_result": "running",
        "claw_candidate_update": "realtime_edit",
        "claw_finalize_ready": "finalizing",
        "claw_loop_complete": "finalizing",
    }
    phase = job.status if terminal else str(event_phase_map.get(latest_event) or status_state.get("stage") or _infer_phase(worker_log_text))
    phase_label = _phase_label(phase, language=language)
    phase_note = str(
        parsed.get("phase_note")
        or status_state.get("message")
        or _infer_action_from_worker_log(worker_log_text)
        or ""
    )
    parsed_memory_counts = parsed.get("memory_counts") or {}
    status_memory_counts = status_state.get("memory_counts") or {}
    memory_counts: Dict[str, int] = {}
    for key in MEMORY_COUNT_KEYS:
        raw_val = status_memory_counts.get(key, parsed_memory_counts.get(key, 0))
        try:
            memory_counts[key] = int(raw_val or 0)
        except Exception:
            memory_counts[key] = 0

    stalled = False
    stall_reason = ""
    if job.status == "running":
        if not worker_log_text and elapsed_seconds > 20:
            stalled = True
            stall_reason = "No worker logs yet after 20s; the worker may not have started correctly."
        elif idle_seconds is not None and idle_seconds > 90:
            stalled = True
            stall_reason = f"No new logs for {idle_seconds}s; the job may be stalled."

        if status_state and status_state.get("updated_at"):
            try:
                ts = str(status_state.get("updated_at")).replace("Z", "+00:00")
                sdt = datetime.fromisoformat(ts)
                if sdt.tzinfo is None:
                    sdt = sdt.replace(tzinfo=timezone.utc)
                idle_status = max(0, int((now - sdt.astimezone(timezone.utc)).total_seconds()))
                if idle_status > 90:
                    stalled = True
                    stall_reason = f"Status heartbeat not updated for {idle_status}s; the job may be stalled."
            except Exception:
                pass

    snapshot.update(
        {
            "phase": phase,
            "phase_label": phase_label,
            "phase_note": phase_note,
            "elapsed_seconds": elapsed_seconds,
            "current_chapter": current_chapter,
            "planned_total": planned_total,
            "chapter_words": chapter_words,
            "total_words": total_words,
            "percent": percent,
            "last_log_at": last_log_at,
            "idle_seconds": idle_seconds if idle_seconds is not None else -1,
            "stalled": stalled,
            "stall_reason": stall_reason,
            "memory_counts": memory_counts if isinstance(memory_counts, dict) else {},
        }
    )
    return snapshot


def _load_chapter_outputs(run_dir: Path, include_content: bool = True) -> List[Dict]:
    return chapter_outputs(run_dir, include_content=include_content)


def _latest_chapter_file(run_dir: Path, chapter_no: int) -> Optional[Path]:
    return next((path for path in latest_chapter_files(run_dir) if int(path.name.split("_")[1]) == chapter_no), None)


def _latest_chapter_files(run_dir: Path) -> List[Path]:
    return latest_chapter_files(run_dir)


@app.get("/")
def index(request: Request, db: Session = Depends(get_db)):
    if _current_user(request, db):
        return _redirect("/dashboard")
    return _redirect("/select-mode")


@app.get("/select-mode")
@app.get("/mode-a")
@app.get("/mode-b")
def shared_portal_entry(request: Request):
    return RedirectResponse(_shared_portal_url(request, request.url.path), status_code=status.HTTP_303_SEE_OTHER)


@app.get("/register")
def register_page(request: Request, db: Session = Depends(get_db)):
    if _current_user(request, db):
        return _redirect("/dashboard")
    return _redirect("/select-mode")


@app.post("/register")
def register(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    db: Session = Depends(get_db),
):
    _ = (request, email, password, confirm_password, db)
    return _redirect("/select-mode")


@app.get("/login")
def login_page(request: Request, db: Session = Depends(get_db)):
    if _current_user(request, db):
        return _redirect("/dashboard")
    return _redirect("/select-mode")


@app.post("/login")
def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    _ = (request, email, password, db)
    return _redirect("/select-mode")


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return _redirect("/login")


@app.get("/dashboard")
def dashboard(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    return templates.TemplateResponse("dashboard.html", _console_context(request, db, user, active_nav="chat"))


@app.get("/console/chat")
def console_chat(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    return templates.TemplateResponse("dashboard.html", _console_context(request, db, user, active_nav="chat"))


@app.get("/console/sessions")
def console_sessions(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    return templates.TemplateResponse("console_sessions.html", _console_context(request, db, user, active_nav="sessions"))


@app.get("/console/tasks")
def console_tasks(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    return _redirect_with_notice(
        "/console/chat",
        message="Claw runs are now shown directly inside the chat workspace.",
    )

@app.get("/console/workspace")
def console_workspace(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    return _redirect(f"/console/manuscript{_session_suffix(request, db, user)}")


def _session_suffix(request: Request, db: Session, user: User) -> str:
    active_session_id = request.query_params.get("session_id", "").strip()
    if active_session_id:
        return f"?session_id={active_session_id}"
    try:
        context = _console_context(request, db, user, active_nav="chat")
        active_session = context.get("active_session")
        if active_session and getattr(active_session, "id", None):
            return f"?session_id={active_session.id}"
    except Exception:
        pass
    return ""


@app.get("/console/manuscript")
def console_manuscript(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    return _redirect(f"/console/manuscript/read{_session_suffix(request, db, user)}")


@app.get("/console/memory")
def console_memory(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    return _redirect(f"/console/memory/banks{_session_suffix(request, db, user)}")


@app.get("/console/manuscript/read")
def console_manuscript_read(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    book = current_book.get()
    manuscripts = db.scalars(select(GenerationJob).where(
        GenerationJob.book_id == book.id, GenerationJob.user_id == user.id,
        GenerationJob.run_id != "").order_by(GenerationJob.id.desc())).all() if book else []
    return templates.TemplateResponse(
        "console_manuscript_read.html",
        _console_page_context(request, db, user, active_nav="manuscript_read", workspace_section="read",
                              manuscripts=manuscripts),
    )


@app.get("/console/manuscript/outline")
def console_manuscript_outline(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    return templates.TemplateResponse(
        "console_manuscript_outline.html",
        _console_page_context(request, db, user, active_nav="manuscript_outline", workspace_section="outline"),
    )


@app.get("/console/manuscript/planning")
def console_manuscript_planning(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    return templates.TemplateResponse(
        "console_manuscript_planning.html",
        _console_page_context(request, db, user, active_nav="manuscript_planning", workspace_section="planning"),
    )


@app.get("/console/memory/banks")
def console_memory_banks(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    return _redirect(f"/console/memory/zone/premise{_session_suffix(request, db, user)}")


@app.get("/console/memory/entries")
def console_memory_entries(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    return _redirect(f"/console/memory/zone/premise{_session_suffix(request, db, user)}")


@app.get("/console/memory/revision")
def console_memory_revision_redirect(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    return _redirect(f"/console/memory/zone/revision_loop{_session_suffix(request, db, user)}")


_MEMORY_ZONE_META = {
    "premise": "Story Premise",
    "author_brief": "Author Brief",
    "chapter_planning": "Chapter Planning",
    "revision_loop": "Revision Loop",
    "canon_state": "Canon State",
    "runtime": "Runtime",
}


@app.get("/console/memory/zone/{slug}")
def console_memory_zone(slug: str, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    if slug not in _MEMORY_ZONE_META:
        slug = "premise"
    name = _MEMORY_ZONE_META[slug]
    return templates.TemplateResponse(
        "console_memory_zone.html",
        _console_page_context(
            request, db, user,
            active_nav=f"memory_zone_{slug}",
            workspace_section="entries",
            memory_zone_slug=slug,
            memory_zone_name_en=name,
        ),
    )
def _console_memory_revision_legacy(request: Request, db: Session = Depends(get_db)):
    pass  # replaced by zone routes above


@app.get("/console/storyboard")
def console_storyboard(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    return templates.TemplateResponse("console_storyboard.html", _console_context(request, db, user, active_nav="storyboard"))


@app.get("/console/characters")
def console_characters(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    context = _console_context(request, db, user, active_nav="characters")
    book = current_book.get()
    context["intimate_notes"] = prose_style.load(_prose_folder()).character_notes if book else {}
    return templates.TemplateResponse("console_characters.html", context)


@app.post("/console/characters")
def add_character(request: Request, name: str = Form(..., max_length=160), profile: str = Form("", max_length=12000), db: Session = Depends(get_db)):
    _require_user(request, db)
    book = current_book.get()
    if not book or not name.strip():
        raise HTTPException(422, "Choose a book and enter a character name")
    _portal_memory_system(book.memory_key).store_character(name.strip(), profile.strip(), book.title)
    return _redirect("/console/characters")


@app.get("/console/world")
def console_world(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    return templates.TemplateResponse("console_world.html", _console_context(request, db, user, active_nav="world"))


def _prose_folder():
    book = current_book.get()
    if not book:
        raise HTTPException(404, "Choose a Book first.")
    return book_memory_dir(Config(require_api_key=False), book.memory_key)


@app.post("/console/characters/intimate-note")
def save_intimate_note(request: Request, name: str = Form(..., min_length=1, max_length=160),
                       note: str = Form("", max_length=3000), previous: str = Form(""), db: Session = Depends(get_db)):
    _require_user(request, db)
    prose_style.update(_prose_folder(), previous, character=name.strip().casefold(), note=note)
    return _redirect_with_notice("/console/characters", message="Intimate character note saved.")


@app.post("/console/style")
def save_prose_style(request: Request, level: prose_style.Level = Form(...), note: str = Form("", max_length=3000),
                     previous_level: prose_style.Level = Form(...), previous_note: str = Form(""), db: Session = Depends(get_db)):
    _require_user(request, db)
    prose_style.update(_prose_folder(), [previous_level, previous_note], level=level, note=note)
    return _redirect_with_notice("/console/style", message="Book prose settings saved.")


@app.get("/console/style")
def console_style(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    context = _console_context(request, db, user, active_nav="style")
    if current_book.get():
        context.update(prose_settings=prose_style.load(_prose_folder()), prose_levels=prose_style.LEVELS)
    return templates.TemplateResponse("console_style.html", context)


@app.get("/console/skills")
def console_skills(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    return templates.TemplateResponse("console_skills.html", _console_context(request, db, user, active_nav="skills"))


@app.post("/capabilities/{slug}/toggle")
def toggle_capability(slug: str, request: Request, enabled: str = Form("0"), db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")

    normalized_slug = str(slug or "").strip().lower()
    spec = capability_map().get(normalized_slug)
    if not spec:
        return _redirect_with_notice("/console/skills", error="Unknown capability")
    if spec.always_enabled:
        return _redirect_with_notice("/console/skills", error="This core capability cannot be disabled")

    row = db.execute(
        select(CapabilityPreference).where(
            CapabilityPreference.user_id == user.id,
            CapabilityPreference.slug == normalized_slug,
        )
    ).scalar_one_or_none()
    enabled_value = str(enabled or "0").strip().lower() in {"1", "true", "yes", "on"}

    if row:
        row.enabled = enabled_value
    else:
        row = CapabilityPreference(user_id=user.id, slug=normalized_slug, enabled=enabled_value)
        db.add(row)

    db.commit()
    state_text = "enabled" if enabled_value else "disabled"
    name_text = spec.name_en
    return _redirect_with_notice("/console/skills", message=f"{name_text} {state_text}")


@app.get("/console/mcp")
def console_mcp(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    return templates.TemplateResponse("console_mcp.html", _console_context(request, db, user, active_nav="mcp"))


@app.get("/console/agents")
def console_agents(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    return templates.TemplateResponse("console_agents.html", _console_context(request, db, user, active_nav="agents"))


@app.get("/console/models")
def console_models(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    context = _console_context(request, db, user, active_nav="models")
    book = current_book.get()
    return templates.TemplateResponse("console_models.html", {**context, "usage":model_accounting.book_usage(db, book.id if book else None)})


@app.get("/console/env")
def console_env(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    ctx = _console_context(request, db, user, active_nav="env")
    ctx["env_form_values"] = {
        "web_execution_mode": os.getenv("WEB_EXECUTION_MODE", settings.execution_mode),
        "web_claw_max_steps": os.getenv("WEB_CLAW_MAX_STEPS", str(settings.claw_max_steps)),
        "web_max_iterations": os.getenv("WEB_MAX_ITERATIONS", str(settings.max_iterations)),
        "web_max_total_iterations": os.getenv("WEB_MAX_TOTAL_ITERATIONS", str(settings.max_total_iterations)),
        "min_chapter_chars": os.getenv("MIN_CHAPTER_CHARS", "3000"),
        "max_chapter_chars": os.getenv("MAX_CHAPTER_CHARS", "0"),
        "web_max_chapter_subrounds": os.getenv("WEB_MAX_CHAPTER_SUBROUNDS", str(settings.max_chapter_subrounds)),
        "web_chapters_per_iter": os.getenv("WEB_CHAPTERS_PER_ITER", str(settings.chapters_per_iter)),
        "web_fast_mode": os.getenv("WEB_FAST_MODE", "1" if settings.fast_mode else "0"),
        "web_full_cycle_interval": os.getenv("WEB_FULL_CYCLE_INTERVAL", str(settings.full_cycle_interval)),
        "web_eval_interval": os.getenv("WEB_EVAL_INTERVAL", str(settings.eval_interval)),
        "temperature": os.getenv("TEMPERATURE", "0.7"),
        "max_tokens": os.getenv("MAX_TOKENS", "8000"),
        "llm_timeout_seconds": os.getenv("LLM_TIMEOUT_SECONDS", "0"),
        "llm_max_retries": os.getenv("LLM_MAX_RETRIES", "1"),
        "web_memory_only_mode": os.getenv("WEB_MEMORY_ONLY_MODE", "1"),
        "context_max_chars": os.getenv("CONTEXT_MAX_CHARS", "12000"),
        "recent_context_items": os.getenv("RECENT_CONTEXT_ITEMS", "4"),
        "web_enable_evaluator": os.getenv("WEB_ENABLE_EVALUATOR", "0"),
        "web_enable_rag": os.getenv("WEB_ENABLE_RAG", "0"),
        "web_enable_static_kb": os.getenv("WEB_ENABLE_STATIC_KB", "0"),
        "turning_point_enabled": os.getenv("TURNING_POINT_ENABLED", "1"),
    }
    return templates.TemplateResponse("console_env.html", ctx)


@app.post("/console/env/save")
async def save_env_settings(
    request: Request,
    db: Session = Depends(get_db),
    # Claw engine
    web_execution_mode: str = Form("claw"),
    web_claw_max_steps: str = Form("6"),
    web_max_iterations: str = Form("8"),
    web_max_total_iterations: str = Form("16"),
    # Chapter writing
    min_chapter_chars: str = Form("3000"),
    max_chapter_chars: str = Form("0"),
    web_max_chapter_subrounds: str = Form("2"),
    web_chapters_per_iter: str = Form("1"),
    # Performance
    web_fast_mode: str = Form("1"),
    web_full_cycle_interval: str = Form("3"),
    web_eval_interval: str = Form("8"),
    # LLM params
    temperature: str = Form("0.7"),
    max_tokens: str = Form("8000"),
    llm_timeout_seconds: str = Form("0"),
    llm_max_retries: str = Form("1"),
    # Memory & context
    web_memory_only_mode: str = Form("1"),
    context_max_chars: str = Form("12000"),
    recent_context_items: str = Form("4"),
    # Advanced
    web_enable_evaluator: str = Form("0"),
    web_enable_rag: str = Form("0"),
    web_enable_static_kb: str = Form("0"),
    turning_point_enabled: str = Form("1"),
):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")


    def _flag(v: str) -> str:
        return "1" if str(v or "0").strip() in {"1", "true", "on", "yes"} else "0"

    def _int_clamp(v: str, lo: int, hi: int, default: int) -> str:
        try:
            return str(max(lo, min(hi, int(v or default))))
        except (ValueError, TypeError):
            return str(default)

    def _float_clamp(v: str, lo: float, hi: float, default: float) -> str:
        try:
            return f"{max(lo, min(hi, float(v or default))):.2f}"
        except (ValueError, TypeError):
            return str(default)

    new_env: Dict[str, str] = {
        "WEB_EXECUTION_MODE": (web_execution_mode or "claw").strip().lower(),
        "WEB_CLAW_MAX_STEPS": _int_clamp(web_claw_max_steps, 4, 200, 6),
        "WEB_MAX_ITERATIONS": _int_clamp(web_max_iterations, 1, 500, 8),
        "WEB_MAX_TOTAL_ITERATIONS": _int_clamp(web_max_total_iterations, 1, 1000, 16),
        "MIN_CHAPTER_CHARS": _int_clamp(min_chapter_chars, 300, 50000, 3000),
        "MAX_CHAPTER_CHARS": _int_clamp(max_chapter_chars, 0, 100000, 0),
        "WEB_MAX_CHAPTER_SUBROUNDS": _int_clamp(web_max_chapter_subrounds, 1, 10, 2),
        "WEB_CHAPTERS_PER_ITER": _int_clamp(web_chapters_per_iter, 1, 10, 1),
        "WEB_FAST_MODE": _flag(web_fast_mode),
        "WEB_FULL_CYCLE_INTERVAL": _int_clamp(web_full_cycle_interval, 0, 20, 3),
        "WEB_EVAL_INTERVAL": _int_clamp(web_eval_interval, 1, 50, 8),
        "TEMPERATURE": _float_clamp(temperature, 0.0, 2.0, 0.7),
        "MAX_TOKENS": _int_clamp(max_tokens, 1000, 32000, 8000),
        "LLM_TIMEOUT_SECONDS": _int_clamp(llm_timeout_seconds, 0, 3600, 0),
        "LLM_MAX_RETRIES": _int_clamp(llm_max_retries, 0, 10, 1),
        "WEB_MEMORY_ONLY_MODE": _flag(web_memory_only_mode),
        "CONTEXT_MAX_CHARS": _int_clamp(context_max_chars, 2000, 100000, 12000),
        "RECENT_CONTEXT_ITEMS": _int_clamp(recent_context_items, 1, 20, 4),
        "WEB_ENABLE_EVALUATOR": _flag(web_enable_evaluator),
        "WEB_ENABLE_RAG": _flag(web_enable_rag),
        "WEB_ENABLE_STATIC_KB": _flag(web_enable_static_kb),
        "TURNING_POINT_ENABLED": _flag(turning_point_enabled),
    }

    # Persist to .env file (creates or updates)
    env_path = BASE_DIR / ".env"
    existing_lines: List[str] = []
    written_keys: set = set()
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                existing_lines.append(line)
                continue
            key = stripped.split("=", 1)[0].strip()
            if key in new_env:
                existing_lines.append(f"{key}={new_env[key]}")
                written_keys.add(key)
            else:
                existing_lines.append(line)
    for key, value in new_env.items():
        if key not in written_keys:
            existing_lines.append(f"{key}={value}")
    try:
        atomic_write_text(env_path, "\n".join(existing_lines) + "\n", encoding="utf-8")
    except OSError:
        return _redirect_with_notice("/console/env", error="Settings were not saved. Check directory permissions or container mounts.")

    # Update os.environ so the current process picks up new values immediately
    for key, value in new_env.items():
        os.environ[key] = value

    # Update frozen settings object live (bypass frozen constraint via object.__setattr__)
    _settings_map = {
        "WEB_EXECUTION_MODE": ("execution_mode", str),
        "WEB_CLAW_MAX_STEPS": ("claw_max_steps", int),
        "WEB_MAX_ITERATIONS": ("max_iterations", int),
        "WEB_MAX_TOTAL_ITERATIONS": ("max_total_iterations", int),
        "WEB_FAST_MODE": ("fast_mode", lambda v: v == "1"),
        "WEB_FULL_CYCLE_INTERVAL": ("full_cycle_interval", int),
        "WEB_CHAPTERS_PER_ITER": ("chapters_per_iter", int),
        "WEB_MAX_CHAPTER_SUBROUNDS": ("max_chapter_subrounds", int),
        "WEB_EVAL_INTERVAL": ("eval_interval", int),
    }
    for env_key, (attr, converter) in _settings_map.items():
        if env_key in new_env:
            try:
                object.__setattr__(settings, attr, converter(new_env[env_key]))
            except Exception:
                pass

    msg = "✓ Settings saved. Takes effect on the next generation run."
    return _redirect_with_notice("/console/env", message=msg)

@app.get("/console/status")
def console_status(request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    return templates.TemplateResponse("console_status.html", _console_context(request, db, user, active_nav="status"))


@app.get("/providers/{provider}/models")
def available_models(provider: str, request: Request, refresh: bool = False, db: Session = Depends(get_db)):
    user = _require_user(request, db)
    return _model_catalog(db, user.id, provider, refresh)


def _model_catalog(db, user_id, provider, refresh):
    spec = _provider_specs_for_user(db, user_id).get(provider)
    if not spec:
        raise HTTPException(404, "Provider not available")
    try:
        key = model_connections.credential(db, user_id, spec, optional=True)
        return {"models": model_connections.catalog(spec, key, user_id, refresh)}
    except ValueError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/api-keys")
def save_api_key(
    request: Request,
    provider: str = Form(...),
    api_key: str = Form(...),
    db: Session = Depends(get_db),
):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    if _modelless_mode_enabled():
        return _reject_when_modelless(request, path="/console/models")

    provider = provider.strip().lower()
    api_key = api_key.strip()
    provider_specs = _provider_specs_for_user(db, user.id)

    if provider not in provider_specs:
        return _redirect_with_notice("/console/models", error="Unsupported provider")
    if len(api_key) < 12:
        return _redirect_with_notice("/console/models", error="API key looks too short")

    cred = db.execute(
        select(ApiCredential).where(
            ApiCredential.user_id == user.id,
            ApiCredential.provider == provider,
        )
    ).scalar_one_or_none()

    encrypted = encrypt_api_key(api_key)
    if cred:
        cred.encrypted_key = encrypted
        cred.key_hint = _mask_hint(api_key)
    else:
        cred = ApiCredential(
            user_id=user.id,
            provider=provider,
            encrypted_key=encrypted,
            key_hint=_mask_hint(api_key),
        )
        db.add(cred)

    db.commit()
    return _redirect_with_notice("/console/models", message="API key saved")



@app.post("/providers")
def save_provider_config(
    request: Request,
    slug: str = Form(...),
    label: str = Form(""),
    kind: str = Form("custom"),
    base_url: str = Form(...),
    model: str = Form(...),
    wire_api: str = Form("chat"),
    db: Session = Depends(get_db),
):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    if _modelless_mode_enabled():
        return _reject_when_modelless(request, path="/console/models")

    norm_slug = normalize_slug(slug)
    norm_label = (label or "").strip()
    norm_base_url = (base_url or "").strip()
    norm_model = (model or "").strip()
    norm_wire_api = normalize_wire_api(wire_api)

    if kind not in {"local","custom"}:
        raise HTTPException(422,"Invalid connection type")
    if not is_valid_slug(norm_slug):
        return _redirect_with_notice(
            "/console/models",
            error="Invalid slug. Use 2-32 chars (a-z, 0-9, _, -)",
        )
    if not norm_base_url:
        return _redirect_with_notice("/console/models", error="Base URL cannot be empty")
    if not norm_model:
        return _redirect_with_notice("/console/models", error="Model cannot be empty")
    if not (norm_base_url.startswith("http://") or norm_base_url.startswith("https://")):
        return _redirect_with_notice(
            "/console/models",
            error="Base URL must start with http:// or https://",
        )

    base_specs = get_provider_specs(settings)
    if norm_slug in base_specs and not db.scalar(select(ProviderConfig).where(ProviderConfig.slug == norm_slug, ProviderConfig.user_id == user.id)):
        return _redirect_with_notice(
            "/console/models",
            error="This provider slug is reserved. Use another slug",
        )

    row = db.execute(
        select(ProviderConfig).where(
            ProviderConfig.user_id == user.id,
            ProviderConfig.slug == norm_slug,
        )
    ).scalar_one_or_none()

    if row:
        _clear_moved_key(db, row, norm_base_url)
        row.kind = kind
        row.label = norm_label
        row.base_url = norm_base_url
        row.model = norm_model
        row.wire_api = norm_wire_api
    else:
        row = ProviderConfig(
            user_id=user.id,
            slug=norm_slug,
            kind=kind,
            label=norm_label,
            base_url=norm_base_url,
            model=norm_model,
            wire_api=norm_wire_api,
        )
        db.add(row)

    db.commit()
    return _redirect_with_notice("/console/models", message="Custom provider saved")



@app.post("/providers/{slug}/delete")
def delete_provider_config(slug: str, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    if _modelless_mode_enabled():
        return _reject_when_modelless(request, path="/console/models")

    norm_slug = normalize_slug(slug)
    row = db.execute(
        select(ProviderConfig).where(
            ProviderConfig.user_id == user.id,
            ProviderConfig.slug == norm_slug,
        )
    ).scalar_one_or_none()
    if not row:
        return _redirect_with_notice("/console/models", error="Custom provider not found")

    cred = db.execute(
        select(ApiCredential).where(
            ApiCredential.user_id == user.id,
            ApiCredential.provider == norm_slug,
        )
    ).scalar_one_or_none()
    if cred:
        db.delete(cred)

    db.delete(row)
    db.commit()
    return _redirect_with_notice("/console/models", message="Custom provider deleted")



@app.post("/idea-copilot/start")
def start_idea_copilot(
    request: Request,
    idea: str = Form(...),
    provider: str = Form(...),
    model: str = Form(""),
    db: Session = Depends(get_db),
):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    if _modelless_mode_enabled():
        return _reject_when_modelless(request, path="/console/chat")

    idea = idea.strip()
    provider = provider.strip().lower()
    if not idea:
        return _redirect_with_notice("/console/chat", error="Idea cannot be empty")

    provider_specs = _provider_specs_for_user(db, user.id)
    spec = provider_specs.get(provider)
    if not spec:
        return _redirect_with_notice("/console/chat", error="Unsupported provider")

    try:
        spec = model_connections.select_model(spec, model)
        api_key = _provider_api_key(db, user.id, provider)
    except ValueError as exc:
        return _redirect_with_notice("/console/chat", error=str(exc))

    source_language = "en"
    state_payload = {
        "version": 2,
        "messages": [],
        "refined_idea": idea,
        "round": 0,
        "preferred_language": source_language,
        "source_language": source_language,
        "ui_language": source_language,
        "translation_mode": "english",
    }
    book = current_book.get() or new_book(db, user.id, idea)
    session = IdeaCopilotSession(
        user_id=user.id,
        book_id=book.id,
        provider=provider,
        model_selection=model_connections.snapshot(spec),
        status="active",
        original_idea=idea,
        refined_idea=idea,
        conversation_json=dump_state(state_payload),
        round_count=0,
        readiness_score=0,
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    state = load_state(session.conversation_json)
    agent = IdeaCopilotAgent(spec, api_key)
    try:
        with model_accounting.recording(session.book_id, f"chat {session.id}"):
            turn = agent.generate_turn(
                original_idea=idea,
                state={**state, "book_notes": _book_notes(session.book_id)},
                latest_user_reply="",
            )
    except Exception as exc:
        turn = {
            "role": "assistant",
            "analysis": (
                f"The first ideation turn failed: {exc}"
            ),
            "refined_idea": idea,
            "questions": [],
            "readiness": 10,
            "ready_hint": (
                "Add more detail about premise, voice, and hard constraints."
            ),
            "language": "en",
            "style_targets": [],
            "memory_targets": [],
        }

    state = append_assistant_turn(state, turn)
    session.conversation_json = dump_state(state)
    session.refined_idea = str(state.get("refined_idea") or idea)
    session.round_count = int(state.get("round", 0) or 0)
    session.readiness_score = int(turn.get("readiness", 0) or 0)
    db.commit()
    _remember_session_turn(session, state, turn=turn)
    return _redirect(f"/console/chat?session_id={session.id}")


@app.get("/idea-copilot/{session_id}")
def idea_copilot_detail(session_id: int, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")

    session = db.get(IdeaCopilotSession, session_id)
    if not session or session.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Idea session not found")
    return _redirect(f"/console/chat?session_id={session.id}")


@app.post("/idea-copilot/{session_id}/reply")
def idea_copilot_reply(
    session_id: int,
    request: Request,
    reply: str = Form(...),
    provider: str = Form(""),
    model: str = Form(""),
    db: Session = Depends(get_db),
):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    if _modelless_mode_enabled():
        return _reject_when_modelless(request, path=f"/console/chat?session_id={session_id}")

    session = db.get(IdeaCopilotSession, session_id)
    if not session or session.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Idea session not found")
    if session.status != "active":
        return _redirect_with_notice(f"/console/chat?session_id={session_id}", error="Session is not active")

    answer = reply.strip()
    if not answer:
        if request.headers.get("x-requested-with") == "fetch":
            return JSONResponse({"ok": False, "error": "Reply cannot be empty"}, status_code=400)
        return _redirect_with_notice(f"/console/chat?session_id={session_id}", error="Reply cannot be empty")

    state = load_state(session.conversation_json)
    if state.get("reply_pending"):
        if request.headers.get("x-requested-with") == "fetch":
            return JSONResponse({"ok": False, "error": "Previous reply is still processing"}, status_code=409)
        return _redirect_with_notice(f"/console/chat?session_id={session_id}", error="Previous reply is still processing")

    provider_specs = _provider_specs_for_user(db, user.id)
    target = provider.strip() or session.provider
    spec = provider_specs.get(target)
    if not spec:
        if request.headers.get("x-requested-with") == "fetch":
            return JSONResponse({"ok": False, "error": "Provider is no longer available"}, status_code=400)
        return _redirect_with_notice(f"/console/chat?session_id={session_id}", error="Provider is no longer available")
    try:
        spec = model_connections.select_model(spec, model, session.model_selection if target == session.provider and (not model or model == session.selected_model) else None)
        _provider_api_key(db, user.id, target)
    except ValueError as exc:
        if request.headers.get("x-requested-with") == "fetch":
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return _redirect_with_notice(f"/console/chat?session_id={session_id}", error=str(exc))

    state = append_user_reply(state, answer)
    state["reply_pending"] = True
    state["reply_error"] = ""
    state["reply_started_at"] = datetime.now(timezone.utc).isoformat()
    accepted = db.execute(update(IdeaCopilotSession).where(
        IdeaCopilotSession.id == session.id,
        IdeaCopilotSession.status == "active",
        IdeaCopilotSession.conversation_json == session.conversation_json,
    ).values(provider=target, model_selection=model_connections.snapshot(spec), conversation_json=dump_state(state), refined_idea=str(state.get("refined_idea") or session.refined_idea or session.original_idea))).rowcount
    db.commit()
    if not accepted:
        if request.headers.get("x-requested-with") == "fetch":
            return JSONResponse({"ok": False, "error": "The session changed; refresh before sending another reply"}, status_code=409)
        return _redirect_with_notice(f"/console/chat?session_id={session_id}", error="The session changed; refresh before sending another reply")
    db.refresh(session)
    _remember_session_turn(session, state, user_reply=answer)
    _start_idea_copilot_reply_worker(session_id)

    provider_label = spec.label if spec else session.provider.upper()
    payload = _idea_session_payload(session, state, provider_label)
    if request.headers.get("x-requested-with") == "fetch":
        return JSONResponse({"ok": True, **payload}, status_code=202)
    return _redirect(f"/console/chat?session_id={session_id}")


@app.get("/api/idea-copilot/{session_id}/state")
def idea_copilot_state(session_id: int, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    session = db.get(IdeaCopilotSession, session_id)
    if not session or session.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Idea session not found")

    state = load_state(session.conversation_json)
    provider_specs = _provider_specs_for_user(db, user.id)
    provider_label = provider_specs.get(session.provider).label if provider_specs.get(session.provider) else session.provider.upper()
    return JSONResponse({"ok": True, **_idea_session_payload(session, state, provider_label)})


@app.post("/api/idea-copilot/{session_id}/brief")
async def idea_copilot_brief_edit(
    session_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    """Allow users to directly edit the refined_idea brief for a session."""
    user = _current_user(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    session = db.get(IdeaCopilotSession, session_id)
    if not session or session.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    try:
        body = await _json_payload(request)
        new_brief = str(body.get("brief", "")).strip()
    except Exception:
        form = await request.form()
        new_brief = str(form.get("brief", "")).strip()

    if not new_brief:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="brief cannot be empty")

    session.refined_idea = new_brief
    # Also patch the conversation state so polls return updated brief
    state = load_state(session.conversation_json)
    state["refined_idea"] = new_brief
    if state.get("reply_pending"):
        state["reply_pending"] = False
        state["reply_error"] = "The brief was updated. Send a new message to continue."
    session.conversation_json = dump_state(state)
    db.commit()
    return JSONResponse({"ok": True, "refined_idea": new_brief})


@app.post("/idea-copilot/{session_id}/confirm")
def idea_copilot_confirm(session_id: int, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    if _modelless_mode_enabled():
        return _reject_when_modelless(request, path=f"/console/chat?session_id={session_id}")

    session = db.get(IdeaCopilotSession, session_id)
    if not session or session.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Idea session not found")
    if session.status != "active":
        if session.final_job_id:
            return _redirect(f"/console/chat?session_id={session_id}")
        return _redirect_with_notice(f"/console/chat?session_id={session_id}", error="Session is not active")

    try:
        _start_generation_from_idea_session(db, session, user.id)
    except ValueError as exc:
        return _redirect_with_notice(f"/console/chat?session_id={session_id}", error=str(exc))
    return _redirect_with_notice(
        f"/console/chat?session_id={session_id}",
        message="NovelClaw has entered creation mode.",
    )


@app.post("/idea-copilot/{session_id}/cancel")
def idea_copilot_cancel(session_id: int, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")

    session = db.get(IdeaCopilotSession, session_id)
    if not session or session.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Idea session not found")

    canceled_run = False
    if session.final_job_id:
        job = _session_job(db, session)
        canceled_run = _cancel_generation_job(job, reason="[user] canceled via session close.")

    if session.status in {"active", "confirmed"}:
        session.status = "canceled"
    state = load_state(session.conversation_json)
    state["reply_pending"] = False
    state["reply_error"] = ""
    session.conversation_json = dump_state(state)
    session.finished_at = datetime.now(timezone.utc)
    db.commit()
    return _redirect_with_notice(
        "/console/chat",
        message="Writing session closed. The linked run was stopped." if canceled_run else "Writing session closed.",
    )


@app.post("/idea-copilot/{session_id}/delete")
def idea_copilot_delete(
    session_id: int,
    request: Request,
    next_path: str = Form("/console/sessions", alias="next"),
    db: Session = Depends(get_db),
):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")

    session = db.get(IdeaCopilotSession, session_id)
    if not session or session.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Idea session not found")

    target_path = _safe_next_path(next_path, "/console/sessions")
    if session.final_job_id:
        job = _session_job(db, session)
        _cancel_generation_job(job, reason="[user] canceled via session delete.")
        if job:
            _detach_job_from_sessions(db, job)
    db.delete(session)
    db.commit()
    return _redirect_with_notice(
        target_path,
        message="Writing session deleted. The linked run was canceled and detached.",
    )


@app.post("/jobs")
def create_job(
    request: Request,
    idea: str = Form(...),
    provider: str = Form(...),
    model: str = Form(""),
    db: Session = Depends(get_db),
):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    if _modelless_mode_enabled():
        return _reject_when_modelless(request, path="/console/chat")

    idea = idea.strip()
    provider = provider.strip().lower()
    provider_specs = _provider_specs_for_user(db, user.id)

    if not idea:
        return _redirect_with_notice("/console/chat", error="Idea cannot be empty")
    if provider not in provider_specs:
        return _redirect_with_notice("/console/chat", error="Unsupported provider")

    try:
        _provider_api_key(db, user.id, provider)
    except ValueError:
        return _redirect("/dashboard?error=Save+your+API+key+for+that+provider+first")

    book = current_book.get() or new_book(db, user.id, idea)
    job = _create_generation_job(db, user.id, provider, idea, book_id=book.id, model=model)
    _start_generation_worker(job.id)
    return _redirect_with_notice(
        "/console/chat",
        message="Claw has started in the current chat workspace.",
    )


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: int, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")

    job = db.get(GenerationJob, job_id)
    if not job or job.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if _cancel_generation_job(job):
        db.commit()
        return _redirect_with_notice(
            "/console/chat",
            message="The Claw run was canceled. You can continue giving instructions in the current chat.",
        )

    return _redirect_with_notice(
        "/console/chat",
        message="There is no cancelable Claw run right now.",
    )


@app.post("/jobs/{job_id}/delete")
def delete_job(job_id: int, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")

    job = db.get(GenerationJob, job_id)
    if not job or job.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job.status in {"queued", "running"}:
        return _redirect_with_notice(
            "/console/chat",
            error="A running Claw task cannot be deleted directly. Cancel it first.",
        )

    run_id = (job.run_id or "").strip()
    run_dir = _resolve_run_dir(run_id) if run_id else None
    _detach_job_from_sessions(db, job)

    db.delete(job)
    db.commit()

    if run_dir and run_dir.exists():
        try:
            shutil.rmtree(run_dir, ignore_errors=True)
        except Exception:
            pass

    return _redirect_with_notice(
        "/console/chat",
        message="Claw run record deleted.",
    )


@app.get("/jobs/{job_id}")
def job_detail(job_id: int, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")

    job = db.get(GenerationJob, job_id)
    if not job or job.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    related_session = db.execute(
        select(IdeaCopilotSession)
        .where(IdeaCopilotSession.final_job_id == job.id)
        .order_by(IdeaCopilotSession.updated_at.desc())
    ).scalar_one_or_none()

    if related_session:
        return _redirect_with_notice(
            f"/console/chat?session_id={related_session.id}",
            message="This Claw run is now embedded in the chat workspace.",
        )
    return _redirect_with_notice(
        "/console/chat",
        message="This Claw run is now embedded in the chat workspace.",
    )


@app.get("/api/idea-copilot/{session_id}/live")
def idea_copilot_live(session_id: int, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    session = db.get(IdeaCopilotSession, session_id)
    if not session or session.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Idea session not found")

    if not session.final_job_id:
        return JSONResponse(
            {
                "session_id": session.id,
                "status": "idle",
                "run_id": "",
                "progress_log": "",
                "progress_log_raw": "",
                "updated_at": session.updated_at.isoformat() if session.updated_at else "",
                "error_message": "",
                "progress_snapshot": _default_progress_snapshot(_ui_language(request)),
            }
        )

    job = _session_job(db, session)
    if not job or job.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Claw run not found")

    worker_log_text = ""
    progress_log_text = ""
    run_dir: Optional[Path] = None
    if job.run_id:
        run_dir = _resolve_run_dir(job.run_id)
        worker_log_text = _tail_text(run_dir / "worker.log")
        progress_log_text = _tail_text(run_dir / "progress.log")
    try:
        progress_snapshot = _build_progress_snapshot(job, run_dir, worker_log_text, progress_log_text)
    except Exception:
        progress_snapshot = _default_progress_snapshot(_job_language(job))

    job_language = _job_language(job)
    progress_log_raw = progress_log_text
    progress_log_text = _render_progress_log(progress_log_text, language=job_language)

    return JSONResponse(
        {
            "session_id": session.id,
            "status": job.status,
            "run_id": job.run_id,
            "progress_log": progress_log_text,
            "progress_log_raw": progress_log_raw,
            "updated_at": job.updated_at.isoformat() if job.updated_at else "",
            "error_message": job.error_message or "",
            "progress_snapshot": progress_snapshot,
        }
    )

@app.get("/jobs/{job_id}/logs")
def job_logs(job_id: int, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    job = db.get(GenerationJob, job_id)
    if not job or job.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    worker_log_text = ""
    progress_log_text = ""
    run_dir: Optional[Path] = None
    if job.run_id:
        run_dir = _resolve_run_dir(job.run_id)
        worker_log_text = _tail_text(run_dir / "worker.log")
        progress_log_text = _tail_text(run_dir / "progress.log")
    try:
        progress_snapshot = _build_progress_snapshot(job, run_dir, worker_log_text, progress_log_text)
    except Exception:
        progress_snapshot = _default_progress_snapshot(_job_language(job))

    job_language = _job_language(job)
    progress_log_raw = progress_log_text
    progress_log_text = _render_progress_log(progress_log_text, language=job_language)

    return JSONResponse(
        {
            "job_id": job.id,
            "status": job.status,
            "run_id": job.run_id,
            "worker_log": worker_log_text,
            "progress_log": progress_log_text,
            "progress_log_raw": progress_log_raw,
            "updated_at": job.updated_at.isoformat() if job.updated_at else "",
            "error_message": job.error_message or "",
            "progress_snapshot": progress_snapshot,
        }
    )


@app.get("/jobs/{job_id}/chapters")
def job_chapters(job_id: int, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    job = db.get(GenerationJob, job_id)
    if not job or job.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    chapters: List[Dict] = []
    if job.run_id:
        run_dir = _resolve_run_dir(job.run_id)
        chapters = _load_chapter_outputs(run_dir)

    return JSONResponse(
        {
            "job_id": job.id,
            "status": job.status,
            "run_id": job.run_id,
            "chapter_count": len(chapters),
            "chapters": chapters,
        }
    )


@app.get("/jobs/{job_id}/download/output")
def download_job_output(job_id: int, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    job = db.get(GenerationJob, job_id)
    if not job or job.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    run_id = str(job.run_id or "").strip()
    filename = f"job_{job.id}_{run_id or 'no_run'}_output.txt"
    return output_download_response(_resolve_run_dir(run_id) if run_id else None, job.output_path, filename)


@app.get("/jobs/{job_id}/download/chapter/{chapter_no}")
def download_job_chapter(job_id: int, chapter_no: int, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    job = db.get(GenerationJob, job_id)
    if not job or job.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    if chapter_no <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid chapter number")
    if not job.run_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No chapter output available yet")

    run_dir = _resolve_run_dir(job.run_id)
    chapter_path = _latest_chapter_file(run_dir, chapter_no)
    if not chapter_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chapter output not found")

    return FileResponse(
        str(chapter_path),
        media_type="text/plain; charset=utf-8",
        filename=chapter_path.name,
    )


@app.get("/jobs/{job_id}/download/chapters")
def download_job_chapters(job_id: int, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    job = db.get(GenerationJob, job_id)
    if not job or job.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    if not job.run_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No chapter output available yet")

    run_dir = _resolve_run_dir(job.run_id)
    return chapters_archive_response(run_dir, job.id)


@app.get("/api/runs/{run_id}/memory-banks")
def api_run_memory_banks(run_id: str, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    book = current_book.get()
    if book:
        run_id = book.memory_key
    elif not _job_for_run_id(db, user.id, run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    index = _load_editable_memory_index_for_run(run_id)
    language = _ui_language(request)
    return JSONResponse(
        {
            "run_id": run_id,
            "topic_hint": _memory_topic_hint(index),
            "groups": _build_memory_bank_groups_from_index(index, language),
        }
    )


@app.post("/api/runs/{run_id}/memory-banks/{bank}/entries")
async def api_add_memory_bank_entry(run_id: str, bank: str, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    if bank not in MemorySystem.CLAW_BANKS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory bank not found")
    book = current_book.get()
    if book:
        run_id = book.memory_key
    elif not _job_for_run_id(db, user.id, run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    body = await _json_payload(request)
    content = str(body.get("content") or "").strip()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Content cannot be empty")

    index = _load_editable_memory_index_for_run(run_id)
    topic = str(body.get("topic") or "").strip() or _memory_topic_hint(index)
    chapter_value = body.get("chapter")
    metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    kind = str(body.get("kind") or metadata.get("kind") or "").strip()
    source = str(body.get("source") or metadata.get("source") or "manual_workspace").strip() or "manual_workspace"
    timestamp = datetime.now().isoformat()
    entry = {
        "id": f"manual_{bank}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(6)}",
        "bank": bank,
        "topic": topic,
        "content": content,
        "timestamp": timestamp,
        "metadata": {
            **metadata,
            "kind": kind or metadata.get("kind") or "manual_note",
            "source": source,
            "manual_edit": True,
        },
    }
    if chapter_value not in (None, ""):
        entry["metadata"]["chapter"] = chapter_value
    index["claw"].setdefault(bank, []).append(entry)
    _save_memory_index_for_run(run_id, index)

    language = _ui_language(request)
    return JSONResponse(
        {
            "ok": True,
            "run_id": run_id,
            "selected_bank": bank,
            "entry": _serialize_claw_entry(entry),
            "groups": _build_memory_bank_groups_from_index(index, language),
        }
    )


@app.post("/api/runs/{run_id}/memory-banks/{bank}/entries/{entry_id}")
async def api_update_memory_bank_entry(run_id: str, bank: str, entry_id: str, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    if bank not in MemorySystem.CLAW_BANKS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory bank not found")
    book = current_book.get()
    if book:
        run_id = book.memory_key
    elif not _job_for_run_id(db, user.id, run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    body = await _json_payload(request)
    content = str(body.get("content") or "").strip()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Content cannot be empty")

    index = _load_editable_memory_index_for_run(run_id)
    entries = index["claw"].setdefault(bank, [])
    target = next((item for item in entries if str(item.get("id") or "") == entry_id), None)
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory entry not found")

    topic = str(body.get("topic") or "").strip()
    metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    target["content"] = content
    if topic:
        target["topic"] = topic
    target_metadata = target.get("metadata") if isinstance(target.get("metadata"), dict) else {}
    target_metadata.update(metadata)
    target_metadata["manual_edit"] = True
    target_metadata["edited_at"] = datetime.now().isoformat()
    target_metadata["source"] = str(body.get("source") or target_metadata.get("source") or "manual_workspace").strip() or "manual_workspace"
    target["metadata"] = target_metadata
    _save_memory_index_for_run(run_id, index)

    language = _ui_language(request)
    return JSONResponse(
        {
            "ok": True,
            "run_id": run_id,
            "selected_bank": bank,
            "entry": _serialize_claw_entry(target),
            "groups": _build_memory_bank_groups_from_index(index, language),
        }
    )


def _scene_chapter(request, db, run_id, chapter_no):
    user = _require_user(request, db)
    job = _job_for_run_id(db, user.id, run_id)
    if not job or not current_book.get() or job.book_id != current_book.get().id:
        raise HTTPException(404, "Manuscript not found in this book")
    path = _latest_chapter_file(_resolve_run_dir(run_id), chapter_no)
    if not path:
        raise HTTPException(404, "Chapter not found")
    return path


@app.post("/console/manuscript/new-chapter")
def new_manual_chapter(request: Request, run: str = Form(""), after: int = Form(0), db: Session = Depends(get_db)):
    user = _require_user(request, db)
    book = current_book.get()
    if not book:
        raise HTTPException(404, "Choose a book first")
    lock_dir = book_memory_dir(Config(require_api_key=False), book.memory_key)
    lock_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_dir / "chapter-create.lock"), timeout=10):
        job = _job_for_run_id(db, user.id, run) if run else db.scalar(
            select(GenerationJob).where(GenerationJob.book_id == book.id, GenerationJob.run_id != "")
            .order_by(GenerationJob.id.desc()).limit(1))
        if run and (not job or job.book_id != book.id):
            raise HTTPException(404, "Manuscript not found")
        if not job:
            job = GenerationJob(user_id=user.id, book_id=book.id, provider="manual", status="succeeded",
                                idea=book.title, run_id="draft_" + secrets.token_hex(16))
            db.add(job)
        if job.status in {"queued", "running"}:
            raise HTTPException(409, "Wait for the writing run to finish before adding a chapter.")
        folder = _resolve_run_dir(job.run_id)
        existing = latest_chapter_files(folder)
        last = int(existing[-1].name.split("_")[1]) if existing else 0
        if last != after:
            raise HTTPException(409, "The chapter list changed. Refresh before adding another chapter.")
        number = last + 1
        target = folder / "chapters" / f"chapter_{number:02d}_iter_1_final.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch(exist_ok=False)
        try:
            db.commit()
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return _redirect(f"/books/{book.id}/console/manuscript/edit/{job.run_id}/{number}")


@app.get("/console/manuscript/edit/{run_id}/{chapter_no}")
def chapter_scene_editor(run_id: str, chapter_no: int, request: Request, db: Session = Depends(get_db)):
    path = _scene_chapter(request, db, run_id, chapter_no)
    user = _require_user(request, db)
    context = _console_page_context(request, db, user, active_nav="manuscript_read", workspace_section="read")
    job = _job_for_run_id(db, user.id, run_id)
    chat = context.get("active_session")
    selection = job.model_selection or (chat.model_selection if chat else None)
    if selection:
        pinned = json.loads(selection)
        context.update(scene_provider=pinned["slug"], scene_model=pinned["model"])
    return templates.TemplateResponse("scene_editor.html", {
        **context, "prose_levels": prose_style.LEVELS, "book_level": prose_style.load(_prose_folder()).level, "scene_run": run_id, "scene_chapter": chapter_no, "scene_doc": scene_manuscript.document(path),
        "chapter_numbers": [int(p.name.split("_")[1]) for p in latest_chapter_files(_resolve_run_dir(run_id))],
    })


@app.put("/api/runs/{run_id}/chapters/{chapter_no}/scenes")
def save_chapter_scenes(run_id: str, chapter_no: int, edit: scene_manuscript.ChapterEdit,
                        request: Request, db: Session = Depends(get_db)):
    return scene_manuscript.save(_scene_chapter(request, db, run_id, chapter_no), edit)


@app.get("/api/runs/{run_id}/chapters/{chapter_no}/scenes/{scene_id}/context")
def preview_scene_context(run_id: str, chapter_no: int, scene_id: str, revision: str,
                          request: Request, db: Session = Depends(get_db)):
    path = _scene_chapter(request, db, run_id, chapter_no)
    return scene_manuscript.scene_context(_resolve_run_dir(run_id), path, scene_id, revision, prose_style.load(_prose_folder()))


@app.post("/api/runs/{run_id}/chapters/{chapter_no}/scenes/{scene_id}/ask")
def ask_claw_in_scene(run_id: str, chapter_no: int, scene_id: str, request: Request,
                      revision: str = Form(...), question: str = Form("", max_length=12000),
                      action: Literal["ask", "draft", "revise", "review"] = Form("ask"),
                      provider: str = Form(...), model: str = Form(""), db: Session = Depends(get_db)):
    path = _scene_chapter(request, db, run_id, chapter_no)
    user = _require_user(request, db)
    spec = _provider_specs_for_user(db, user.id).get(provider)
    if not spec or _modelless_mode_enabled():
        raise HTTPException(409, "Choose a configured provider to ask Claw.")
    if action == "ask" and not question.strip():
        raise HTTPException(422, "Enter a question or writing instruction.")
    spec = _chosen(spec, model)
    packet = scene_manuscript.scene_context(_resolve_run_dir(run_id), path, scene_id, revision, prose_style.load(_prose_folder()))
    cfg = Config(require_api_key=False)
    try:
        cfg.api_key = _provider_api_key(db, user.id, provider)
    except ValueError:
        raise HTTPException(409, "Check this provider’s key in Models.")
    cfg.model_kind = spec.kind
    cfg.llm_provider, cfg.api_base_url, cfg.model_name, cfg.wire_api = spec.slug, spec.base_url, spec.model, spec.wire_api
    cfg.language = "en"
    try:
        with model_accounting.recording(current_book.get().id, f"chapter {chapter_no} / scene {scene_id}"):
            client = LLMClient(cfg)
            try:
                reply = client.chat(messages=scene_manuscript.scene_messages(action, packet, question.strip()))
            finally:
                client.close()
    except Exception as exc:
        from utils.llm_client import model_failure_message
        raise HTTPException(502, model_failure_message(exc)) from exc
    return {"reply": reply}


@app.post("/api/runs/{run_id}/chapters/{chapter_no}/content")
async def api_update_chapter_content(run_id: str, chapter_no: int, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    if not _job_for_run_id(db, user.id, run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    body = await _json_payload(request)
    content_raw = body.get("content")
    content = str(content_raw if content_raw is not None else "")
    if not content.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Content cannot be empty")

    run_dir = _resolve_run_dir(run_id)
    chapter_path = _latest_chapter_file(run_dir, chapter_no)
    if not chapter_path or not chapter_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chapter output not found")

    normalized = content.rstrip() + "\n"
    if scene_manuscript.metadata_path(chapter_path).exists():
        raise HTTPException(409, "Open Edit chapter & scenes to preserve this chapter’s saved scene boundaries.")
    try:
        save_text_version(chapter_path, normalized, base_content=str(body["base_content"]) if "base_content" in body else None)
    except ContentConflictError:
        raise HTTPException(status_code=409, detail="This chapter has changed. Keep your edits, then refresh and merge them.")
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    title = lines[0] if lines else chapter_path.name
    return JSONResponse(
        {
            "ok": True,
            "run_id": run_id,
            "chapter": {
                "chapter": int(chapter_no),
                "iteration": int(re.search(r"_iter_(\d+)_", chapter_path.name).group(1)) if re.search(r"_iter_(\d+)_", chapter_path.name) else 0,
                "filename": chapter_path.name,
                "title": title,
                "content": normalized.strip(),
            },
        }
    )


@app.post("/api/runs/{run_id}/outlines/{outline_id}")
async def api_update_outline_asset(run_id: str, outline_id: str, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    book = current_book.get()
    if book:
        run_id = book.memory_key
    elif not _job_for_run_id(db, user.id, run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    body = await _json_payload(request)
    content_raw = body.get("content")
    content = str(content_raw if content_raw is not None else "")
    if not content.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Content cannot be empty")

    index = _load_editable_memory_index_for_run(run_id)
    target = _find_memory_bucket_item(index, "outlines", outline_id)
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Outline not found")

    title = str(body.get("title") or "").strip()
    structure = target.get("structure") if isinstance(target.get("structure"), dict) else {}
    if title:
        structure["title"] = title
    target["structure"] = structure
    target["content"] = content.strip()
    target["timestamp"] = datetime.now().isoformat()
    _save_memory_index_for_run(run_id, index)

    return JSONResponse(
        {
            "ok": True,
            "run_id": run_id,
            "outline": {
                "id": str(target.get("id") or ""),
                "kind": str(structure.get("kind") or "outline"),
                "chapter": structure.get("chapter"),
                "title": str(structure.get("title") or body.get("title") or "").strip(),
                "content": str(target.get("content") or "").strip(),
                "timestamp": str(target.get("timestamp") or ""),
                "source": str(structure.get("source") or ""),
            },
        }
    )


# ---------------------------------------------------------------------------
# OpenClaw ask_user IPC endpoints
# The claw worker subprocess writes claw_ask.json when it needs user input.
# The frontend polls /api/runs/{run_id}/pending-question and shows the question.
# When the user replies, POST /api/runs/{run_id}/answer writes claw_reply.json.
# The worker subprocess reads it and continues.
# ---------------------------------------------------------------------------

@app.get("/api/runs/{run_id}/pending-question")
def api_pending_question(run_id: str, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    run_dir = _resolve_run_dir(run_id)
    ask_path = run_dir / "claw_ask.json"
    if not ask_path.exists():
        return JSONResponse({"pending": False, "question": None})
    try:
        payload = json.loads(ask_path.read_text(encoding="utf-8"))
        return JSONResponse({"pending": True, "question": payload.get("question", ""), "ts": payload.get("ts", 0)})
    except Exception:
        return JSONResponse({"pending": False, "question": None})


@app.post("/api/runs/{run_id}/answer")
async def api_submit_answer(run_id: str, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    if not _job_for_run_id(db, user.id, run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    body = await _json_payload(request)
    answer = str(body.get("answer") or "").strip()
    if not answer:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Answer cannot be empty")
    run_dir = _resolve_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    reply_path = run_dir / "claw_reply.json"
    atomic_write_text(reply_path,
        json.dumps({"answer": answer, "ts": __import__("time").time()}, ensure_ascii=False),
        encoding="utf-8",
    )
    return JSONResponse({"ok": True})


@app.post("/api/runs/{run_id}/interrupt")
async def api_interrupt_run(run_id: str, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    if not _job_for_run_id(db, user.id, run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    body = await _json_payload(request)
    message = str(body.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty")

    run_dir = _resolve_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    interrupt_path = run_dir / "claw_interrupt.json"
    payload = {"messages": []}
    if interrupt_path.exists():
        try:
            existing = json.loads(interrupt_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and isinstance(existing.get("messages"), list):
                payload["messages"] = list(existing.get("messages") or [])
            elif isinstance(existing, dict):
                legacy_message = str(existing.get("message") or "").strip()
                if legacy_message:
                    payload["messages"].append(
                        {"message": legacy_message, "ts": float(existing.get("ts") or 0.0)}
                    )
            elif isinstance(existing, list):
                payload["messages"] = list(existing)
        except Exception:
            payload = {"messages": []}

    payload["messages"].append({"message": message, "ts": __import__("time").time()})
    atomic_write_text(interrupt_path, json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return JSONResponse({"ok": True, "queued": len(payload["messages"])})


# ---------------------------------------------------------------------------
# Between-chapter IPC endpoints
# After each chapter the worker writes claw_chapter_complete.json.
# The frontend polls /api/runs/{run_id}/chapter-complete and shows a checkpoint
# card. The user can leave an instruction; POST /api/runs/{run_id}/chapter-message
# writes claw_chapter_msg.json which the worker reads before the next chapter.
# ---------------------------------------------------------------------------

@app.get("/api/runs/{run_id}/chapter-complete")
def api_chapter_complete(run_id: str, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    run_dir = _resolve_run_dir(run_id)
    complete_path = run_dir / "claw_chapter_complete.json"
    if not complete_path.exists():
        return JSONResponse({"pending": False})
    try:
        payload = json.loads(complete_path.read_text(encoding="utf-8"))
        return JSONResponse({"pending": True, **payload})
    except Exception:
        return JSONResponse({"pending": False})


@app.post("/api/runs/{run_id}/chapter-message")
async def api_chapter_message(run_id: str, request: Request, db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    body = await _json_payload(request)
    message = str(body.get("message") or "").strip()
    raw_updates = body.get("memory_updates") if isinstance(body.get("memory_updates"), list) else []
    memory_updates: List[Dict[str, object]] = []
    for item in raw_updates:
        if not isinstance(item, dict):
            continue
        bank = str(item.get("bank") or "").strip()
        content = str(item.get("content") or "").strip()
        if bank not in MemorySystem.CLAW_BANKS or not content:
            continue
        memory_updates.append(
            {
                "bank": bank,
                "content": content,
                "topic": str(item.get("topic") or "").strip(),
                "kind": str(item.get("kind") or "between_chapter_note").strip() or "between_chapter_note",
                "source": str(item.get("source") or "checkpoint_ui").strip() or "checkpoint_ui",
            }
        )
    run_dir = _resolve_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Write instruction for the worker
    if message or memory_updates:
        atomic_write_text(run_dir / "claw_chapter_msg.json",
            json.dumps({"message": message, "memory_updates": memory_updates, "ts": __import__("time").time()}, ensure_ascii=False),
            encoding="utf-8",
        )
    # Always clear the chapter-complete signal so the frontend hides the checkpoint card
    complete_path = run_dir / "claw_chapter_complete.json"
    try:
        complete_path.unlink(missing_ok=True)
    except Exception:
        pass
    return JSONResponse({"ok": True})



@app.get("/books")
def list_books(request: Request, q: str = "", page: int = 1, tool: str = "", db: Session = Depends(get_db)):
    user = _current_user(request, db)
    if not user:
        return _redirect("/login")
    search = q.strip()[:160]
    return templates.TemplateResponse("book_library.html", {
        "request": request, "user": user, "library_view": True, "search": search,
        "tool": book_library.switch_tool(tool), "notice": request.session.pop("library_notice", ""),
        "deleted_book": request.session.pop("removed_book_id", None),
        **book_library.catalog(db, user.id, search, page),
    })


@app.post("/books/{book_id:int}/series")
def set_book_series(request: Request, series: str = Form("", max_length=160), tool: str = "", q: str = "", page: int = 1, db: Session = Depends(get_db)):
    book = current_book.get()
    book.series = series.strip()
    db.commit()
    request.session["library_notice"] = "Series updated."
    return _redirect(book_library.library_location(tool, q, page))


@app.get("/books/{book_id:int}/delete-book")
def confirm_book_deletion(request: Request, tool: str = "", q: str = "", page: int = 1):
    book = current_book.get()
    return templates.TemplateResponse("book_delete.html", {
        "request": request, "library_view": True, "target": book,
        "confirmation": book_library.confirmation.dumps(book_library.receipt(book)),
        "back": book_library.library_location(tool, q, page),
    })


@app.post("/books/{book_id:int}/delete-book")
def delete_book(request: Request, confirmation: str = Form(...), tool: str = "", q: str = "", page: int = 1, db: Session = Depends(get_db)):
    book = current_book.get()
    book_id, title = book.id, book.title
    try:
        with book_activity(book_id, deleting=True):
            fresh = db.get(Book, book_id, populate_existing=True)
            if fresh is None:
                return _redirect(book_library.library_location(tool, q, page))
            book_library.remove_book(db, fresh, confirmation)
    except HTTPException as exc:
        return templates.TemplateResponse("book_delete.html", {
            "request": request, "library_view": True, "target": book, "error": exc.detail,
            "confirmation": book_library.confirmation.dumps(book_library.receipt(book)),
            "back": book_library.library_location(tool, q, page),
        }, status_code=exc.status_code)
    if request.session.get("last_book") == book_id:
        request.session.pop("last_book", None)
    request.session["library_notice"] = f"Deleted {title}."
    request.session["removed_book_id"] = book_id
    return _redirect(book_library.library_location(tool, q, page))


@app.post("/books")
def create_book(request: Request, title: str = Form(..., min_length=1, max_length=160), db: Session = Depends(get_db)):
    user = _require_user(request, db)
    title = title.strip()
    if not title:
        raise HTTPException(422, "Enter a book title")
    book = new_book(db, user.id, title)
    db.commit()
    return _redirect(f"/books/{book.id}/console/chat")


# Register the same views and actions at book-qualified URLs. Links, forms and
# polls use that explicit scope, so another tab cannot redirect an in-flight edit.
for route in list(app.routes):
    if isinstance(route, APIRoute) and workspace_path(route.path):
        app.add_api_route(
            "/books/{book_id:int}" + route.path, route.endpoint,
            methods=route.methods, name="book_" + route.name, include_in_schema=False,
        )
