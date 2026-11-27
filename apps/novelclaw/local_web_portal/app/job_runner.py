from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from utils.file_io import atomic_write_text, tail_text
from utils.processes import run_logged_process

from .db import SessionLocal
from . import model_connections
from .models import Book, CapabilityPreference, GenerationJob, IdeaCopilotSession
from .provider_registry import (
    ProviderSpec,
    get_provider_specs,
)
from capability_registry import default_enabled_capability_slugs, normalize_capability_slugs
from .settings import BASE_DIR, RUNS_DIR, settings


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _cancel_flag_path(run_id: str) -> Path:
    return RUNS_DIR / run_id / "cancel.flag"


def _is_cancel_requested(run_id: str) -> bool:
    try:
        return _cancel_flag_path(run_id).exists()
    except Exception:
        return False


def _job_status_is_running(db: Session, job_id: int) -> bool:
    # Use a fresh DB session to avoid stale transaction snapshots.
    _ = db  # keep signature stable for existing callers
    with SessionLocal() as s:
        fresh = s.get(GenerationJob, job_id)
        return bool(fresh and fresh.status == "running")


def _enabled_capabilities_for_user(db: Session, user_id: int) -> set[str]:
    rows = (
        db.execute(
            select(CapabilityPreference).where(CapabilityPreference.user_id == user_id)
        )
        .scalars()
        .all()
    )
    enabled = set(default_enabled_capability_slugs())
    for row in rows:
        slug = str(getattr(row, "slug", "") or "").strip().lower()
        if not slug:
            continue
        if getattr(row, "enabled", False):
            enabled.add(slug)
        else:
            enabled.discard(slug)
    return normalize_capability_slugs(enabled)


def _provider_env(
    provider: str,
    api_key: str,
    provider_specs: Optional[dict[str, ProviderSpec]] = None,
    language: Optional[str] = None,
    source_language: Optional[str] = None,
    translation_mode: Optional[str] = None,
    enabled_capabilities: Optional[set[str]] = None,
) -> dict[str, str]:
    provider = provider.lower().strip()
    provider_specs = provider_specs or get_provider_specs(settings)
    spec = provider_specs.get(provider)
    if not spec:
        raise ValueError(f"Unsupported provider: {provider}")

    env = {
        "LLM_PROVIDER": provider,
        "LLM_API_KEY": api_key,
        "LLM_BASE_URL": spec.base_url,
        "MODEL_NAME": spec.model,
        "NOVELCLAW_MODEL_KIND": spec.kind,
        "WIRE_API": spec.wire_api,
        "RUNS_DIR": str(RUNS_DIR),
        "MAX_ITERATIONS": str(settings.max_iterations),
        "MAX_TOTAL_ITERATIONS": str(settings.max_total_iterations),
        "EXECUTION_MODE": settings.execution_mode,
        "CLAW_MAX_STEPS": str(settings.claw_max_steps),
        "LANGUAGE": (language or settings.ui_language or "en").lower(),
        "SOURCE_LANGUAGE": (source_language or language or settings.ui_language or "en").lower(),
        "TRANSLATION_MODE": "english",
        "FAST_MODE": "1" if settings.fast_mode else "0",
        "FULL_CYCLE_INTERVAL": str(settings.full_cycle_interval),
        "CHAPTERS_PER_ITER": str(settings.chapters_per_iter),
        "MAX_CHAPTER_SUBROUNDS": str(settings.max_chapter_subrounds),
        "EVAL_INTERVAL": str(settings.eval_interval),
        "LLM_TIMEOUT_SECONDS": os.getenv("LLM_TIMEOUT_SECONDS", "0"),
        "LLM_MAX_RETRIES": os.getenv("LLM_MAX_RETRIES", "1"),
        # Web portal workers must not trigger HuggingFace embedding downloads.
        "MEMORY_ONLY_MODE": "1",
        "ENABLE_RAG": "0",
        "ENABLE_STATIC_KB": "0",
        "ENABLE_EVALUATOR": os.getenv("WEB_ENABLE_EVALUATOR", "1"),
        "EMBEDDING_MODEL": "none",
        "DISABLE_EMBEDDING_DOWNLOADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        # Force UTF-8 output from worker process on Windows to avoid mojibake.
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        # Force unbuffered stdout/stderr so web tail logs update in near real-time.
        "PYTHONUNBUFFERED": "1",
        "APP_DATABASE_URL": settings.database_url,
        "APP_RUNS_DIR": str(RUNS_DIR),
        "ENABLED_CAPABILITIES": ",".join(sorted(enabled_capabilities or default_enabled_capability_slugs())),
    }

    # Backward-compat env aliases used by legacy config paths.
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_BASE_URL"] = spec.base_url

    if provider == "deepseek":
        env["DEEPSEEK_API_KEY"] = api_key
        env["DEEPSEEK_BASE_URL"] = spec.base_url
    elif provider == "openai":
        env["OPENAI_API_KEY"] = api_key
        env["OPENAI_BASE_URL"] = spec.base_url
    elif provider == "codex":
        env["CODEX_API_KEY"] = api_key
        env["CODEX_BASE_URL"] = spec.base_url

    return env


def _is_en(language: Optional[str]) -> bool:
    return str(language or "").lower().startswith("en")


def _txt(language: Optional[str], zh: str, en: str) -> str:
    return en if _is_en(language) else zh


def _save_job_artifacts(run_id: str, result: dict) -> tuple[str, str]:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    output_path = run_dir / "output.txt"
    result_json_path = run_dir / "result.json"
    round_results_path = run_dir / "round_results.json"
    metadata_path = run_dir / "metadata.json"
    final_text = result.get("final_text", "")
    language = str(result.get("language") or "en").lower()
    is_en = language.startswith("en")

    header_lines = [
        f"Topic: {result.get('topic', '')}" if is_en else f"主题: {result.get('topic', '')}",
        f"Text type: {result.get('text_type', '')}" if is_en else f"文本类型: {result.get('text_type', '')}",
        f"Length: {result.get('length', 0)} chars" if is_en else f"文本长度: {result.get('length', 0)} 字",
        f"Iterations: {result.get('iterations', 0)}" if is_en else f"迭代次数: {result.get('iterations', 0)}",
        "",
        "=" * 60,
        "Generated text:" if is_en else "生成的文本：",
        "=" * 60,
        "",
        final_text,
    ]
    atomic_write_text(output_path, "\n".join(header_lines), encoding="utf-8")

    metadata = {
        "run_id": run_id,
        "topic": result.get("topic", ""),
        "text_type": result.get("text_type", ""),
        "genre": result.get("genre", ""),
        "style_tags": result.get("style_tags", []),
        "length": result.get("length", 0),
        "iterations": result.get("iterations", 0),
        "chapters_written": result.get("chapters_written", 0),
        "best_reward": result.get("best_reward", 0),
        "consistency": result.get("consistency", {}),
        "created_at": datetime.now().isoformat(),
    }
    atomic_write_text(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    atomic_write_text(result_json_path, json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    atomic_write_text(round_results_path,
        json.dumps(result.get("round_results", []), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(output_path), final_text[:600].strip()


def _tail_text(path: Path, max_chars: int = 4000) -> str:
    return tail_text(path, max_chars)


def _run_worker_subprocess(
    idea: str,
    provider: str,
    api_key: str,
    run_id: str,
    provider_specs: Optional[dict[str, ProviderSpec]] = None,
    language: Optional[str] = None,
    source_language: Optional[str] = None,
    translation_mode: Optional[str] = None,
    enabled_capabilities: Optional[set[str]] = None,
    is_running_check: Optional[Callable[[], bool]] = None,
    book_key: str = "",
    usage_book_id: int | None = None,
) -> dict:
    worker_out = RUNS_DIR / run_id / "worker_result.json"
    worker_out.parent.mkdir(parents=True, exist_ok=True)
    worker_log = RUNS_DIR / run_id / "worker.log"

    idea_path = worker_out.parent / "input.txt"
    atomic_write_text(idea_path, idea)

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "local_web_portal.worker_process",
        "--idea-file",
        str(idea_path),
        "--output-json",
        str(worker_out),
    ]

    env = os.environ.copy()
    env.update(
        _provider_env(
            provider,
            api_key,
            provider_specs=provider_specs,
            language=language,
            source_language=source_language,
            translation_mode=translation_mode,
            enabled_capabilities=enabled_capabilities,
        )
    )
    env["MODEL_USAGE_BOOK_ID"] = str(usage_book_id or "")
    env["RUN_ID"] = run_id
    env["NOVELCLAW_BOOK_MEMORY_KEY"] = book_key

    return_code = run_logged_process(
        cmd, cwd=BASE_DIR.parent, env=env, log_path=worker_log,
        cancel_requested=lambda: _is_cancel_requested(run_id),
        is_running=is_running_check,
        timeout=settings.job_timeout_seconds,
        idle_timeout=settings.job_idle_timeout_seconds,
    )

    if not worker_out.exists():
        raise RuntimeError(
            "Worker did not produce output file.\n"
            + f"log_tail:\n{_tail_text(worker_log, max_chars=2500)}"
        )

    payload = json.loads(worker_out.read_text(encoding="utf-8"))
    payload["_stdout"] = _tail_text(worker_log, max_chars=4000)
    payload["_stderr"] = ""
    payload["_returncode"] = return_code
    payload["_worker_log_path"] = str(worker_log)
    return payload


def run_generation_job(job_id: int) -> None:
    db = SessionLocal()
    try:
        # Claim once in the database, including when two requests start together.
        claimed = db.execute(
            update(GenerationJob)
            .where(GenerationJob.id == job_id, GenerationJob.status == "queued")
            .values(status="running", updated_at=_utcnow())
        ).rowcount
        db.commit()
        if not claimed:
            return
        job = db.get(GenerationJob, job_id)


        specs = model_connections.connections(db, job.user_id)
        spec = model_connections.select_model(specs.get(job.provider), snapshot=job.model_selection)
        raw_api_key = model_connections.credential(db, job.user_id, spec)
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_u{job.user_id}_j{job.id}"
        job.run_id = run_id
        job.updated_at = _utcnow()
        db.commit()

        book = db.get(Book, job.book_id) if job.book_id else None
        worker_payload = _run_worker_subprocess(
            job.idea,
            job.provider,
            raw_api_key,
            run_id,
            provider_specs={spec.slug: spec},
            usage_book_id=job.book_id,
            language="en",
            source_language="en",
            translation_mode="english",
            enabled_capabilities=_enabled_capabilities_for_user(db, job.user_id),
            book_key=book.memory_key if book else "",
            is_running_check=lambda: _job_status_is_running(db, job_id),
        )

        if not _job_status_is_running(db, job_id):
            return

        if not worker_payload.get("ok") or worker_payload.get("_returncode", 0) != 0:
            worker_error = worker_payload.get("error", "Unknown worker error.")
            worker_tb = worker_payload.get("traceback", "")
            worker_stderr = worker_payload.get("_stderr", "")
            raise RuntimeError(
                f"{worker_error}\n\n{worker_tb}\n\nworker_stderr:\n{worker_stderr}"
            )

        result = worker_payload.get("result", {})
        output_path, excerpt = _save_job_artifacts(run_id, result)

        # Cancellation wins even if it arrives while artifacts are being saved.
        db.execute(
            update(GenerationJob)
            .where(GenerationJob.id == job_id, GenerationJob.status == "running")
            .values(status="succeeded", run_id=run_id, output_path=output_path,
                    result_excerpt=excerpt, error_message="", finished_at=_utcnow(),
                    updated_at=_utcnow())
        )
        db.commit()
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        try:
            job = db.get(GenerationJob, job_id, populate_existing=True)
            if job and job.status == "running":
                if job.run_id:
                    run_dir = RUNS_DIR / job.run_id
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "error.txt").write_text(str(exc), encoding="utf-8")
                db.execute(
                    update(GenerationJob)
                    .where(GenerationJob.id == job_id, GenerationJob.status == "running")
                    .values(status="failed", error_message=str(exc)[:4000],
                            finished_at=_utcnow(), updated_at=_utcnow())
                )
                db.commit()
        except Exception as db_exc:
            fallback_dir = RUNS_DIR / f"job_{job_id}_db_failure"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            (fallback_dir / "error.txt").write_text(
                f"primary_error={exc}\n\ndb_error={db_exc}",
                encoding="utf-8",
            )
    finally:
        db.close()
