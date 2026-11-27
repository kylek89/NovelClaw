"""Resolve the same connection for the web UI, agent API and subprocess worker."""

from dataclasses import asdict, replace
from decimal import Decimal, InvalidOperation
import json
import os
from threading import Lock
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, HTTPRedirectHandler

from sqlalchemy import select

from .models import ApiCredential, ProviderConfig
from .provider_registry import ProviderSpec, get_provider_specs
from .security import decrypt_api_key
from .settings import settings


def connections(db, user_id):
    specs = get_provider_specs(settings)
    for row in db.scalars(
        select(ProviderConfig).where(ProviderConfig.user_id == user_id)
    ):
        # Preserve custom connections predating the new presets.
        if row.slug not in specs or specs[row.slug].kind in {"local", "openrouter"}:
            specs[row.slug] = ProviderSpec(
                row.slug,
                row.label or row.slug,
                row.base_url,
                row.model,
                row.wire_api,
                row.kind or "custom",
            )
    return specs


def env_key(slug, kind="custom"):
    names = [slug.upper().replace("-", "_") + "_API_KEY"]
    if kind == "custom":
        names.append("LLM_API_KEY")
    for name in names:
        value = os.getenv(name, "").strip()
        if value and not value.lower().startswith(("your-", "change-")):
            return value
    return ""


def credential(db, user_id, spec, *, optional=False):
    row = db.scalar(
        select(ApiCredential).where(
            ApiCredential.user_id == user_id, ApiCredential.provider == spec.slug
        )
    )
    if row:
        try:
            return decrypt_api_key(row.encrypted_key)
        except Exception as exc:
            raise ValueError("Could not unlock the saved provider key.") from exc
    key = env_key(spec.slug, spec.kind)
    if key:
        return key
    if optional:
        return ""
    if spec.kind == "local":
        return "local-no-auth"  # Non-secret placeholder for OpenAI-compatible clients.
    raise ValueError("Save an API key for this provider in Models.")


def select_model(spec, model="", snapshot=None):
    if spec is None:
        raise ValueError("This provider is no longer available.")
    if snapshot:
        saved = ProviderSpec(**json.loads(snapshot))
        if (saved.slug, saved.base_url, saved.kind) != (
            spec.slug,
            spec.base_url,
            spec.kind,
        ):
            raise ValueError(
                "The provider address changed. Choose the connection again before sending."
            )
        return saved
    chosen = (model or spec.model).strip()
    if not chosen or len(chosen) > 256 or any(ord(c) < 32 for c in chosen):
        raise ValueError("Choose a model or enter its model ID.")
    return replace(spec, model=chosen)


def snapshot(spec):
    return json.dumps(asdict(spec))


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_cache = {}
_lock = Lock()


def catalog(spec, key, user_id, refresh=False):
    # Private catalogs never share a cache entry between users or credentials.
    import hashlib

    cache_key = (user_id, spec.base_url, hashlib.sha256(key.encode()).digest())
    with _lock:
        hit = _cache.get(cache_key)
        if hit and not refresh and hit[0] > time.monotonic():
            return hit[1]
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    url = spec.base_url.rstrip("/") + "/models"
    try:
        with build_opener(NoRedirect()).open(
            Request(url, headers=headers), timeout=8
        ) as resp:
            data = resp.read(8_000_001)
        if len(data) > 8_000_000:
            raise ValueError("The model catalog is too large.")
        rows = json.loads(data)["data"]
        if not isinstance(rows, list):
            raise ValueError("Expected a list of models.")
    except HTTPError as exc:
        raise ValueError(
            f"Model server returned HTTP {exc.code}. Check the connection and credentials."
        ) from exc
    except (
        URLError,
        TimeoutError,
        KeyError,
        TypeError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(
            "Could not read the model catalog. Check the server address and availability."
        ) from exc
    models = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            continue
        arch = row.get("architecture") or {}
        if arch.get("output_modalities") and "text" not in arch["output_modalities"]:
            continue
        pricing = row.get("pricing") or {}

        def price(name):
            try:
                value = Decimal(str(pricing[name])) * 1_000_000
                return (
                    format(value.normalize(), "f")
                    if value.is_finite() and value >= 0
                    else None
                )
            except (InvalidOperation, KeyError, TypeError):
                return None

        models.append(
            {
                "id": row["id"],
                "context": row.get("context_length") or row.get("max_model_len"),
                "input_price": price("prompt"),
                "output_price": price("completion"),
                "tools": ("tools" in row["supported_parameters"])
                if isinstance(row.get("supported_parameters"), list)
                else None,
            }
        )
    with _lock:
        now = time.monotonic()
        for old in [k for k, v in _cache.items() if v[0] <= now]:
            del _cache[old]
        _cache[cache_key] = (now + 300, models)
    return models
