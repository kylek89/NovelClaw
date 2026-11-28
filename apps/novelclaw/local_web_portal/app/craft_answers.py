"""Apply a reviewed chat answer to a field on an existing Book entry."""
from datetime import datetime, timezone
from fastapi import HTTPException

FIELDS = {"character": ("Occupation", "Role", "Goal", "Motivation", "Conflict", "Relationships", "Description"),
          "world": ("Rule", "Boundary", "Impact", "Category", "Description")}
BUCKETS = {"character": "characters", "world": "world_settings"}


def entries(index, kind):
    return [r for r in index.get(BUCKETS[kind], []) if r.get("id")]


def fields(row, kind):
    raw = row.get("character_info") or row.get("setting_info") or row.get("content") or ""
    values = {}
    for line in raw.splitlines():
        key, sep, val = line.partition(":")
        if sep:
            values[key.strip().casefold()] = val.strip()
    values.update({key.casefold(): str(val) for key, val in (row.get("attributes") or {}).items()})
    values.update(row.get("author_fields", {}))
    return {field: values.get(field.casefold(), "") for field in FIELDS[kind]}


def apply(index, kind, entry_id, field, previous, value, source):
    row = next((r for r in entries(index, kind) if r["id"] == entry_id), None)
    if not row:
        raise HTTPException(404, "That entry is unavailable in this Book.")
    if field not in FIELDS[kind] or not value.strip() or len(value) > 12000:
        raise HTTPException(422, "Choose a field and enter its new value.")
    value = value.strip()
    history = row.setdefault("author_changes", [])
    if any(change["source"] == source and change["field"] == field and change["value"] == value for change in history):
        return
    if fields(row, kind)[field] != previous:
        raise HTTPException(409, "This field changed. Reopen the answer to see its current value.")
    row.setdefault("author_fields", {})[field.casefold()] = value
    if kind == "character":
        attrs = row.setdefault("attributes", {})
        key = next((key for key in attrs if key.casefold() == field.casefold()), field)
        attrs[key] = value
    row["timestamp"] = datetime.now(timezone.utc).isoformat()
    history.append({"source": source, "field": field, "previous": previous, "value": value})
