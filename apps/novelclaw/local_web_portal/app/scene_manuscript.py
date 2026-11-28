"""Structured scenes alongside the ordinary, downloadable manuscript text."""

import hashlib
import json
from uuid import NAMESPACE_URL, uuid5

from fastapi import HTTPException
from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Literal
from utils.artifacts import latest_chapter_files
from utils.file_io import atomic_write_text
from . import prose_style


class StateFact(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    character: str = Field(min_length=1, max_length=120)
    trait: str = Field(min_length=1, max_length=120)
    value: str = Field(min_length=1, max_length=1200)
    phase: Literal["entry", "change"]
    intimate: bool = False


class Scene(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-f0-9-]{32,36}$")
    text: str
    day: int | None = Field(default=None, ge=1)
    time: str = Field(default="", pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$|^$")
    pov: str = Field(default="", max_length=160)
    participants: str = Field(default="", max_length=1000)
    prose_level: prose_style.Level | None = None
    include_intimate: bool = False
    facts: list[StateFact] = Field(default_factory=list, max_length=500)

    @field_validator("facts")
    @classmethod
    def unique_facts(cls, facts):
        keys = {(f.character.casefold(), f.trait.casefold(), f.phase) for f in facts}
        if len(keys) != len(facts):
            raise ValueError(
                "Use one value for each character, trait and phase in a scene."
            )
        return facts


class ChapterEdit(BaseModel):
    revision: str
    scenes: list[Scene] = Field(min_length=1, max_length=2000)

    @field_validator("scenes")
    @classmethod
    def distinct_ids(cls, scenes):
        if len({s.id for s in scenes}) != len(scenes):
            raise ValueError("Scene identifiers must be distinct.")
        return scenes


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def metadata_path(path):
    return path.with_suffix(".scenes.json")


def _read(path):
    text = path.read_text(encoding="utf-8")
    meta = metadata_path(path)
    if not meta.exists():
        scene_id = uuid5(NAMESPACE_URL, path.name).hex
        return [Scene(id=scene_id, text=text)]
    snapshots = json.loads(meta.read_text(encoding="utf-8"))
    rows = snapshots.get(digest(text))
    if rows is None:
        raise HTTPException(
            409,
            "The manuscript was edited outside the scene editor. Its saved scene boundaries need reconciliation.",
        )
    scenes = []
    start = 0
    for item in rows:
        length = item["length"]
        scenes.append(
            Scene(
                **{k: v for k, v in item.items() if k != "length"},
                text=text[start : start + length],
            )
        )
        start += length
    if start != len(text):
        raise HTTPException(409, "Saved scene boundaries do not match the manuscript.")
    return scenes


def _revision(path, scenes):
    return digest(
        path.name
        + json.dumps(
            [s.model_dump() for s in scenes], ensure_ascii=False, sort_keys=True
        )
    )


def document(path):
    with FileLock(str(path) + ".lock", timeout=10):
        scenes = _read(path)
        return {
            "revision": _revision(path, scenes),
            "scenes": [s.model_dump() for s in scenes],
        }


def _snapshot(scenes):
    return [
        {**scene.model_dump(exclude={"text"}), "length": len(scene.text)}
        for scene in scenes
    ]


def publish_scenes(path, rows):
    """Publish a new worker artifact only after its matching sidecar is durable."""
    scenes = [Scene.model_validate(row) for row in rows]
    text = "".join(scene.text for scene in scenes)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=10):
        if path.exists():
            raise FileExistsError(path)
        atomic_write_text(
            metadata_path(path),
            json.dumps({digest(text): _snapshot(scenes)}, ensure_ascii=False),
        )
        atomic_write_text(path, text)


def save(path, edit):
    with FileLock(str(path) + ".lock", timeout=10):
        old = _read(path)
        if edit.revision != _revision(path, old):
            raise HTTPException(
                409,
                "This chapter changed in another tab. Your unsaved edits are still here.",
            )
        old_text = "".join(s.text for s in old)
        text = "".join(s.text for s in edit.scenes)
        # Either side of an interrupted text replacement has matching boundaries.
        snapshots = {
            digest(old_text): _snapshot(old),
            digest(text): _snapshot(edit.scenes),
        }
        atomic_write_text(
            metadata_path(path), json.dumps(snapshots, ensure_ascii=False)
        )
        if old_text != text:
            atomic_write_text(path, text)
        return {"revision": _revision(path, edit.scenes)}


def scene_context(run_dir, path, scene_id, revision, style=None):
    selected = document(path)
    if revision != selected["revision"]:
        raise HTTPException(409, "The chapter changed. Reload before asking Claw.")
    chapter = int(path.name.split("_")[1])
    scenes = selected["scenes"]
    pos = next((i for i, s in enumerate(scenes) if s["id"] == scene_id), None)
    if pos is None:
        raise HTTPException(404, "Scene not found")
    chapters = {chapter: scenes}
    for other in latest_chapter_files(run_dir):
        if other != path and metadata_path(other).exists():
            chapters[int(other.name.split("_")[1])] = document(other)["scenes"]
    return build_packet(chapters, chapter, pos, style or prose_style.BookStyle())


def build_packet(chapters, chapter, pos, style):
    target = chapters[chapter][pos]
    level = target.get("prose_level") or style.level
    private = target.get("include_intimate") and level in {"sensual", "explicit"}
    cast = {
        name.strip().casefold()
        for name in target.get("participants", "").split(",")
        if name.strip()
    }
    if target.get("pov"):
        cast.add(target["pov"].strip().casefold())
    prior = []
    for number, rows in chapters.items():
        for index, row in enumerate(rows):
            if not row["facts"] or (number, index) == (chapter, pos):
                continue
            manuscript_before = (number, index) < (chapter, pos)
            if target["day"] is None:
                order = (number, index)
                earlier = manuscript_before
            elif row["day"] is None:
                continue
            else:
                day, clock = row["day"], row["time"]
                if day == target["day"] and bool(clock) != bool(target["time"]):
                    continue  # Same-day order is unknown.
                order = (day, clock, number, index)
                earlier = order < (target["day"], target["time"], chapter, pos)
            if earlier:
                prior.append((order, row))
    state = {}

    def apply(facts, phase):
        for fact in facts:
            if fact["phase"] == phase:
                state[(fact["character"].casefold(), fact["trait"].casefold())] = fact

    for _, row in sorted(prior, key=lambda item: item[0]):
        apply(row["facts"], "entry")
        apply(row["facts"], "change")
    apply(target["facts"], "entry")

    def line(f):
        return f"{f['character']} · {f['trait']}: {f['value']}"

    def visible(fact):
        return not fact.get("intimate") or (
            private and fact["character"].casefold() in cast
        )

    entry = (
        "\n".join(line(f) for f in state.values() if visible(f))
        or "No established state."
    )
    changes = (
        "\n".join(
            line(f) for f in target["facts"] if f["phase"] == "change" and visible(f)
        )
        or "No recorded changes."
    )
    context = (
        f"Chapter {chapter}, scene {pos + 1}\nStory day: {target['day'] or 'unspecified'}"
        f"\nTime: {target['time'] or 'unspecified'}\nPOV: {target.get('pov') or 'unspecified'}"
        f"\nScene cast: {target.get('participants') or 'unspecified'}"
        f"\n{prose_style.direction(style, level)}\n\nOn entry\n{entry}\n\n"
        f"Changes during this scene\n{changes}\n\nScene prose\n{target['text']}"
    )

    if private:
        notes = [
            f"{name}: {style.character_notes[name]}"
            for name in sorted(cast)
            if name in style.character_notes
        ]
        if notes:
            context += (
                "\n\nIntimate character notes (background, subordinate to scene-time facts)\n"
                + "\n".join(notes)
            )
    return {"context": context, "on_entry": entry}


ACTIONS = {
    "ask": "Answer the author's question or writing instruction.",
    "draft": "Draft this scene. Return only narrative prose.",
    "revise": "Revise this scene. Preserve its story beats and the requested intensity. Return only revised prose.",
    "review": "Review the prose for conflicts with scene-time facts, relationship state, POV slips, boundaries, or an unwanted fade-out. Preserve the requested intensity. Identify specific issues, or say that none were found.",
}


def scene_messages(action, packet, instruction="", draft=None):
    parts = [
        {
            "role": "system",
            "content": "You are Claw, the author's scene writing partner. Use the provided scene evidence. Entry facts apply until a recorded change. Keep the POV character's knowledge distinct from author knowledge. "
            + ACTIONS[action],
        },
        {"role": "user", "content": packet["context"]},
    ]
    if draft is not None:
        parts.append(
            {"role": "user", "content": "Current draft for this scene:\n" + draft}
        )
    if instruction:
        parts.append({"role": "user", "content": instruction})
    return parts
