"""Book-owned prose settings. Private notes stay outside searchable story memory."""

import json
from typing import Literal

from fastapi import HTTPException
from filelock import FileLock
from pydantic import BaseModel, Field
from utils.file_io import atomic_write_text

Level = Literal["brief", "closed", "sensual", "explicit"]
LEVELS = {
    "brief": "Follow the writing brief",
    "closed": "Fade to black",
    "sensual": "Sensual",
    "explicit": "Explicit",
}
GUIDANCE = {
    "brief": "Follow the author's writing brief.",
    "closed": "Keep intimate activity off page. Cut away or move to the aftermath.",
    "sensual": "Keep intimacy on page with sensory and emotional detail, without graphic description.",
    "explicit": "When the scene calls for intimacy, write it on page at the requested detail level. Do not introduce intimacy into unrelated beats.",
}


class BookStyle(BaseModel):
    level: Level = "brief"
    note: str = Field(default="", max_length=3000)
    character_notes: dict[str, str] = Field(default_factory=dict)


def load(folder):
    path = folder / "prose_style.json"
    return (
        BookStyle.model_validate_json(path.read_text(encoding="utf-8"))
        if path.exists()
        else BookStyle()
    )


def update(folder, expected, *, level=None, note="", character=None):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "prose_style.json"
    with FileLock(str(path) + ".lock", timeout=10):
        style = load(folder)
        current = (
            style.character_notes.get(character, "")
            if character is not None
            else [style.level, style.note]
        )
        if current != expected:
            raise HTTPException(
                409, "These settings changed in another tab. Reload before saving."
            )
        if character is None:
            style.level, style.note = level, note.strip()
        elif note.strip():
            style.character_notes[character] = note.strip()
        else:
            style.character_notes.pop(character, None)
        atomic_write_text(path, json.dumps(style.model_dump(), ensure_ascii=False))


def direction(style, level):
    text = f"Prose style: {LEVELS[level]}. {GUIDANCE[level]}"
    if style.note:
        text += "\nAuthor's vocabulary, tone and boundaries: " + style.note
    return text
