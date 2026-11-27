"""Portable Book data independent of sessions and generation runs."""

import json
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def new_id():
    return uuid4().hex


ID = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
Text = Annotated[str, Field(max_length=500_000)]
Name = Annotated[str, Field(min_length=1, max_length=180, pattern=r"\S")]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: ID = Field(default_factory=new_id)


class Scene(Record):
    reviewed: bool = False
    title: Name = "Opening"
    time: str = Field(default="", max_length=180)
    pov: str = Field(default="", max_length=180)
    prose: Text = ""
    direction: Text = ""


class Chapter(Record):
    title: Name
    outline: Text = ""
    scenes: list[Scene] = Field(default_factory=lambda: [Scene()], min_length=1)


class Fact(Record):
    scene: ID
    tag: Literal["Physical", "Mental / emotional", "Relationship", "Knowledge", "NSFW"]
    previous: Text = ""
    value: Text = Field(min_length=1)
    excerpt: Text = Field(min_length=1)


class Character(Record):
    name: Name
    profile: Text = ""
    private: Text = ""
    history: list[Fact] = Field(default_factory=list)


class Candidate(BaseModel):
    name: Name
    profile: Text
    saved: ID | None = None


class Message(Record):
    role: Literal["author", "claw"]
    text: Text
    lane: Literal["sfw", "nsfw"] = "sfw"
    provider: str = Field(default="", max_length=250)
    candidates: list[Candidate] = Field(default_factory=list)
    destination: str = Field(default="", max_length=180)


class Chat(Record):
    title: Name = "Book conversation"
    character: ID | None = None
    messages: list[Message] = Field(default_factory=list)


class WritingVersion(Record):
    field: Literal["outline", "prose", "profile", "private", "premise"]
    target: str
    text: Text


class BookData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: Name
    premise: Text = ""
    chapters: list[Chapter] = Field(default_factory=list, max_length=999)
    versions: list[WritingVersion] = Field(default_factory=list)
    cast: list[Character] = Field(default_factory=list)
    chats: list[Chat] = Field(default_factory=lambda: [Chat()], min_length=1)

    @model_validator(mode="after")
    def validate_links(self):
        scenes = [s.id for c in self.chapters for s in c.scenes]
        cast = {c.id for c in self.cast}
        ids = [
            *scenes,
            *(c.id for c in self.chapters),
            *(c.id for c in self.cast),
            *(c.id for c in self.chats),
        ]
        ids.extend(m.id for c in self.chats for m in c.messages)
        ids.extend(f.id for c in self.cast for f in c.history)
        ids.extend(v.id for v in self.versions)
        if len(ids) != len(set(ids)):
            raise ValueError("Book contains duplicate IDs.")
        scene_ids = set(scenes)
        if any(c.character and c.character not in cast for c in self.chats):
            raise ValueError("A chat refers to an unavailable character.")
        if any(f.scene not in scene_ids for c in self.cast for f in c.history):
            raise ValueError("A character change refers to an unavailable scene.")
        return self


def remember(book, field, target, previous, replacement):
    if previous and previous != replacement:
        book.versions.append(WritingVersion(field=field, target=target, text=previous))


def item(items, item_id):
    return next((x for x in items if x.id == item_id), None)


def writing_context(book, chat, lane, chapter=None, scene=None):
    before = []
    eligible = set()
    ordered_scenes = [s for ch in book.chapters for s in ch.scenes]
    order = {s.id: i for i, s in enumerate(ordered_scenes)}
    if chapter and not scene:
        first = chapter.scenes[0]
        limit = order[first.id] - 1
    else:
        limit = order[scene.id] if scene else len(ordered_scenes)
    for index, sc in enumerate(ordered_scenes):
        if index > limit:
            break
        eligible.add(sc.id)
        if (
            scene
            and sc.prose
            and sc.id != scene.id
            and sc.pov.strip().casefold() == scene.pov.strip().casefold()
        ):
            before.append(sc.prose[-2400:])
    cast = []
    for c in book.cast:
        facts = sorted(
            (
                f
                for f in c.history
                if f.scene in eligible
                and f.excerpt in ordered_scenes[order[f.scene]].prose
                and (lane == "nsfw" or f.tag != "NSFW")
                and (
                    not scene
                    or f.tag not in {"Knowledge", "Mental / emotional"}
                    or c.name.casefold() == scene.pov.strip().casefold()
                )
            ),
            key=lambda f: order[f.scene],
        )
        profile = {
            "name": c.name,
            "baseline": c.profile,
            "approved_changes": [f.model_dump() for f in facts],
        }
        if lane == "nsfw":
            profile["private"] = c.private
        cast.append(profile)
    data = {
        "book": book.title,
        "brain_dump": book.premise,
        "cast": cast,
        "focus": next((c.name for c in book.cast if c.id == chat.character), None),
    }
    if chapter:
        index = book.chapters.index(chapter)
        data.update(
            chapter=index + 1,
            outline=chapter.outline,
            earlier_outlines=[c.outline for c in book.chapters[:index]],
        )
    if scene:
        data.update(scene=scene.model_dump(), preceding_prose=before[-2:])
    return json.dumps(data, ensure_ascii=False)
