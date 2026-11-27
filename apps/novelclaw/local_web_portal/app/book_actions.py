from fastapi import HTTPException
from .book_data import BookData, Chapter, Character, Chat, Fact, Scene, item, remember


def require(items, key):
    found = item(items, key)
    if found is None:
        raise HTTPException(404, "This item is unavailable in this Book.")
    return found


def change(book: BookData, cmd):
    """Apply one explicit author edit. IDs in the response open the saved destination."""
    action = cmd.action
    if action == "review_next":
        ch = require(book.chapters, cmd.chapter)
        sc = require(ch.scenes, cmd.scene)
        if not sc.prose.strip():
            raise HTTPException(
                422, "Save some prose before marking this scene reviewed."
            )
        sc.reviewed = True
        ordered = [(c, s) for c in book.chapters for s in c.scenes]
        index = next(i for i, (_, s) in enumerate(ordered) if s.id == sc.id)
        following, scene = ordered[min(index + 1, len(ordered) - 1)]
        return {"chapter": following.id, "scene": scene.id}
    elif action == "premise":
        remember(book, "premise", "book", book.premise, cmd.text)
        book.title, book.premise = cmd.name, cmd.text
    elif action == "chapter_add":
        ch = Chapter(title=cmd.name or f"Chapter {len(book.chapters) + 1}")
        book.chapters.append(ch)
        return {"chapter": ch.id}
    elif action in ("outline", "scene_add", "scene", "split"):
        ch = require(book.chapters, cmd.chapter)
        if action == "outline":
            remember(book, "outline", ch.id, ch.outline, cmd.text)
            ch.outline = cmd.text
            if cmd.name:
                ch.title = cmd.name
        elif action == "scene_add":
            sc = Scene(title=cmd.name or f"Scene {len(ch.scenes) + 1}")
            ch.scenes.append(sc)
            return {"chapter": ch.id, "scene": sc.id}
        else:
            sc = require(ch.scenes, cmd.scene)
            remember(
                book,
                "prose",
                sc.id,
                sc.prose,
                cmd.text if action == "scene" else cmd.text[: cmd.offset],
            )
            if (sc.prose, sc.time, sc.pov, sc.direction) != (
                cmd.text,
                cmd.time,
                cmd.pov,
                cmd.direction,
            ) or action == "split":
                sc.reviewed = False
            sc.prose, sc.time, sc.pov, sc.direction = (
                cmd.text,
                cmd.time,
                cmd.pov,
                cmd.direction,
            )
            if cmd.name:
                sc.title = cmd.name
            if action == "split":
                if not 0 < cmd.offset < len(sc.prose):
                    raise HTTPException(
                        422, "Place the cursor between the two passages."
                    )
                next_scene = Scene(
                    title=f"Scene {len(ch.scenes) + 1}",
                    prose=sc.prose[cmd.offset :],
                    pov=sc.pov,
                )
                sc.prose = sc.prose[: cmd.offset]
                ch.scenes.insert(ch.scenes.index(sc) + 1, next_scene)
                return {"chapter": ch.id, "scene": next_scene.id}
    elif action == "save_candidates":
        chat = require(book.chats, cmd.chat)
        msg = require(chat.messages, cmd.message)
        for picked in cmd.picks:
            if not 0 <= picked.index < len(msg.candidates):
                raise HTTPException(422, "Invalid cast selection.")
            source = msg.candidates[picked.index]
            if source.saved:
                continue
            c = (
                Character(name=picked.name, private=picked.profile)
                if msg.lane == "nsfw"
                else Character(name=picked.name, profile=picked.profile)
            )
            book.cast.append(c)
            source.saved = c.id
    elif action == "character_add":
        c = Character(name=cmd.name, profile=cmd.text)
        book.cast.append(c)
        return {"character": c.id}
    elif action == "character":
        c = require(book.cast, cmd.character)
        remember(book, "profile", c.id, c.profile, cmd.text)
        remember(book, "private", c.id, c.private, cmd.private)
        c.name, c.profile, c.private = cmd.name, cmd.text, cmd.private
    elif action == "profile_capture":
        chat = require(book.chats, cmd.chat)
        c = require(book.cast, chat.character)
        reply = require(chat.messages, cmd.message)
        if reply.role != "claw" or not cmd.selected:
            raise HTTPException(422, "Select profile details from a Claw reply.")
        parts = reply.text.split("\n\n")
        if any(i < 0 or i >= len(parts) for i in cmd.selected):
            raise HTTPException(422, "The selection is invalid.")
        target = "private" if reply.lane == "nsfw" else "profile"
        existing = getattr(c, target).split("\n\n")
        additions = [
            parts[i]
            for i in sorted(set(cmd.selected))
            if parts[i].strip() and parts[i] not in existing
        ]
        if additions:
            remember(book, target, c.id, getattr(c, target), additions[0])
            setattr(
                c, target, "\n\n".join(x for x in [getattr(c, target), *additions] if x)
            )
    elif action == "fact":
        c = require(book.cast, cmd.character)
        ch = require(book.chapters, cmd.chapter)
        sc = require(ch.scenes, cmd.scene)
        if not cmd.excerpt.strip() or cmd.excerpt not in sc.prose:
            raise HTTPException(
                422, "Quote the source scene exactly before approving this change."
            )
        c.history.append(
            Fact(
                scene=sc.id,
                tag=cmd.tag,
                previous=cmd.previous,
                value=cmd.text,
                excerpt=cmd.excerpt,
            )
        )
    elif action in ("chat_add", "develop"):
        if cmd.character:
            c = require(book.cast, cmd.character)
            if action == "develop":
                existing = next((x for x in book.chats if x.character == c.id), None)
                if existing:
                    return {"chat": existing.id}
            title = cmd.name or f"Develop {c.name}"
        else:
            title = cmd.name or "Book conversation"
        chat = Chat(title=title, character=cmd.character)
        book.chats.append(chat)
        return {"chat": chat.id}
    else:
        raise HTTPException(422, "Unknown Book edit.")
    return {}
