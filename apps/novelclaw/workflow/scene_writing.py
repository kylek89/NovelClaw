"""Generate a chapter with the same bounded scene evidence used by the editor."""

from pathlib import Path
from uuid import uuid4

from local_web_portal.app import scene_manuscript as manuscript
from utils.artifacts import latest_chapter_files


def source_chapters(folder):
    if not folder:
        return {}
    return {
        int(path.name.split("_")[1]): manuscript.document(path)["scenes"]
        for path in latest_chapter_files(Path(folder))
        if manuscript.metadata_path(path).exists()
    }


def chapter_direction(chapters, chapter, style):
    # Without an authored scene, use the Book default and previous public facts.
    blank = manuscript.Scene(id=uuid4().hex, text="").model_dump()
    packet = manuscript.build_packet({**chapters, chapter: [blank]}, chapter, 0, style)
    return (
        packet["context"]
        + "\nPreserve the requested intensity. Check for POV slips, factual contradictions and unwanted fade-outs."
    )


def write_chapter(client, chapters, chapter, style, brief, target_words):
    rows = chapters.get(chapter) or [
        manuscript.Scene(id=uuid4().hex, text="").model_dump()
    ]
    source = {**chapters, chapter: rows}
    outputs = []
    findings = []
    review_rule = "Reply with PASS if the scene follows the brief, POV, recorded facts and chosen intensity. Otherwise list the specific changes needed."
    for index, scene in enumerate(rows):
        packet = manuscript.build_packet(source, chapter, index, style)
        instruction = (
            f"Book brief (scene-time evidence takes precedence):\n{brief}\n\n"
            f"Write only scene {index + 1} of {len(rows)}. Aim for about {max(1, target_words // len(rows))} words."
        )

        def ask(action, directions, draft=None):
            reply = client.chat(
                messages=manuscript.scene_messages(action, packet, directions, draft)
            ).strip()
            if not reply:
                raise ValueError(
                    f"Claw returned an empty {action} for scene {index + 1}."
                )
            return reply

        draft = ask("draft", instruction)
        review = ask("review", review_rule, draft)
        if review.rstrip(".").upper() != "PASS":
            draft = ask(
                "revise", instruction + "\nRevision feedback:\n" + review, draft
            )
            review = ask("review", review_rule, draft)
            if review.rstrip(".").upper() != "PASS":
                findings.append(f"Scene {index + 1}: {review}")
        outputs.append(
            {**scene, "text": draft + ("\n\n" if index + 1 < len(rows) else "")}
        )
    text = "".join(row["text"] for row in outputs)
    return {
        "story_text": text,
        "scenes": outputs,
        "round_results": [{"role": "writer", "type": "writer", "content": text}],
        "consistency": {"overall_consistent": not findings, "all_issues": findings},
    }
