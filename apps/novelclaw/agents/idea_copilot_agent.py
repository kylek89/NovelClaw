from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config import Config
from utils.llm_client import LLMClient


@dataclass(frozen=True)
class ProviderRuntime:
    slug: str
    base_url: str
    model: str
    wire_api: str = "chat"
    kind: str = "custom"


class IdeaCopilotAgent:
    """English-language collaborative ideation agent for NovelClaw."""

    def __init__(self, provider_spec: Any, api_key: str):
        self.provider = _to_provider_runtime(provider_spec)
        self.api_key = api_key

    def generate_turn(
        self,
        *,
        original_idea: str,
        state: Dict[str, Any],
        latest_user_reply: str,
    ) -> Dict[str, Any]:
        return generate_assistant_turn(
            original_idea=original_idea,
            state=state,
            latest_user_reply=latest_user_reply,
            provider_spec=self.provider,
            api_key=self.api_key,
        )

    @staticmethod
    def load_state(raw: str) -> Dict[str, Any]:
        return load_state(raw)

    @staticmethod
    def dump_state(state: Dict[str, Any]) -> str:
        return dump_state(state)

    @staticmethod
    def append_user(state: Dict[str, Any], reply: str) -> Dict[str, Any]:
        return append_user_reply(state, reply)

    @staticmethod
    def append_assistant(state: Dict[str, Any], turn: Dict[str, Any]) -> Dict[str, Any]:
        return append_assistant_turn(state, turn)

    @staticmethod
    def latest_turn(state: Dict[str, Any]) -> Dict[str, Any]:
        return latest_assistant_turn(state)

    @staticmethod
    def to_generation_idea(original_idea: str, state: Dict[str, Any]) -> str:
        return build_generation_idea(original_idea, state)


GENERATION_SCOPES = {"auto", "all", "limited", "chapter_by_chapter"}
CHAPTER_PAUSE_MODES = {"manual_each_chapter", "run_to_end", "auto"}


def default_generation_preferences() -> Dict[str, Any]:
    return {
        "generation_scope": "auto",
        "requested_chapters": 0,
        "chapter_pause_mode": "manual_each_chapter",
        "user_request": "",
    }


def normalize_generation_preferences(value: Any) -> Dict[str, Any]:
    out = dict(default_generation_preferences())
    if isinstance(value, dict):
        scope = str(value.get("generation_scope") or out["generation_scope"]).strip().lower()
        if scope in GENERATION_SCOPES:
            out["generation_scope"] = scope
        pause_mode = str(value.get("chapter_pause_mode") or out["chapter_pause_mode"]).strip().lower()
        if pause_mode in CHAPTER_PAUSE_MODES:
            out["chapter_pause_mode"] = pause_mode
        try:
            requested = max(0, int(value.get("requested_chapters") or 0))
        except Exception:
            requested = 0
        out["requested_chapters"] = requested
        out["user_request"] = str(value.get("user_request") or "").strip()[:400]

    if out["generation_scope"] == "limited" and out["requested_chapters"] <= 0:
        out["requested_chapters"] = 1
    if out["generation_scope"] == "all":
        out["requested_chapters"] = 0
    if out["generation_scope"] == "chapter_by_chapter":
        out["chapter_pause_mode"] = "manual_each_chapter"
    if out["chapter_pause_mode"] == "auto":
        out["chapter_pause_mode"] = (
            "run_to_end" if out["generation_scope"] in {"all", "limited"} else "manual_each_chapter"
        )
    return out


def merge_generation_preferences(state: Dict[str, Any], updates: Any) -> Dict[str, Any]:
    merged = normalize_generation_preferences(state.get("generation_preferences"))
    if isinstance(updates, dict):
        candidate = dict(merged)
        for key in ("generation_scope", "requested_chapters", "chapter_pause_mode", "user_request"):
            if key in updates and updates.get(key) not in {None, ""}:
                candidate[key] = updates.get(key)
        merged = normalize_generation_preferences(candidate)
    return merged


def _to_provider_runtime(spec: Any) -> ProviderRuntime:
    if isinstance(spec, ProviderRuntime):
        return spec
    if isinstance(spec, dict):
        return ProviderRuntime(
            slug=str(spec.get("slug") or ""),
            base_url=str(spec.get("base_url") or ""),
            model=str(spec.get("model") or ""),
            wire_api=str(spec.get("wire_api") or "chat"),
            kind=spec.get("kind","custom"),
        )
    return ProviderRuntime(
        slug=str(getattr(spec, "slug", "") or ""),
        base_url=str(getattr(spec, "base_url", "") or ""),
        model=str(getattr(spec, "model", "") or ""),
        wire_api=str(getattr(spec, "wire_api", "chat") or "chat"),
        kind=getattr(spec,"kind","custom"),
    )


def load_state(raw: str) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "version": 3,
        "messages": [],
        "refined_idea": "",
        "round": 0,
        "preferred_language": "en",
        "source_language": "en",
        "ui_language": "en",
        "translation_mode": "english",
        "generation_preferences": default_generation_preferences(),
        "reply_pending": False,
        "reply_error": "",
        "reply_started_at": "",
    }
    if not raw:
        return base
    try:
        data = json.loads(raw)
    except Exception:
        return base
    if not isinstance(data, dict):
        return base
    out = dict(base)
    msgs = data.get("messages")
    if isinstance(msgs, list):
        out["messages"] = msgs
    out["refined_idea"] = str(data.get("refined_idea") or "")
    out["author_edited_summary"] = str(data.get("author_edited_summary") or "")
    out["preferred_language"] = "en"
    out["source_language"] = str(data.get("source_language") or out["preferred_language"] or "en").lower()
    out["ui_language"] = "en"
    out["translation_mode"] = "english"
    out["generation_preferences"] = normalize_generation_preferences(data.get("generation_preferences"))
    out["reply_pending"] = bool(data.get("reply_pending") or False)
    out["reply_error"] = str(data.get("reply_error") or "")
    out["reply_started_at"] = str(data.get("reply_started_at") or "")
    try:
        out["round"] = max(0, int(data.get("round") or 0))
    except Exception:
        out["round"] = 0
    return out


def dump_state(state: Dict[str, Any]) -> str:
    payload = {
        "version": 3,
        "messages": state.get("messages", []),
        "refined_idea": state.get("refined_idea", ""),
        "author_edited_summary": state.get("author_edited_summary", ""),
        "round": int(state.get("round", 0) or 0),
        "preferred_language": state.get("preferred_language", "en"),
        "source_language": state.get("source_language", state.get("preferred_language", "en")),
        "ui_language": state.get("ui_language", "en"),
        "translation_mode": state.get("translation_mode", "english"),
        "generation_preferences": normalize_generation_preferences(state.get("generation_preferences")),
        "reply_pending": bool(state.get("reply_pending") or False),
        "reply_error": str(state.get("reply_error") or ""),
        "reply_started_at": str(state.get("reply_started_at") or ""),
    }
    return json.dumps(payload, ensure_ascii=False)


def _build_client(spec: ProviderRuntime, api_key: str, preferred_language: str) -> LLMClient:
    cfg = Config(require_api_key=False)
    cfg.language = "en"
    cfg.llm_provider = spec.slug
    cfg.api_key = api_key
    cfg.api_base_url = spec.base_url
    cfg.model_kind = spec.kind
    cfg.model_name = spec.model
    cfg.wire_api = spec.wire_api
    cfg.temperature = 0.6
    cfg.max_tokens = min(int(getattr(cfg, "max_tokens", 1600) or 1600), 1600)
    return LLMClient(cfg)


def _extract_json_object(text: str) -> Optional[str]:
    s = (text or "").strip()
    if not s:
        return None
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _history_text(messages: List[Dict[str, Any]], limit: int = 12) -> str:
    if not messages:
        return "(no prior conversation yet)"
    rows: List[str] = []
    recent = messages[-limit:]
    last_assistant = next((item for item in reversed(recent) if item.get("role") == "assistant"), None)
    for item in recent:
        role = str(item.get("role") or "").strip().lower()
        if role == "assistant":
            analysis = str(item.get("analysis") or "").strip()
            refined = str(item.get("refined_idea") or "").strip()
            questions = item.get("questions") or []
            answer = analysis if item is last_assistant else analysis[:300]
            row = f"assistant analysis: {answer}"
            if refined:
                row += f" | brief: {refined[:300]}"
            if isinstance(questions, list) and questions:
                row += " | questions: " + " ; ".join(str(q) for q in _normalize_questions(questions))
                if item.get("question_options"):
                    row += " | suggested answers: " + format_options(questions, item["question_options"])
            rows.append(row)
        elif role == "user":
            rows.append(f"user: {str(item.get('content') or '').strip()[:400]}")
            if item.get("selected_options"):
                rows.append("Author chose: " + " | ".join(item["selected_options"]))
    return "\n".join(rows) if rows else "(no prior conversation yet)"


def _normalize_questions(value: Any) -> List[str]:
    if isinstance(value, list):
        out = []
        for item in value[:3]:
            text = str(item or "").strip()
            if text:
                out.append(text[:200])
        return out
    if isinstance(value, str) and value.strip():
        return [value.strip()[:200]]
    return []


def requested_questions(state, reply):
    """Resolve direct requests about a saved question set, including after a missed reply."""
    text = " ".join(reply.lower().split()).strip(" .!?")
    wants_options = re.fullmatch(
        r"(?:please )?(?:give|show|suggest)(?: me)? (?:three|3) (?:possible |different )?"
        r"(?:answers|options|choices) (?:for|to) (?:each(?: of (?:the|your|those|these))? "
        r"|(?:the|your|those|these) (?:three |3 )?)questions?", text
    )
    if not wants_options and not re.fullmatch(
        r"(?:(?:can|could|would) you )?(?:please )?"
        r"(?:answer|help(?: me| us)? (?:with|answer|to answer|work through)|"
        r"suggest answers (?:to|for)|give me answers (?:to|for)|work through) "
        r"(?:(?:all|the|your|those|these|original|last|three|3) )*"
        r"questions(?: you asked)?(?: please)?", text
    ):
        return []
    messages = state.get("messages") or []
    candidates = messages if "original" in text else reversed(messages)
    for item in candidates:
        if item.get("role") == "assistant" and item.get("questions"):
            return _normalize_questions(item["questions"])
    return []


def valid_options(groups, questions):
    return (isinstance(groups, list) and len(groups) == len(questions)
            and all(isinstance(group, list) and len(group) == 3
                    and all(isinstance(text, str) and text.strip() for text in group)
                    and len({text.strip().casefold() for text in group}) == 3 for group in groups))


def format_options(questions, groups):
    return "\n".join(
        f"{i}. {q}\n" + "\n".join(f"{i}{letter}. {choice}" for letter, choice in zip("ABC", choices))
        for i, (q, choices) in enumerate(zip(questions, groups), 1)
    )


def _generate_options(client, state, original_idea, questions, latest_user_reply):
    context = {
        "author_premise": original_idea,
        "current_brief": state.get("refined_idea") or original_idea,
        "book_notes": state.get("book_notes") or "",
        "recent_author_messages": [item.get("content", "") for item in state.get("messages", [])
                                  if item.get("role") == "user"][-12:],
        "questions_in_order": questions,
        "request": latest_user_reply,
    }
    raw = client.chat(messages=[
        {"role": "system", "content": (
            "For EACH listed question give THREE distinctly different possible answers a novelist "
            "could use. Each choice is one short, concrete story idea grounded in the author's premise. "
            "Return only JSON with the key question_options and an array of arrays of strings. "
            "One inner array per question, in order, with exactly three strings each. Omit labels."
        )},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ], temperature=0.6, max_tokens=1400)
    try:
        payload = json.loads(_extract_json_object(raw or "") or "{}")
    except ValueError:
        payload = {}
    groups = payload.get("question_options") if isinstance(payload, dict) else None
    if not valid_options(groups, questions):
        raise ValueError("The model did not supply three distinct choices for every question. Try again or choose another model.")
    return groups


def _normalize_turn(parsed: Dict[str, Any], fallback_refined: str, fallback_text: str, preferred_language: str) -> Dict[str, Any]:
    refined = str(parsed.get("refined_idea") or fallback_refined or "").strip()
    analysis = str(
        parsed.get("analysis")
        or fallback_text
        or (
            "Please refine the premise, constraints, and intended output."
        )
    ).strip()
    questions = _normalize_questions(parsed.get("questions"))
    try:
        readiness = max(0, min(100, int(parsed.get("readiness") or 0)))
    except Exception:
        readiness = 0
    ready_hint = str(
        parsed.get("ready_hint")
        or (
            "Keep clarifying the premise, world rules, character arcs, and style goals."
        )
    ).strip()
    style_targets = _normalize_questions(parsed.get("style_targets"))
    memory_targets = _normalize_questions(parsed.get("memory_targets"))
    return {
        "role": "assistant",
        "analysis": analysis[:2400],
        "refined_idea": refined[:5000],
        "questions": questions,
        "readiness": readiness,
        "ready_hint": ready_hint[:600],
        "language": "en",
        "style_targets": style_targets,
        "memory_targets": memory_targets,
    }


def _fallback_turn(state: Dict[str, Any], original_idea: str, raw_response: str) -> Dict[str, Any]:
    preferred_language = "en"
    fallback_refined = str(state.get("refined_idea") or "").strip() or original_idea.strip()
    return _normalize_turn(
        parsed={},
        fallback_refined=fallback_refined,
        fallback_text=raw_response or (
            "I need a bit more detail about the story, style, and constraints before moving forward."
        ),
        preferred_language=preferred_language,
    )


def generate_assistant_turn(
    *,
    original_idea: str,
    state: Dict[str, Any],
    latest_user_reply: str,
    provider_spec: Any,
    api_key: str,
) -> Dict[str, Any]:
    runtime = _to_provider_runtime(provider_spec)
    preferred_language = "en"
    source_language = "en"
    client = _build_client(runtime, api_key, preferred_language)
    questions = requested_questions(state, latest_user_reply)
    if questions:
        groups = _generate_options(client, state, original_idea, questions, latest_user_reply)
        return {
            "role": "assistant", "language": "en", "questions": questions, "question_options": groups,
            "analysis": "Here are three possible answers for each question.",
            "refined_idea": state.get("refined_idea") or original_idea,
            "readiness": latest_assistant_turn(state).get("readiness", 0),
            "ready_hint": "Pick a letter for each question, mix choices, or give your own answer.",
            "style_targets": [], "memory_targets": [],
        }
    current_refined = str(state.get("refined_idea") or "").strip() or original_idea.strip()
    history = _history_text(state.get("messages") or [])

    system_prompt = f"""You are NovelClaw's collaborative ideation agent.
Goal: turn a raw story idea into a stable writing brief that can drive a real agentic writing loop.
Rules:
1) Answer the author's latest request in the visible "analysis" field. When asked for help with your questions, give concrete possible answers grounded in the story, one numbered answer per requested question in its original order. If the author refers to an earlier set or a single question, use that set or question.
2) Ask up to three questions only when clarification will help. When the author asks for answers or suggestions, return "questions": [] unless they also ask for new questions. Keep suggestions provisional and preserve the current brief and targets until the author chooses or adds story details.
3) Focus on conflict, cast, world rules, voice, structure, and hard constraints.
4) If the user clearly asks to start writing or generation, prepare the brief for execution instead of forcing more refinement.
5) Output strict JSON only, in English.
6) For EVERY new question, include three distinct, concrete possible answers in question_options.
Use one inner array of three short strings per question, in order. An author can reply with codes
like 1B, 2A, 3C. Use only their selected or custom answers when updating the brief.
7) Suggest a destination for each question in question_destinations: "premise" for lasting
story rules and setup, "author_brief" for book-wide writing instructions, "chat" for everything
else, including alternate possibilities, chapter plans and scene-specific details.
JSON schema:
{{
  "analysis": "string",
  "refined_idea": "string",
  "questions": [],
  "question_options": [],
  "question_destinations": [],
  "readiness": 0,
  "ready_hint": "string",
  "language": "en",
  "style_targets": ["voice or tone target"],
  "memory_targets": ["memory item that should be remembered"]
}}
"""

    user_prompt = f"""[original idea]
{original_idea}

[current brief]
{current_refined}

[source language]
{source_language}

[target language]
{preferred_language}

[book notes]
{state.get("book_notes") or "(none)"}

[recent conversation]
{history}

[author's latest request]
{latest_user_reply or '(Opening turn: help the author develop the original idea.)'}
"""

    raw = client.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.6,
        max_tokens=1400,
    )

    block = _extract_json_object(raw or "")
    if not block:
        return _fallback_turn(state, original_idea, raw or "")
    try:
        parsed = json.loads(block)
    except Exception:
        return _fallback_turn(state, original_idea, raw or "")
    if not isinstance(parsed, dict):
        return _fallback_turn(state, original_idea, raw or "")
    turn = _normalize_turn(
        parsed=parsed,
        fallback_refined=current_refined,
        fallback_text=raw or "",
        preferred_language=preferred_language,
    )
    if turn["questions"]:
        destinations = parsed.get("question_destinations")
        if isinstance(destinations, list) and len(destinations) == len(turn["questions"]):
            turn["question_destinations"] = [
                kind if kind in ("premise", "author_brief") else "chat" for kind in destinations
            ]
        groups = parsed.get("question_options")
        if not valid_options(groups, turn["questions"]):
            groups = _generate_options(client, state, original_idea, turn["questions"], latest_user_reply)
        turn["question_options"] = groups
    return turn


def resolve_question_answers(state, packet):
    """Resolve the exact card the author saw, including older saved cards."""
    if not isinstance(packet, dict):
        raise ValueError("Invalid answer card.")
    for turn in reversed(state.get("messages", [])):
        if (turn.get("role") == "assistant" and turn.get("questions")
                and turn["questions"] == packet.get("questions")
                and (turn.get("question_options") or []) == packet.get("options")):
            break
    else:
        raise ValueError("These questions have changed. Refresh the chat before sending.")
    picks = packet.get("picks")
    if not isinstance(picks, list) or len(picks) != len(turn["questions"]):
        raise ValueError("Invalid answer selection.")
    answers = []
    for index, pick in enumerate(picks):
        if pick is None:
            continue
        options = turn.get("question_options") or []
        if type(pick) is int and index < len(options) and 0 <= pick < len(options[index]):
            text = options[index][pick]
        elif isinstance(pick, str) and pick.strip() and len(pick) <= 2000:
            text = pick.strip()
        else:
            raise ValueError("Choose an answer or enter up to 2,000 characters of your own.")
        answers.append(f"{turn['questions'][index]} — {text}")
    if not answers:
        raise ValueError("Choose at least one answer.")
    return answers


def append_user_reply(state: Dict[str, Any], reply: str, *, selected_options=None) -> Dict[str, Any]:
    content = (reply or "").strip()
    if not content:
        return state
    out = load_state(dump_state(state))
    msgs = list(out.get("messages") or [])
    message = {"role": "user", "content": content[:3000]}
    if selected_options:
        message.update(content=content, selected_options=selected_options)
    elif re.fullmatch(r"[1-3]\s*-?\s*[ABC](?:[\s,;]+[1-3]\s*-?\s*[ABC])*[.!]?", content, re.I):
        prior = next((m for m in reversed(msgs) if m.get("role") == "assistant" and m.get("questions")), {})
        questions = prior.get("questions", [])
        groups = prior.get("question_options")
        if questions and valid_options(groups, questions):
            picks = re.findall(r"([1-3])\s*-?\s*([ABC])", content.upper())
            if all(int(n) <= len(groups) for n, _ in picks):
                message["selected_options"] = [
                    f"{questions[int(n)-1]} — {groups[int(n)-1]['ABC'.index(letter)]}"
                    for n, letter in picks
                ]
    msgs.append(message)
    out["messages"] = msgs[-30:]
    return out


def append_assistant_turn(state: Dict[str, Any], turn: Dict[str, Any]) -> Dict[str, Any]:
    out = load_state(dump_state(state))
    msgs = list(out.get("messages") or [])
    msgs.append(turn)
    out["messages"] = msgs[-30:]
    out["refined_idea"] = str(turn.get("refined_idea") or out.get("refined_idea") or "")
    out["round"] = int(out.get("round", 0) or 0) + 1
    return out


def latest_assistant_turn(state: Dict[str, Any]) -> Dict[str, Any]:
    msgs = list(state.get("messages") or [])
    for item in reversed(msgs):
        if str(item.get("role") or "") == "assistant":
            return item
    return {
        "role": "assistant",
        "analysis": (
            "Start a writing session and I will refine the idea into a stronger writing brief."
        ),
        "refined_idea": state.get("refined_idea") or "",
        "questions": [],
        "readiness": 0,
        "ready_hint": "",
        "language": str(state.get("preferred_language") or "en"),
        "style_targets": [],
        "memory_targets": [],
    }


def build_generation_idea(original_idea: str, state: Dict[str, Any], *, approved_documents=None) -> str:
    refined = str(state.get("refined_idea") or "").strip() or original_idea.strip()
    messages = list(state.get("messages") or [])
    generation_preferences = normalize_generation_preferences(state.get("generation_preferences"))
    if approved_documents is not None:
        parts = []
        if "premise" not in approved_documents:
            parts.append("[Starting idea]\n" + (state.get("author_edited_summary") or original_idea))
        decided_questions = {line.partition(" — ")[0] for doc in approved_documents.values()
                             for line in doc.splitlines() if " — " in line}
        chosen = []
        for item in messages:
            if item.get("role") != "user":
                continue
            if "chat_options" in item or "selected_options" in item:
                answers = item.get("chat_options", item.get("selected_options", []))
                chosen.extend(answer for answer in answers if answer.partition(" — ")[0] not in decided_questions)
            elif item.get("content") and not approved_documents and item["content"] != original_idea:
                chosen.append(item["content"])
        if chosen:
            parts.append("[Choices for this conversation]\n" + "\n".join(dict.fromkeys(chosen)))
        parts.append("[Execution preference]\n" + json.dumps(generation_preferences, ensure_ascii=False))
        return "\n\n".join(parts)
    qa: List[str] = []
    style_targets: List[str] = []
    memory_targets: List[str] = []
    for item in messages[-20:]:
        role = str(item.get("role") or "")
        if role == "assistant":
            qs = item.get("questions") or []
            if isinstance(qs, list):
                for q in qs[:3]:
                    q_text = str(q or "").strip()
                    if q_text:
                        qa.append(f"Q: {q_text}")
            for token in item.get("style_targets") or []:
                t = str(token or "").strip()
                if t and t not in style_targets:
                    style_targets.append(t)
            for token in item.get("memory_targets") or []:
                t = str(token or "").strip()
                if t and t not in memory_targets:
                    memory_targets.append(t)
        elif role == "user":
            a = str(item.get("content") or "").strip()
            if a:
                qa.append(f"A: {a}")
                qa.extend(f"Chosen: {text}" for text in item.get("selected_options", []))

    preferred_language = "en"
    source_language = "en"
    translation_mode = "english"

    header = "[NovelClaw generation brief]"
    language_profile = (
        ("[language profile]\n"
        f"preferred_output_language: {preferred_language}\n"
        f"source_language: {source_language}\n"
        f"translation_mode: {translation_mode}")
    )
    style_block = ""
    if style_targets:
        style_block = (
            "\n\n[style targets]\n"
        ) + "\n".join(f"- {item}" for item in style_targets[:10])
    memory_block = ""
    if memory_targets:
        memory_block = (
            "\n\n[memory targets]\n"
        ) + "\n".join(f"- {item}" for item in memory_targets[:10])
    execution_header = "[execution preference]"
    execution_block = (
        f"\n\n{execution_header}\n"
        f"generation_scope: {generation_preferences['generation_scope']}\n"
        f"requested_chapters: {generation_preferences['requested_chapters'] or 'auto'}\n"
        f"chapter_pause_mode: {generation_preferences['chapter_pause_mode']}\n"
        f"user_request: {generation_preferences['user_request'] or '-'}"
    )
    if qa:
        qa_text = "\n".join(qa[-16:])
        qa_header = "[recent ideation QA]"
        return f"{header}\n{refined}\n\n{language_profile}{style_block}{memory_block}{execution_block}\n\n{qa_header}\n{qa_text}".strip()
    return f"{header}\n{refined}\n\n{language_profile}{style_block}{memory_block}{execution_block}".strip()
