from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set


@dataclass(frozen=True)
class CapabilitySpec:
    slug: str
    name_en: str
    category_en: str
    description_en: str
    manager_action: Optional[str] = None
    default_enabled: bool = True
    always_enabled: bool = False
    order: int = 0


CAPABILITY_REGISTRY: List[CapabilitySpec] = [
    CapabilitySpec(
        slug="plot_strategy",
        name_en="Plot Strategy",
        category_en="Claw Actions",
        description_en="Plan chapter progression, conflict escalation, and turning beats before drafting.",
        manager_action="plot_strategy",
        order=10,
    ),
    CapabilitySpec(
        slug="retrieve_context",
        name_en="Context Retrieval",
        category_en="Claw Actions",
        description_en="Pull relevant dynamic-memory context, facts, and recent writing state back into the loop.",
        manager_action="retrieve_context",
        order=20,
    ),
    CapabilitySpec(
        slug="enrich_character",
        name_en="Character Enrichment",
        category_en="Claw Actions",
        description_en="Strengthen motives, relationships, behavioral boundaries, and character consistency.",
        manager_action="enrich_character",
        order=30,
    ),
    CapabilitySpec(
        slug="enrich_world",
        name_en="World Enrichment",
        category_en="Claw Actions",
        description_en="Clarify world rules, scene constraints, setting facts, and reusable canon.",
        manager_action="enrich_world",
        order=40,
    ),
    CapabilitySpec(
        slug="inspect_workspace",
        name_en="Inspect Workspace",
        category_en="Local Tools",
        description_en="Inspect the local run workspace, chapter files, and current memory assets before acting.",
        manager_action="inspect_workspace",
        order=45,
    ),
    CapabilitySpec(
        slug="draft_chapter",
        name_en="Draft Chapter",
        category_en="Core Writing",
        description_en="Generate the main prose draft for the current chapter.",
        manager_action="draft_chapter",
        always_enabled=True,
        order=50,
    ),
    CapabilitySpec(
        slug="rewrite_chapter",
        name_en="Rewrite Chapter",
        category_en="Core Writing",
        description_en="Rewrite or compress drafts when quality, pacing, or length misses the target.",
        manager_action="rewrite_chapter",
        always_enabled=True,
        order=60,
    ),
    CapabilitySpec(
        slug="finalize",
        name_en="Finalize",
        category_en="Core Writing",
        description_en="Finalize the chapter once the current candidate is acceptable.",
        manager_action="finalize",
        always_enabled=True,
        order=70,
    ),
    CapabilitySpec(
        slug="sync_storyboard",
        name_en="Sync Storyboard",
        category_en="Local Tools",
        description_en="Write the current chapter brief and Claw plan into local workspace files and outline assets.",
        manager_action="sync_storyboard",
        order=80,
    ),
    CapabilitySpec(
        slug="sync_characters",
        name_en="Sync Characters",
        category_en="Local Tools",
        description_en="Extract current character state and write it into local files plus character memory.",
        manager_action="sync_characters",
        order=90,
    ),
    CapabilitySpec(
        slug="sync_world",
        name_en="Sync World",
        category_en="Local Tools",
        description_en="Extract world rules and continuity facts into local workspace files and world memory.",
        manager_action="sync_world",
        order=100,
    ),
    CapabilitySpec(
        slug="idea_analyzer",
        name_en="Idea Analyzer",
        category_en="Support Skills",
        description_en="Turn a rough idea into a stable brief with genre, cast, goals, and constraints.",
        order=110,
    ),
    CapabilitySpec(
        slug="analyzer",
        name_en="Task Analyzer",
        category_en="Support Skills",
        description_en="Understand whether the current step needs planning, drafting, revision, or cleanup.",
        order=120,
    ),
    CapabilitySpec(
        slug="dynamic_memory",
        name_en="Dynamic Memory",
        category_en="Memory System",
        description_en="Persist reusable story state, chapter briefs, facts, and working memory across iterations.",
        always_enabled=True,
        order=130,
    ),
    CapabilitySpec(
        slug="turning_point_tracker",
        name_en="Turning Point Tracker",
        category_en="Support Skills",
        description_en="Track whether chapters are meaningfully advancing the story.",
        order=140,
    ),
    CapabilitySpec(
        slug="consistency_checker",
        name_en="Consistency Checker",
        category_en="Support Skills",
        description_en="Check continuity across character, world, timeline, and established facts.",
        order=150,
    ),
    CapabilitySpec(
        slug="realtime_editor",
        name_en="Realtime Editor",
        category_en="Support Skills",
        description_en="Find weak spans in a draft and apply targeted revision passes.",
        order=160,
    ),
    CapabilitySpec(
        slug="evaluator",
        name_en="Evaluator",
        category_en="Support Skills",
        description_en="Score coherence, pacing, emotional impact, and task fit of candidate drafts.",
        order=170,
    ),
    CapabilitySpec(
        slug="judge",
        name_en="Judge",
        category_en="Support Skills",
        description_en="Break ties between close chapter candidates when rewards are too similar.",
        order=180,
    ),
]


def capability_map() -> Dict[str, CapabilitySpec]:
    return {item.slug: item for item in CAPABILITY_REGISTRY}


def default_enabled_capability_slugs() -> Set[str]:
    return {item.slug for item in CAPABILITY_REGISTRY if item.default_enabled or item.always_enabled}


def normalize_capability_slugs(values: Iterable[str]) -> Set[str]:
    known = capability_map()
    out: Set[str] = set()
    for value in values:
        slug = str(value or "").strip().lower()
        if slug in known:
            out.add(slug)
    for item in CAPABILITY_REGISTRY:
        if item.always_enabled:
            out.add(item.slug)
    return out


def enabled_capability_slugs_from_env(raw: str) -> Set[str]:
    text = str(raw or "").strip()
    if not text:
        return default_enabled_capability_slugs()
    return normalize_capability_slugs(text.split(","))


def claw_action_specs() -> List[CapabilitySpec]:
    return [item for item in CAPABILITY_REGISTRY if item.manager_action]


def enabled_claw_actions(enabled_slugs: Iterable[str]) -> Set[str]:
    enabled = normalize_capability_slugs(enabled_slugs)
    actions: Set[str] = set()
    for item in claw_action_specs():
        if item.always_enabled or item.slug in enabled:
            actions.add(str(item.manager_action))
    return actions
