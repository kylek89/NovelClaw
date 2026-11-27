"""
Task Execution Module (Compositive Executor)
"""
from typing import Any, Dict, List, Optional, Tuple
import math
from pathlib import Path
from utils.file_io import atomic_write_text, save_text_version
from utils.text_metrics import text_length, truncate_text, summarize_consistency
from utils.artifacts import latest_chapter_files
from config import Config
import json
from utils.llm_client import LLMClient
from rag.retriever import Retriever
from rag.memory_system import MemorySystem
from rag.consistency_checker import ConsistencyChecker
from rag.turning_point_tracker import TurningPointTracker
from rag.realtime_editor import RealtimeEditor
from workflow.analyzer import Analyzer
from workflow.organizer import AdaptiveOrganizer
from workflow.claw_manager import OpenClawManager
from utils.idea_analyzer import IdeaAnalyzer
from agents.plot_agent import PlotAgent
from agents.character_agent import CharacterAgent
from agents.world_agent import WorldAgent
from agents.retrieval_agent import RetrievalAgent
from agents.writer_agent import WriterAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.judge_agent import JudgeAgent
from workflow.reward_system import RewardSystem
import re


class CompositiveExecutor:
    """Task Execution Module: Coordinate agents to complete text generation"""

    def __init__(self, config: Config):
        self.config = config
        self.llm_client = LLMClient(config)
        self.lang = "en"
        self.enabled_capabilities = set(getattr(config, "enabled_capabilities", set()) or set())
        # Lock the book title; default to 'topic', but override if a title appears in the outline
        self.book_title: Optional[str] = None

        # Initialize the Retriever only when RAG is explicitly enabled; otherwise, disable it completely.
        self.retriever = Retriever(config, collection_name="knowledge_base") if bool(getattr(config, "enable_rag", False)) else None

        # Retain only dynamic memory; disable the static knowledge base
        self.static_kb = None

        # Initialize the memory system (dynamic memory RAG)
        self.memory_system = MemorySystem(config)

        # Initialize the plot twist tracker
        self.turning_point_tracker = TurningPointTracker(
            config, self.llm_client, self.memory_system
        )

        # Initialize the consistency checker
        self.consistency_checker = ConsistencyChecker(
            config, self.llm_client, self.memory_system
        )

        # Initialize the real-time modification system
        self.realtime_editor = RealtimeEditor(
            config, self.llm_client, self.memory_system, self.turning_point_tracker
        )

        # Initialize the writing analysis component.
        self.analyzer = Analyzer(config, self.llm_client)
        self.idea_analyzer = IdeaAnalyzer(config, self.llm_client)
        self.organizer = AdaptiveOrganizer(config, self.llm_client)
        self.chapter_counter = 0  # Chapter counter for long-form chaptering
        self.main_characters: List[str] = []  # Locked protagonist name anchor
        self._refresh_agents_for_language()
        self.reward_system = RewardSystem(config)

    def _claw_mode(self) -> bool:
        return str(getattr(self.config, "execution_mode", "claw") or "claw").lower() == "claw"

    def _build_claw_bootstrap_plan(
        self,
        *,
        topic: str,
        text_type: str,
        target_length: int,
        task_analysis: Dict,
        genre: Optional[str],
        style_tags: Optional[List[str]],
    ) -> Dict:
        estimated_iterations = self._estimate_chapter_count(target_length or 20000)
        per_chapter_target = max(800, int((target_length or 4000) / max(estimated_iterations, 1)))
        strategy = "claw_autonomous_loop"
        self._append_progress_event(
            "claw_bootstrap",
            (
                f"topic={self._safe_excerpt(topic, 80)}, text_type={text_type}, "
                f"genre={genre or '-'}, steps={int(getattr(self.config, 'claw_max_steps', 6) or 6)}, "
                f"chapters={estimated_iterations}, per_chapter_target={per_chapter_target}, "
                f"style_tags={','.join(style_tags or []) or '-'}, "
                f"complexity={task_analysis.get('complexity', 'unknown')}"
            ),
        )
        return {
            "strategy": strategy,
            "estimated_iterations": estimated_iterations,
            "agent_sequence": ["writer"],
            "chapter_plan": {"per_chapter_target": per_chapter_target},
            "use_rag": False,
            "planner": "claw",
        }

    def _normalize_target_length(self, target_length: Any, default: int = 5000) -> int:
        """Normalize target-length input into a safe integer.

        Rule:
        - If an explicit numeric value can be parsed, use it.
        - Otherwise, fall back directly to the default length.
        """
        if isinstance(target_length, (int, float)):
            return max(1, int(target_length))

        raw = str(target_length or "").strip()
        if not raw:
            return default

        try:
            return max(1, int(raw))
        except Exception:
            pass

        digit_match = re.search(r"(\d{3,7})", raw)
        if digit_match:
            try:
                return max(1, int(digit_match.group(1)))
            except Exception:
                pass

        return default


    def _log(self, tag: str, message: str) -> None:
        print(f"[{tag}] {message}")

    def _capability_enabled(self, slug: str) -> bool:
        return slug in self.enabled_capabilities

    def _refresh_agents_for_language(self) -> None:
        self.agents = {
            "plot": PlotAgent(self.config, self.llm_client, self.retriever, self.memory_system, self.static_kb),
            "character": CharacterAgent(self.config, self.llm_client, self.retriever, self.memory_system, self.static_kb),
            "world": WorldAgent(self.config, self.llm_client, self.retriever, self.memory_system, self.static_kb),
            "retrieval": RetrievalAgent(self.config, self.llm_client, self.retriever, self.memory_system, self.static_kb),
            "writer": WriterAgent(self.config, self.llm_client, self.retriever, self.memory_system, self.static_kb),
        }
        self.evaluator_agent = EvaluatorAgent(
            self.config,
            self.llm_client,
            self.retriever,
            self.memory_system,
            self.static_kb,
        ) if self._capability_enabled("evaluator") else None
        self.judge_agent = JudgeAgent(self.config, self.llm_client) if self._capability_enabled("judge") else None

    def _remember_project_claw_context(
        self,
        *,
        idea: str,
        topic: str,
        text_type: str,
        target_length: int,
        genre: Optional[str],
        style_tags: Optional[List[str]],
    ) -> None:
        source_language = "en"
        self.memory_system.store_claw_memory(
            "language_profile",
            (
                f"preferred_language={self.lang}\n"
                f"source_language={source_language}\n"
                "translation_mode=english"
            ),
            topic,
            metadata={"stage": "project_seed"},
            store_vector=False,
        )
        self.memory_system.store_claw_memory(
            "story_premise",
            idea,
            topic,
            metadata={"stage": "project_seed"},
            store_vector=True,
        )
        self.memory_system.store_claw_memory(
            "task_briefs",
            (
                f"topic={topic}\ntext_type={text_type}\n"
                f"target_length={target_length}\n"
                f"genre={genre or '-'}"
            ),
            topic,
            metadata={"stage": "project_seed"},
            store_vector=True,
        )
        if genre or style_tags:
            self.memory_system.store_claw_memory(
                "style_guide",
                (
                    f"genre={genre or '-'}\n"
                    f"style_tags={', '.join(style_tags or []) or '-'}\n"
                    f"language={self.lang}"
                ),
                topic,
                metadata={"stage": "project_seed"},
                store_vector=True,
            )

    def _select_seed_agent_sequence(self, execution_plan: Dict, chapter_no: int) -> List[str]:
        base_seq = [
            agent_name
            for agent_name in (execution_plan.get("agent_sequence") or [])
            if agent_name in self.agents
        ]
        if not base_seq:
            return ["writer"]

        if str(getattr(self.config, "execution_mode", "claw") or "claw").lower() != "claw":
            return base_seq

        interval = int(getattr(self.config, "full_cycle_interval", 0) or 0)
        if interval <= 0:
            return base_seq
        if chapter_no <= 1 or (chapter_no - 1) % interval == 0:
            return base_seq
        return ["writer"]

    def _generate_unfixed_story(
        self,
        idea: str,
        topic: Optional[str],
        target_length: Optional[int],
        genre: Optional[str],
        style_tags: Optional[List[str]],
        initial_knowledge: Optional[str],
        outline_free: bool = False,
    ) -> Dict:
        """
        Full English mode: default to multi-turn segmented generation, hide titles but preserve hierarchy, and use memory.
        """
        self._log("Workflow", "Unfixed English mode: generating long-form story in segments without explicit headings.")
        topic = topic or idea[:120]
        genre = genre or "general"
        style_tags = style_tags or ["immersive", "tight-pacing"]
        # optional initial knowledge into RAG
        if initial_knowledge and self.retriever is not None:
            self.retriever.add_knowledge(initial_knowledge, {"type": "initial"})
        outline = ""
        if not outline_free:
            outline = self._generate_global_outline(
                idea=idea,
                topic=topic,
                genre=genre,
                style_tags=style_tags,
                target_length=target_length or 5000
            )
        # Segment control
        seg_words = getattr(self.config, "en_segment_words", 1800)
        seg_count = getattr(self.config, "en_segments", 4)
        if target_length:
            seg_count = max(2, min(10, round((target_length or 6000) / max(seg_words, 800))))
        generated_segments = []
        round_results = []

        for idx in range(seg_count):
            prompt = f"""Premise: {idea}
Genre: {genre}
Style tags: {', '.join(style_tags)}
Global outline (if any):
{outline if outline else '[outline skipped]'}

You are writing segment {idx+1}/{seg_count} of a long English story. Continue the narrative with clear structure (setup/rising/climax/resolution across segments), grounded characters, coherent worldbuilding. Avoid headings/markdown/bullets. Aim ~{seg_words} words. Maintain continuity with prior segments via memory; do not restart the story."""
            context_text = None
            if generated_segments:
                context_text = "\n".join(generated_segments[-2:])
            res = self.agents["writer"].generate(
                prompt,
                context=context_text,
                topic=topic,
                genre=genre,
                style_tags=style_tags,
                target_length=seg_words
            )
            content = res.get("content", "")
            round_results.append(res)
            generated_segments.append(content)
            # store into memory for later retrieval
            try:
                self.memory_system.store_generated_text(content, topic, {"source": "unfixed_segment", "segment": idx+1})
            except Exception:
                pass

        final_text = "\n\n".join(generated_segments)
        return {
            "topic": topic,
            "text_type": "creative",
            "language": self.lang,
            "length": len(final_text),
            "iterations": seg_count,
            "chapters_written": seg_count,
            "best_reward": 1.0,
            "final_text": final_text,
            "round_results": round_results,
        }

    def generate_long_text(
        self,
        idea: str,
        topic: Optional[str] = None,
        text_type: Optional[str] = None,
        target_length: Optional[int] = None,
        initial_knowledge: Optional[str] = None,
        genre: Optional[str] = None,
        style_tags: Optional[List[str]] = None,
        auto_analyze: bool = True
    ) -> Dict:
        """
        Generate long text

        Args:
            idea: User's creative idea (required)
            topic: Theme (optional; extracted from 'idea' if not provided)
            text_type: Text type (optional; auto-inferred if not provided)
            target_length: Target length (optional; auto-inferred if not provided)
            initial_knowledge: Initial knowledge base content
            genre: Type (optional; auto-inferred if not provided)
            style_tags: Style tags (optional; auto-inferred if not provided)
            auto_analyze: Whether to automatically analyze the idea (default: True)

        Returns:
            Generated output
        """

        # English unfixed workflow: reserved only for non-Claw-compatible paths; Claw mode uniformly uses the agent loop.
        if (not self._claw_mode()) and getattr(self.config, "workflow_mode", "chaptered") == "unfixed":
            return self._generate_unfixed_story(
                idea=idea,
                topic=topic,
                target_length=target_length,
                genre=genre,
                style_tags=style_tags,
                initial_knowledge=initial_knowledge,
                outline_free=getattr(self.config, "outline_free", False)
            )

        # 0. Automatically analyze the idea (if enabled)
        if auto_analyze:
            self._log("IdeaAnalyzer", "Analyzing the story idea...")
            idea_analysis = self.idea_analyzer.analyze_idea(idea, language=self.lang)

            # Use the analysis results to fill in missing parameters
            if not topic:
                topic = self.idea_analyzer.extract_topic_from_idea(idea)
            if not text_type:
                text_type = idea_analysis.get("text_type", "creative")
            if not target_length:
                target_length = idea_analysis.get("suggested_length", 5000)
            if not genre:
                genre = idea_analysis.get("genre", "general")
            if not style_tags:
                style_tags = idea_analysis.get("style_tags", ["nuanced"])

            self._log("IdeaAnalyzer", "Inferred result:")
            print(f"  - Genre: {genre}")
            print(f"  - Style: {', '.join(style_tags)}")
            print(f"  - Topic: {topic}")
            target_length = self._normalize_target_length(target_length, default=5000)
            print(f"  - Target length: {target_length} words")
        else:
            # Use default values if automatic analysis is disabled
            if not topic:
                topic = idea[:100]  # Use the first 100 characters of the idea as the topic
            if not text_type:
                text_type = "creative"
            if not target_length:
                target_length = 5000
            if not genre:
                genre = "general"
            if not style_tags:
                style_tags = ["delicate"]

        target_length = self._normalize_target_length(target_length, default=5000)

        self._remember_project_claw_context(
            idea=idea,
            topic=topic,
            text_type=text_type,
            target_length=target_length,
            genre=genre,
            style_tags=style_tags,
        )

        # 1. Add initial knowledge (if available)
        if initial_knowledge and self.retriever is not None:
            self.retriever.add_knowledge(initial_knowledge, {"type": "initial"})

        # 2. Task analysis / Claw bootstrap.
        if self._claw_mode():
            complexity = "high" if target_length >= 20000 else ("medium" if target_length >= 8000 else "focused")
            task_analysis = {
                "complexity": complexity,
                "target_length": target_length,
                "controller": "claw",
                "text_type": text_type,
            }
            self._log("Claw", f"Claw is taking over the task and building the initial working context: {topic}")
        else:
            self._log("Analyzer", f"Analyzing task: {topic}")
            task_analysis = self.analyzer.analyze_task(topic, text_type, target_length)
            task_analysis["target_length"] = target_length
            self._log("Analyzer", f"Task complexity: {task_analysis.get('complexity', 'unknown')}")

        # 3. Create execution plan / Claw startup info.
        if self._claw_mode():
            execution_plan = self._build_claw_bootstrap_plan(
                topic=topic,
                text_type=text_type,
                target_length=target_length,
                task_analysis=task_analysis,
                genre=genre,
                style_tags=style_tags,
            )
            self._log("Claw", "Switched to native Claw agent mode: planning, action selection, and execution are all owned by Claw.")
        else:
            self._log("Organizer", "Building execution plan...")
            execution_plan = self.organizer.create_execution_plan(
                task_analysis,
                list(self.agents.keys())
            )
            if getattr(self.config, "fast_mode", False):
                # Fast mode: Writing only, single turn, avoiding repeated calls to plot/character/world
                execution_plan["estimated_iterations"] = 1
                execution_plan["agent_sequence"] = ["writer"]
                execution_plan["strategy"] = "fast_write_only"
            self._log("Organizer", f"Execution strategy: {execution_plan['strategy']}")
        chapter_plan = execution_plan.get("chapter_plan", {})
        # Pre-designed total chapter count (prioritize the number of chapters from the outline)
        planned_total = 0

        # 3.5 Generate global outline / chapter outline.
        if self._claw_mode():
            try:
                planned_total = self._ensure_strategic_outline_assets(
                    idea=idea,
                    topic=topic,
                    genre=genre,
                    style_tags=style_tags,
                    target_length=target_length,
                )
            except Exception as e:
                planned_total = 0
                self._log("Outline", f"Claw strategic outline bootstrap failed: {e}")
            if planned_total <= 0:
                planned_total = int(execution_plan.get("estimated_iterations") or 0)
            self._append_progress_event(
                "claw_outline_policy",
                f"strategic_outline_bootstrap={'ready' if planned_total > 0 else 'fallback'}, support_materials=on_demand, controller=OpenClaw",
            )
        else:
            try:
                global_outline = self._generate_global_outline(
                    idea=idea,
                    topic=topic,
                    genre=genre,
                    style_tags=style_tags,
                    target_length=target_length
                )
                if global_outline:
                    # Attempt to extract the book title from the outline; otherwise, use the topic as the title
                    try:
                        self.book_title = self._extract_book_title(global_outline) or topic
                    except Exception:
                        self.book_title = topic
                    self.memory_system.store_outline(
                        global_outline,
                        topic,
                        structure={"kind": "global_outline"}
                    )
                    self._append_progress_event(
                        "global_outline",
                        self._safe_excerpt(global_outline, 320)
                    )
                    self._log("Outline", "Global outline generated and stored in dynamic memory.")
                    # Lock protagonist name anchor: Extract from the global outline and existing character cards
                    try:
                        self.main_characters = self._extract_main_characters(global_outline)
                        mem_chars = [c.get("name") for c in self.memory_system.get_characters_by_topic(topic) if c.get("name")]
                        for name in mem_chars:
                            if name not in self.main_characters and self._is_valid_character_name(name):
                                self.main_characters.append(name)
                    except Exception:
                        pass
                    try:
                        stored = self._precompute_chapter_outlines(global_outline, topic, genre=genre, style_tags=style_tags)
                        if not stored:
                            self._log("Outline", "Warning: chapter outlines are empty; retrying outline generation once.")
                            stored = self._precompute_chapter_outlines(global_outline, topic, genre=genre, style_tags=style_tags)
                        if not stored:
                            raise RuntimeError("Chapter outline generation failed. Writing stopped before drafting without an outline.")
                        planned_total = len(stored)
                        self._append_progress_event(
                            "chapter_outline_ready",
                            f"count={planned_total}"
                        )
                        self._log("Outline", f"Generated and stored {len(stored)} chapter outlines from the global outline.")
                        # Pre-generate core characters/worldview/retrieval notes (based on the global outline) for subsequent chapter references
                        outline_cards = self._run_outline_stage_agents(global_outline, topic, genre, style_tags)
                        if outline_cards:
                            self._log("Outline", f"Generated {len(outline_cards)} character/world/retrieval setup notes from the outline.")
                            for oc in outline_cards[:3]:
                                ctext = (oc.get("content") or "")[:120].replace("\n", " ")
                                print(f"  - {oc.get('type','')} summary: {ctext}")
                    except Exception as e:
                        self._log("Outline", f"Failed to split chapter outlines: {e}")
            except Exception as e:
                self._log("Outline", f"Failed to generate the global outline: {e}")

        # 4. Execute generation process
        generated_content = []
        iteration = 0
        best_result = None
        chapter_checks = []
        best_reward = 0.0
        # Determine total chapter count: Prioritize the chapter outline count; otherwise, use estimation
        if planned_total <= 0:
            planned_total = self._estimate_chapter_count(target_length or 20000)
        max_iterations = min(planned_total, getattr(self.config, "max_total_iterations", planned_total))

        while iteration < max_iterations:
            self.memory_system.refresh()
            iteration += 1
            self.chapter_counter += 1
            self._log("Executor", f"Starting iteration {iteration}...")

            # Chapter planning: automatically generated only by the fixed workflow; Claw mode leaves it to the manager to decide whether to supplement strategies.
            chapter_plan_text = ""
            if not self._claw_mode():
                try:
                    chapter_plan_text = self._plan_current_chapter_by_agent(
                        topic, self.chapter_counter, genre, style_tags
                    )
                    if chapter_plan_text:
                        self.memory_system.store_outline(
                            chapter_plan_text,
                            topic,
                            structure={"kind": "chapter_plan", "chapter": self.chapter_counter}
                        )
                        self._append_progress_event(
                            "chapter_plan",
                            self._safe_excerpt(chapter_plan_text, 320),
                            self.chapter_counter,
                        )
                except Exception:
                    pass
            chapter_outline_text = self._get_chapter_outline_text(topic, self.chapter_counter)
            outline_title = self._get_chapter_outline_title(topic, self.chapter_counter)
            if chapter_outline_text:
                outline_detail = chapter_outline_text if not outline_title else f"{outline_title} | {chapter_outline_text}"
                self._append_progress_event(
                    "chapter_outline",
                    self._safe_excerpt(outline_detail, 320),
                    self.chapter_counter,
                )

            try:
                self.memory_system.store_chapter_claw_state(
                    topic,
                    self.chapter_counter,
                    title=outline_title or "",
                    outline_text=chapter_outline_text or "",
                    plan_text=chapter_plan_text if 'chapter_plan_text' in locals() else "",
                )
            except Exception:
                pass

            # Dynamic memory only: Disable static knowledge base
            use_static_kb = False

            # Chapter-level target length (adaptive to remaining word count)
            # If a total chapter count is planned, distribute evenly across all target chapters.
            if target_length and planned_total:
                chapter_target = max(
                    800,
                    int(target_length / planned_total)
                )
            else:
                chapter_target = max(
                    800,
                    chapter_plan.get("per_chapter_target", target_length or 4000)
                )
            chapter_min_required, chapter_max_allowed, length_source = self._get_chapter_length_bounds(
                topic=topic,
                chapter_no=self.chapter_counter,
                chapter_target=chapter_target,
            )
            self._append_progress_event(
                "chapter_length_plan",
                f"target={chapter_target}, min={chapter_min_required}, max={chapter_max_allowed}, source={length_source}",
                self.chapter_counter
            )

            # NovelClaw native chapter agent loop: no longer injects fixed workflow pre-execution results for Claw.
            round_results = []
            self._append_progress_event(
                "claw_native_loop",
                "seed_workflow=disabled, controller=OpenClaw, start_from=manager_loop",
                self.chapter_counter,
            )
            chapter_state = self._run_chapter_agentic_loop(
                topic=topic,
                round_results=round_results,
                generated_content=generated_content,
                genre=genre,
                style_tags=style_tags,
                chapter_target=chapter_target,
                chapter_min_required=chapter_min_required,
                chapter_max_allowed=chapter_max_allowed,
            )
            story_text = chapter_state["story_text"]
            story_len = self._measure_text(story_text)
            round_results_for_eval = chapter_state["round_results"]
            evaluation_result = chapter_state.get("evaluation", {})
            consistency_result = chapter_state.get("consistency", {})
            best_reward = max(best_reward, float(chapter_state.get("reward_score", 0.0) or 0.0))

            self._log("Memory", "Storing generated content into memory...")
            for result in round_results_for_eval:
                content = result.get("content", "")
                result_type = result.get("type", "")
                if not content:
                    continue
                if result_type == "character":
                    character_name = self._extract_character_name(content)
                    if character_name:
                        self.memory_system.store_character(character_name, content, topic)
                        self._append_progress_event(
                            "character_setting",
                            f"{character_name} | {self._safe_excerpt(content, 260)}",
                            self.chapter_counter,
                        )
                elif result_type == "world":
                    setting_name = f"{topic}_world_setting"
                    self.memory_system.store_world_setting(setting_name, content, topic)
                    self._append_progress_event(
                        "world_setting",
                        self._safe_excerpt(content, 260),
                        self.chapter_counter,
                    )
                elif result_type == "plot":
                    self.memory_system.store_plot_point(content, topic)
                elif result_type in {"writer", "modified"}:
                    self.memory_system.store_generated_text(
                        content,
                        topic,
                        {"source": "writer_round", "chapter": self.chapter_counter},
                    )

            self._log("TurningPoint", "Detecting turning points...")
            turning_points = []
            if story_text and len(story_text.strip()) > 20:
                turning_points = self.turning_point_tracker.detect_turning_points(
                    story_text, topic, self._build_context(generated_content)
                )
            else:
                self._log("TurningPoint", "Skipped: the current chapter text is empty or too short.")

            if turning_points:
                for tp in turning_points:
                    self.turning_point_tracker.record_turning_point(
                        tp.get("type", "key_events"),
                        tp.get("content", {}),
                        topic,
                    )
            else:
                self.turning_point_tracker.record_turning_point(
                    "turning_point_issue",
                    {"chapter": self.chapter_counter, "note": "No clear turning point was detected in this chapter. Stay aligned with the outline and avoid drifting beyond scope."},
                    topic,
                )

            try:
                issues_text = "\n".join(consistency_result.get("all_issues", [])) or "No major consistency issues"
                self.memory_system.store_outline(
                    f"Chapter {self.chapter_counter} continuity check:\n{issues_text}",
                    topic,
                    structure={"kind": "consistency_issue", "chapter": self.chapter_counter},
                )
            except Exception:
                pass
            # Write the chapter's final text (modified version) to memory.
            chapter_checks.append(consistency_result)
            story_text = self._sanitize_chapter_text(story_text, self.chapter_counter)
            try:
                self.memory_system.store_generated_text(
                    story_text,
                    topic,
                    {"source": "chapter_final", "chapter": self.chapter_counter},
                    store_vector=True
                )
            except Exception:
                pass
            # Write the chapter's final text to a file for traceability.
            self._write_chapter_file(self.chapter_counter, story_text, iteration, topic)

            # Chapter completion signal + inter-chapter user interaction (web mode file IPC, CLI mode hook).
            try:
                self._signal_chapter_complete(self.chapter_counter, story_len, planned_total)
            except Exception:
                pass
            try:
                self._wait_for_between_chapter_interaction(self.chapter_counter, planned_total)
            except Exception:
                pass
            try:
                self._handle_between_chapter_interaction(topic)
            except Exception:
                pass

            # Record progress (word count, chapter).
            try:
                self._write_progress_log(
                    self.chapter_counter,
                    story_len,
                    planned_total,
                    topic,
                    chapter_target=chapter_target,
                    chapter_min_required=chapter_min_required,
                    chapter_max_allowed=chapter_max_allowed,
                )
            except Exception:
                pass

            # Chapter-level summary for subsequent retrieval and consistency.
            summary_text = ""
            rolling = ""
            cards: List[str] = []
            try:
                summary_text = self._summarize_chapter(story_text, topic, self.chapter_counter)
                if summary_text:
                    self.memory_system.store_outline(
                        f"Chapter {self.chapter_counter} summary:\n{summary_text}",
                        topic,
                        structure={"kind": "chapter_summary", "chapter": self.chapter_counter}
                    )

                    # Update the scroll summary (retain only globally essential information to prevent context explosion)
                    rolling = self._update_rolling_summary(topic, summary_text)
                    if rolling:
                        self.memory_system.store_outline(
                            rolling,
                            topic,
                            structure={"kind": "rolling_summary"}
                        )

                    # Extract key fact cards (characters/events/settings/foreshadowing) to help prevent hallucinations
                    cards = self._extract_fact_cards(story_text, topic, self.chapter_counter)
                    for card in cards[:6]:
                        self.memory_system.store_fact_card(
                            card,
                            topic,
                            card_type="chapter_facts",
                            metadata={"chapter": self.chapter_counter}
                        )
            except Exception as e:
                self._log("Summary", f"Failed to summarize the chapter: {e}")

            try:
                self.memory_system.store_chapter_claw_state(
                    topic,
                    self.chapter_counter,
                    title=outline_title or "",
                    outline_text=chapter_outline_text or "",
                    plan_text=chapter_plan_text,
                    summary_text=summary_text,
                    rolling_summary=rolling,
                    fact_cards=cards,
                    story_text=story_text,
                    evaluation_suggestions=(evaluation_result.get("suggestions", []) if isinstance(evaluation_result, dict) else []),
                    consistency_issues=(consistency_result.get("all_issues", []) if isinstance(consistency_result, dict) else []),
                    reward_score=chapter_state.get("reward_score", 0.0),
                    issues_count=chapter_state.get("issues_count", 0),
                )
            except Exception:
                pass

            # Cover every chapter with the current best result (do not rely on scoring)
            best_result = {
                "content": story_text,  # Base content on the novel's main text
                "round_results": round_results_for_eval,
                "evaluation": evaluation_result,
                "iteration": iteration
            }

            # Fixed execution plan: do not use scoring/feedback for dynamic scheduling

            # Update generated content
            generated_content += round_results

            # Stop if the total planned chapter count is reached
            if self.chapter_counter >= planned_total:
                self._log("Executor", f"Reached the planned chapter count ({planned_total}); stopping generation.")
                break

        # 5. Assemble the final text (prioritize stitching by final chapter drafts, then by best results)
        stitched_text = self._stitch_chapters(topic)
        if not stitched_text and best_result:
            stitched_text = best_result["content"]
        if not stitched_text:
            stitched_text = self._combine_texts(generated_content)

        final_text = stitched_text
        # Store the final text
        if final_text:
            self.memory_system.store_generated_text(final_text, topic)
            outline = self._extract_outline(final_text)
            if outline:
                self.memory_system.store_outline(outline, topic)

        # Final consistency check
        final_consistency = summarize_consistency(chapter_checks)

        return {
            "idea": idea,
            "topic": topic,
            "text_type": text_type,
            "language": self.lang,
            "genre": genre,
            "style_tags": style_tags,
            "final_text": final_text,
            "length": len(final_text),
            "iterations": iteration,
            "best_reward": best_reward,
            "task_analysis": task_analysis,
            "execution_plan": execution_plan,
            "evaluation": best_result.get("evaluation", {}) if best_result else {},
            "consistency": final_consistency,
            "round_results": best_result.get("round_results", []) if best_result else generated_content,
            "chapters_written": self.chapter_counter,
            "progress_log": self._summarize_progress()
        }

    def _run_chapter_agentic_loop(
        self,
        *,
        topic: str,
        round_results: List[Dict],
        generated_content: List[Dict],
        genre: Optional[str],
        style_tags: Optional[List[str]],
        chapter_target: int,
        chapter_min_required: int,
        chapter_max_allowed: int,
    ) -> Dict:
        if str(getattr(self.config, "execution_mode", "claw") or "claw").lower() == "claw":
            hook = getattr(self, "_claw_user_interaction_hook", None)
            manager = OpenClawManager(self, user_interaction_hook=hook)
            return manager.run_chapter_loop(
                topic=topic,
                round_results=round_results,
                generated_content=generated_content,
                genre=genre,
                style_tags=style_tags,
                chapter_target=chapter_target,
                chapter_min_required=chapter_min_required,
                chapter_max_allowed=chapter_max_allowed,
            )

        current = self._score_chapter_candidate(
            topic=topic,
            round_results=round_results,
            genre=genre,
            style_tags=style_tags,
            chapter_target=chapter_target,
            chapter_min_required=chapter_min_required,
            chapter_max_allowed=chapter_max_allowed,
        )
        best = current
        max_subrounds = max(1, int(getattr(self.config, "max_chapter_subrounds", 2) or 2))

        for subround in range(2, max_subrounds + 1):
            if not self._should_retry_chapter(best, chapter_min_required, chapter_max_allowed):
                break

            self._append_progress_event(
                "agentic_retry",
                (
                    f"subround={subround}, reward={best.get('reward_score', 0):.3f}, "
                    f"issues={best.get('issues_count', 0)}, length={best.get('story_length', 0)}"
                ),
                self.chapter_counter,
            )
            revised_results = self._rewrite_chapter_with_feedback(
                topic=topic,
                story_text=best.get("story_text", ""),
                generated_content=generated_content,
                evaluation_result=best.get("evaluation", {}),
                consistency_result=best.get("consistency", {}),
                genre=genre,
                style_tags=style_tags,
                chapter_target=chapter_target,
                chapter_min_required=chapter_min_required,
                chapter_max_allowed=chapter_max_allowed,
                subround=subround,
            )
            challenger = self._score_chapter_candidate(
                topic=topic,
                round_results=revised_results,
                genre=genre,
                style_tags=style_tags,
                chapter_target=chapter_target,
                chapter_min_required=chapter_min_required,
                chapter_max_allowed=chapter_max_allowed,
            )
            best = self._select_better_candidate(best, challenger)

        return best

    def _score_chapter_candidate(
        self,
        *,
        topic: str,
        round_results: List[Dict],
        genre: Optional[str],
        style_tags: Optional[List[str]],
        chapter_target: int,
        chapter_min_required: int,
        chapter_max_allowed: int,
    ) -> Dict:
        combined_text = self._combine_texts(round_results)
        story_text = (self._combine_story_text(round_results) or combined_text).strip()
        if story_text:
            story_text = self._enforce_chapter_length_prompt_first(
                topic=topic,
                chapter_no=self.chapter_counter,
                story_text=story_text,
                chapter_target=chapter_target,
                chapter_min_required=chapter_min_required,
                chapter_max_allowed=chapter_max_allowed,
                genre=genre,
                style_tags=style_tags,
            )

        effective_results = list(round_results)
        if story_text and (not effective_results or story_text != (self._combine_story_text(round_results) or combined_text).strip()):
            effective_results = effective_results + [{
                "agent": "LengthManager",
                "role": "editor",
                "content": story_text,
                "type": "modified",
            }]

        self._log("Consistency", "Running consistency check...")
        baseline = self._build_consistency_baseline(topic, self.chapter_counter)
        consistency_result = self.consistency_checker.comprehensive_check(
            story_text,
            topic,
            baseline_outline=baseline,
        )

        self._log("RealtimeEditor", "Detecting sections that need revision...")
        turning_point_notes = self._get_recent_turning_point_notes(topic, limit=4)
        modifications = self.realtime_editor.detect_modification_needs(
            story_text,
            topic,
            consistency_result,
            turning_point_notes,
        )
        if modifications:
            self._log("RealtimeEditor", f"Detected {len(modifications)} sections that need revision.")
            story_text = self.realtime_editor.apply_modifications(story_text, modifications, topic)
            story_text = self._sanitize_chapter_text(story_text, self.chapter_counter)
            effective_results = effective_results + [{
                "agent": "RealtimeEditor",
                "role": "editor",
                "content": story_text,
                "type": "modified",
            }]
            self._log("RealtimeEditor", "Applied revisions.")

        evaluation_result: Dict = {}
        if getattr(self.config, "enable_evaluator", False):
            try:
                eval_payload = self.evaluator_agent.evaluate_multiple(effective_results)
                evaluation_result = eval_payload.get("evaluation", {}) or {}
            except Exception as exc:
                self._log("Evaluator", f"Chapter evaluation failed: {exc}")
                evaluation_result = {}

        reward_result = self.reward_system.calculate_reward(
            story_text,
            evaluation_result,
            consistency_result=consistency_result,
            target_length=chapter_target,
        )
        reward_score = float(reward_result.get("total_reward", 0.0) or 0.0)
        issues_count = len(consistency_result.get("all_issues", []))
        story_length = self._measure_text(story_text)
        self._append_progress_event(
            "agentic_score",
            (
                f"reward={reward_score:.3f}, issues={issues_count}, words={story_length}, "
                f"length_ok={chapter_min_required <= story_length <= chapter_max_allowed}"
            ),
            self.chapter_counter,
        )

        return {
            "story_text": story_text,
            "story_length": story_length,
            "round_results": effective_results,
            "evaluation": evaluation_result,
            "reward": reward_result,
            "reward_score": reward_score,
            "consistency": consistency_result,
            "issues_count": issues_count,
        }

    def _should_retry_chapter(
        self,
        candidate: Dict,
        chapter_min_required: int,
        chapter_max_allowed: int,
    ) -> bool:
        story_length = int(candidate.get("story_length", 0) or 0)
        reward_score = float(candidate.get("reward_score", 0.0) or 0.0)
        issues_count = int(candidate.get("issues_count", 0) or 0)
        evaluation = candidate.get("evaluation", {}) or {}
        suggestions = evaluation.get("suggestions", []) if isinstance(evaluation, dict) else []
        length_ok = chapter_min_required <= story_length <= chapter_max_allowed
        if not length_ok:
            return True
        if issues_count > 0:
            return True
        if reward_score < float(getattr(self.config, "min_quality_score", 0.7) or 0.7):
            return True
        return bool(suggestions and reward_score < 0.82)

    def _rewrite_chapter_with_feedback(
        self,
        *,
        topic: str,
        story_text: str,
        generated_content: List[Dict],
        evaluation_result: Dict,
        consistency_result: Dict,
        genre: Optional[str],
        style_tags: Optional[List[str]],
        chapter_target: int,
        chapter_min_required: int,
        chapter_max_allowed: int,
        subround: int,
    ) -> List[Dict]:
        planning_packet = self._build_planning_packet(
            topic,
            self.chapter_counter,
            current_goal=f"Rewrite chapter {self.chapter_counter}",
        )
        suggestions = evaluation_result.get("suggestions", []) if isinstance(evaluation_result, dict) else []
        suggestions_text = "\n".join(f"- {str(item).strip()}" for item in suggestions[:6] if str(item).strip()) or "- Keep what works and address the identified problems"
        issues = consistency_result.get("all_issues", []) if isinstance(consistency_result, dict) else []
        issues_text = "\n".join(f"- {str(item).strip()}" for item in issues[:6] if str(item).strip()) or "- No significant continuity problems. Refine the phrasing and pacing"
        context_tail = self._safe_excerpt(self._build_context(generated_content), 2200)
        chapter_title = planning_packet.get("chapter_title") or self._get_chapter_outline_title(topic, self.chapter_counter)

        prompt = (f"""You are now the execution layer controlled by a manager-style agentic loop. Rewrite the full current chapter instead of continuing a fixed workflow.

Task: rewrite the entire chapter {self.chapter_counter} and output the improved final chapter only.
Topic: {topic}
Chapter title: {chapter_title or 'Untitled'}
Target length: {chapter_min_required}-{chapter_max_allowed} words
Current agentic subround: #{subround}

Unified planning packet:
{planning_packet.get('packet_text') or 'Stay strictly aligned with the current chapter objective.'}

Current candidate draft:
{story_text}

Consistency issues:
{issues_text}

Evaluation suggestions:
{suggestions_text}

Additional context:
{context_tail or 'None'}

Hard requirements:
1) Do not follow a fixed workflow; directly repair the exposed problems;
2) Preserve correct plot goals, character relations, and world rules;
3) Expand if too short, compress if too long;
4) Output the rewritten chapter prose only, with no explanation or bullets.""")
        rewritten = self.agents["writer"].generate(
            prompt=prompt,
            context=story_text[-3000:],
            topic=topic,
            genre=genre,
            style_tags=style_tags,
            target_length=chapter_target,
            target_min_chars=chapter_min_required,
            target_max_chars=chapter_max_allowed,
        )
        return [rewritten]

    def _select_better_candidate(self, current: Dict, challenger: Dict) -> Dict:
        current_reward = float(current.get("reward_score", 0.0) or 0.0)
        challenger_reward = float(challenger.get("reward_score", 0.0) or 0.0)
        current_issues = int(current.get("issues_count", 0) or 0)
        challenger_issues = int(challenger.get("issues_count", 0) or 0)
        current_length = int(current.get("story_length", 0) or 0)
        challenger_length = int(challenger.get("story_length", 0) or 0)

        if challenger_reward > current_reward + 0.02:
            return challenger
        if challenger_issues < current_issues:
            return challenger
        if abs(challenger_reward - current_reward) <= 0.05 and current.get("story_text") and challenger.get("story_text"):
            winner = self._judge_between_candidates(current, challenger)
            if winner == "challenger":
                return challenger
            if winner == "current":
                return current
        if challenger_reward >= current_reward - 0.01 and challenger_length > current_length:
            return challenger
        return current

    def _judge_between_candidates(self, current: Dict, challenger: Dict) -> str:
        if self.judge_agent is None:
            return "tie"
        try:
            prompt = f"""Compare two candidate chapter versions and pick the stronger final version.\n\nCandidate A:\n{current.get('story_text','')}\n\nCandidate B:\n{challenger.get('story_text','')}\n\nReturn strict JSON only. winner must be candidate / reference / tie. Here candidate = version B, reference = version A."""
            result = self.judge_agent.generate(prompt)
            text = str(result.get("content") or "")
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                return "tie"
            parsed = json.loads(match.group(0))
            winner = str(parsed.get("winner") or "tie").strip().lower()
            if winner == "candidate":
                return "challenger"
            if winner == "reference":
                return "current"
            return "tie"
        except Exception:
            return "tie"

    def _execute_workflow(
        self,
        topic: str,
        execution_plan: Dict,
        previous_content: List[Dict],
        genre: Optional[str] = None,
        style_tags: Optional[List[str]] = None,
        target_length: Optional[int] = None,
        target_min_chars: Optional[int] = None,
        target_max_chars: Optional[int] = None,
        agent_sequence: Optional[List[str]] = None,
        use_static_kb: bool = True
    ) -> List[Dict]:
        """
        Execute workflow steps.

        Args:
            topic: Topic
            execution_plan: Execution plan
            previous_content: Previously generated content

        Returns:
            Current round's generation result
        """
        results = []
        context = self._build_context(previous_content)

        # Execute the Agent according to the plan
        seq = agent_sequence or execution_plan["agent_sequence"]
        for agent_name in seq:
            if agent_name not in self.agents:
                continue

            agent = self.agents[agent_name]
            print(f"[{agent.name}] Generating content...")
            agent.runtime_use_rag_override = bool(execution_plan.get("use_rag", True))
            agent.runtime_use_static_kb_override = bool(use_static_kb)

            # Construct the prompt
            prompt = self._build_agent_prompt(agent_name, topic, context, target_length)

            # Generate content (pass topic, genre, and style_tags to utilize dual RAG)
            try:
                if agent_name == "writer":
                    result = agent.generate(
                        prompt, context,
                        topic=topic,
                        genre=genre,
                        style_tags=style_tags,
                        target_length=target_length,
                        target_min_chars=target_min_chars,
                        target_max_chars=target_max_chars,
                    )
                else:
                    result = agent.generate(
                        prompt, context,
                        topic=topic,
                        genre=genre,
                        style_tags=style_tags
                    )
            finally:
                agent.runtime_use_rag_override = None
                agent.runtime_use_static_kb_override = None
            results.append(result)

            # Update context
            context += f"\n\n[{agent.name}]:\n{result.get('content', '')}"

        return results

    def _build_agent_prompt(
        self, agent_name: str, topic: str, context: str,
        target_length: Optional[int] = None,
    ) -> str:
        """Give each agent its own task while retaining established story context."""
        outline = self.memory_system.get_outline_by_topic(topic) or {}
        base = f"Topic: {topic}\n"
        if outline.get("content"):
            base += "Global outline:\n" + outline["content"] + "\n"
        if context:
            base += "Existing context:\n" + context + "\n"
        names = [str(c["name"]) for c in self.memory_system.get_characters_by_topic(topic) if c.get("name")]
        names = list(dict.fromkeys([*self.main_characters, *names]))
        if names:
            base += "Keep character names consistent: " + ", ".join(names) + "\n"
        if agent_name == "plot":
            return base + "Develop engaging plot and subplots with clear cause and effect."
        if agent_name == "character":
            return base + "Develop character motivations, relationships and behavioral constraints; preserve established identities."
        if agent_name == "world":
            return base + "Develop world rules and scene boundaries, connecting new details to established facts."
        if agent_name == "writer":
            chapter_outline = self._get_chapter_outline_text(topic, self.chapter_counter)
            title = self._get_chapter_outline_title(topic, self.chapter_counter)
            base += f"Current chapter: {self.chapter_counter} {title}\n"
            if chapter_outline:
                base += "Chapter outline:\n" + chapter_outline + "\n"
            if target_length and target_length > 0:
                base += f"Target length: {target_length} characters.\n"
            notes = self._get_recent_turning_point_notes(topic, limit=3)
            if notes:
                base += "Turning points: " + "; ".join(notes) + "\n"
            return base + "Write coherent chapter prose continuing the existing story. Introduce new characters or rules with context. Output prose only, without analysis, outlines or Markdown headings."
        return base + "Continue developing the existing material."

    def _build_context(self, previous_content: List[Dict]) -> str:
        """
        Construct context

        Args:
            previous_content: Previously generated content

        Returns:
            Context text
        """
        if not previous_content:
            return ""

        # Retain only the most recent entries to prevent infinite context expansion
        recent_items = previous_content[-self.config.recent_context_items:] if self.config.recent_context_items > 0 else previous_content

        context_parts = []
        for content in recent_items:
            agent_name = content.get("agent", "Unknown")
            text = content.get("content", "")
            context_parts.append(f"[{agent_name}]:\n{text}")

        combined = "\n\n".join(context_parts)
        max_chars = getattr(self.config, "context_max_chars", 12000)
        if max_chars and len(combined) > max_chars:
            # Retain tail entries that are closer to the current chapter
            combined = combined[-max_chars:]
        return combined

    def _combine_texts(self, results: List[Dict]) -> str:
        """
        Combine multiple generation results

        Args:
            results: List of generation results

        Returns:
            Combined text
        """
        combined = []
        for result in results:
            content = result.get("content", "")
            if content:
                combined.append(content)

        return "\n\n".join(combined)

    def _combine_story_text(self, results: List[Dict]) -> str:
        """
        Combine the 'Novel Text': Concatenate only writer and modified content to prevent mixing retrieval/outline materials into the final story.
        """
        combined = []
        for result in results:
            if result.get("type") not in {"writer", "modified"}:
                continue
            content = result.get("content", "")
            if content:
                combined.append(content)
        return "\n\n".join(combined)

    def _refine_final_text(self, text: str, topic: str) -> str:
        """
        Refine the final text.

        Args:
            text: Original text
            topic: Topic

        Returns:
            Refined text.
        """
        # Use LLM for final integration and optimization.
        prompt = (f"""Please merge the following content into a cohesive, complete English narrative (no bullet points or outlines).

Topic: {topic}

Content:
{text}

Ensure:
1) Logical flow and clear structure.
2) Remove repetition and contradictions.
3) Keep creativity and freshness.
4) Natural transitions between parts.
5) Use narrative prose (scenes, action, inner thoughts, dialogue as appropriate), with natural paragraphs.

Output only the merged text without extra explanation.""")

        messages = [
            {
                "role": "system",
                "content": "You are a professional text editor skilled at merging and polishing narrative prose."
            },
            {"role": "user", "content": prompt}
        ]

        try:
            refined = self.llm_client.chat(messages, temperature=0.5)
            return refined
        except:
            return text

    def _extract_character_name(self, text: str) -> Optional[str]:
        match = re.search(r"(?im)^\s*(?:[-*]\s*)?(?:character(?: name)?|name)\s*:\s*([^\n,;]+)", text)
        return match[1].strip().strip("*") if match else None

    def _extract_outline(self, text: str) -> Optional[str]:
        """Extract outline from text."""
        # Locate titles or chapter markers.
        import re
        lines = text.split('\n')
        outline_lines = []
        for line in lines:
            if self._parse_chapter_heading(line) or re.match(r'^#{1,3}\s', line):
                outline_lines.append(line.strip())
        return '\n'.join(outline_lines) if outline_lines else None

    def _summarize_chapter(self, text: str, topic: str, chapter_index: int) -> str:
        """
        Generate chapter summaries and key points to facilitate subsequent retrieval and consistency control.
        """
        prompt = (f"""Summarize the following chapter and list key characters/events/settings (max 400 words):

Topic: {topic}
Chapter: {chapter_index}
Text:
{text}

Output:
1) Concise summary (200-400 words)
2) Key characters & relationships
3) Key events / foreshadowing
4) Important settings (rules/time/place).""")
        messages = [
            {"role": "system", "content": "You are a long-form chapter summarization assistant who keeps consistency."},
            {"role": "user", "content": prompt}
        ]
        try:
            return self.llm_client.chat(messages, temperature=0.3, max_tokens=self.config.max_tokens)
        except Exception:
            return ""

    def _generate_global_outline(
        self,
        idea: str,
        topic: str,
        genre: Optional[str],
        style_tags: Optional[List[str]],
        target_length: int
    ) -> str:
        """Generate a global outline (chapter structure) to serve as an anchor for long-form consistency."""
        style = ", ".join(style_tags or [])
        est = self._estimate_chapter_count(target_length)
        low = max(5, est - 1)
        high = min(30, est + 2)
        per_hint = getattr(self.config, "per_chapter_target_hint", 3300) or 3300
        prompt = (f"""Based on the premise/idea below, generate a full-book outline for long-form writing. Requirements:
1) Provide {low}-{high} chapter titles with 1-3 bullet sentences each (target ~{per_hint} words per chapter, balanced overall).
2) List core character cards (character: goals/conflicts/relationships).
3) List hard constraints for world/setting rules.
4) List major foreshadowing and expected payoff chapters.
5) Final chapter must deliver a clear ending and resolve main conflicts and foreshadowing—no cliffhangers.
6) Keep it concrete and executable, not vague.

Topic: {topic}
Genre: {genre or 'general'}
Style: {style or 'unspecified'}
Target length: {target_length} words
Idea: {idea}
""")
        messages = [
            {"role": "system", "content": "You are a head writer for long-form fiction, great at producing actionable chapter outlines and setting cards."},
            {"role": "user", "content": prompt}
        ]
        return self.llm_client.chat(messages, temperature=0.4, max_tokens=self.config.max_tokens)

    def _estimate_chapter_count(self, target_length: int) -> int:
        """
        Estimate the number of chapters based on the target word count to avoid excessive splitting caused by the default 12-24 chapters.
        """
        per_hint = getattr(self.config, "per_chapter_target_hint", 3300) or 3300
        est = math.ceil(max(target_length or 20000, 5000) / per_hint)
        # Basic protection: 5-30 chapters.
        return max(5, min(30, est))

    def _latest_outline_by_kind(self, topic: str, kind: str) -> Optional[Dict[str, Any]]:
        outlines = self.memory_system.get_recent_outlines(topic, limit=1, kind=kind)
        return outlines[-1] if outlines else None

    def _ensure_strategic_outline_assets(
        self,
        *,
        idea: str,
        topic: str,
        genre: Optional[str],
        style_tags: Optional[List[str]],
        target_length: int,
    ) -> int:
        global_entry = self._latest_outline_by_kind(topic, "global_outline")
        global_outline = str((global_entry or {}).get("content") or "").strip()
        generated_global = False

        if not global_outline:
            global_outline = str(
                self._generate_global_outline(
                    idea=idea,
                    topic=topic,
                    genre=genre,
                    style_tags=style_tags,
                    target_length=target_length,
                )
                or ""
            ).strip()
            if global_outline:
                try:
                    self.book_title = self._extract_book_title(global_outline) or topic
                except Exception:
                    self.book_title = topic
                self.memory_system.store_outline(
                    global_outline,
                    topic,
                    structure={"kind": "global_outline", "source": "claw_bootstrap"},
                )
                self._append_progress_event("global_outline", self._safe_excerpt(global_outline, 320))
                self._log("Outline", "Claw strategic global outline generated and stored in dynamic memory.")
                generated_global = True

        chapter_outlines = self.memory_system.get_recent_outlines(topic, limit=200, kind="chapter_outline")
        planned_total = len(chapter_outlines)

        if global_outline and planned_total <= 0:
            stored = self._precompute_chapter_outlines(global_outline, topic, genre=genre, style_tags=style_tags)
            if not stored:
                stored = self._precompute_chapter_outlines(global_outline, topic, genre=genre, style_tags=style_tags)
            planned_total = len(stored)
            if planned_total > 0:
                self._append_progress_event("chapter_outline_ready", f"count={planned_total}")
                self._log("Outline", f"Claw strategic chapter outlines prepared: {planned_total} chapters.")

        if generated_global:
            try:
                outline_cards = self._run_outline_stage_agents(global_outline, topic, genre, style_tags)
                if outline_cards:
                    self._log("Outline", f"Generated {len(outline_cards)} character/world/retrieval setup notes from the strategic outline.")
            except Exception as e:
                self._log("Outline", f"Strategic-outline derivative setup failed: {e}")

        return planned_total

    def _build_planning_packet(
        self,
        topic: str,
        chapter_no: int,
        *,
        current_goal: str = "",
        support_results: Optional[List[Dict[str, Any]]] = None,
        limit_per_bank: int = 3,
    ) -> Dict[str, Any]:
        chapter_title = self._get_chapter_outline_title(topic, chapter_no)
        chapter_outline = self._get_chapter_outline_text(topic, chapter_no).strip()
        global_outline = str(((self._latest_outline_by_kind(topic, "global_outline") or {}).get("content")) or "").strip()
        rolling_summary = str(((self._latest_outline_by_kind(topic, "rolling_summary") or {}).get("content")) or "").strip()

        selected_banks = [
            "chapter_briefs",
            "scene_cards",
            "entity_state",
            "relationship_state",
            "world_state",
            "continuity_facts",
            "working_set",
        ]
        grouped = self.memory_system.get_claw_memories(
            topic,
            banks=selected_banks,
            limit_per_bank=max(limit_per_bank * 3, 8),
        )

        def same_chapter(item: Dict[str, Any]) -> bool:
            meta = item.get("metadata") or {}
            value = meta.get("chapter")
            try:
                return int(value) == int(chapter_no)
            except Exception:
                return False

        def pick_entries(bank: str) -> List[Dict[str, Any]]:
            entries = list(grouped.get(bank) or [])
            chapter_entries = [item for item in entries if same_chapter(item)]
            chosen = chapter_entries[-limit_per_bank:] if chapter_entries else entries[-limit_per_bank:]
            return chosen

        def format_entries(entries: List[Dict[str, Any]], per_entry_limit: int = 1200) -> str:
            blocks: List[str] = []
            for index, item in enumerate(entries, start=1):
                content = str(item.get("content") or "").strip()
                if not content:
                    continue
                meta = item.get("metadata") or {}
                meta_parts = []
                if meta.get("chapter") is not None:
                    meta_parts.append(f"chapter={meta.get('chapter')}")
                if meta.get("source_type"):
                    meta_parts.append(f"source={meta.get('source_type')}")
                if meta.get("tool"):
                    meta_parts.append(f"tool={meta.get('tool')}")
                header = f"[{index}]"
                if meta_parts:
                    header += " " + " | ".join(meta_parts)
                blocks.append(f"{header}\n{self._safe_excerpt(content, per_entry_limit)}")
            return "\n\n".join(blocks).strip()

        support_blocks: Dict[str, List[str]] = {}
        for item in list(support_results or [])[-8:]:
            result_type = str(item.get("type") or item.get("role") or "").strip()
            content = str(item.get("content") or "").strip()
            if not result_type or not content:
                continue
            support_blocks.setdefault(result_type, []).append(self._safe_excerpt(content, 1200))

        chapter_briefs = pick_entries("chapter_briefs")
        scene_cards = pick_entries("scene_cards")
        entity_state = pick_entries("entity_state")
        relationship_state = pick_entries("relationship_state")
        world_state = pick_entries("world_state")
        continuity_facts = pick_entries("continuity_facts")
        working_set = pick_entries("working_set")

        sections: List[str] = []
        if current_goal:
            sections.append(f"=== CURRENT GOAL ===\n{current_goal.strip()}")
        if global_outline:
            sections.append(f"=== STRATEGIC GLOBAL OUTLINE ===\n{self._safe_excerpt(global_outline, 2800)}")
        if chapter_outline:
            sections.append(f"=== STRATEGIC CHAPTER OUTLINE ===\n{chapter_outline}")
        if chapter_title:
            sections.append(f"=== CHAPTER TITLE ===\n{chapter_title}")
        if rolling_summary:
            sections.append(f"=== ROLLING SUMMARY ===\n{self._safe_excerpt(rolling_summary, 1800)}")
        if support_blocks.get("plot"):
            sections.append(f"=== TACTICAL LIVE PLOT STRATEGY ===\n{support_blocks['plot'][-1]}")
        brief_text = format_entries(chapter_briefs, per_entry_limit=1600)
        if brief_text:
            sections.append(f"=== TACTICAL CHAPTER BRIEFS ===\n{brief_text}")
        scene_text = format_entries(scene_cards, per_entry_limit=1600)
        if scene_text:
            sections.append(f"=== TACTICAL SCENE CARDS ===\n{scene_text}")
        if support_blocks.get("character"):
            sections.append(f"=== LIVE CHARACTER CONSTRAINTS ===\n{support_blocks['character'][-1]}")
        if support_blocks.get("world"):
            sections.append(f"=== LIVE WORLD CONSTRAINTS ===\n{support_blocks['world'][-1]}")

        continuity_sections = [
            ("ENTITY STATE", entity_state),
            ("RELATIONSHIP STATE", relationship_state),
            ("WORLD STATE", world_state),
            ("CONTINUITY FACTS", continuity_facts),
            ("WORKING SET", working_set),
        ]
        for label, entries in continuity_sections:
            text = format_entries(entries, per_entry_limit=1000)
            if text:
                sections.append(f"=== {label} ===\n{text}")

        packet_text = "\n\n".join(section for section in sections if section.strip()).strip()
        return {
            "chapter_title": chapter_title,
            "global_outline": global_outline,
            "chapter_outline": chapter_outline,
            "chapter_briefs": chapter_briefs,
            "scene_cards": scene_cards,
            "packet_text": packet_text,
        }

    def _update_rolling_summary(self, topic: str, new_chapter_summary: str) -> str:
        """
        Update the rolling summary with new chapter summaries (keep within 800-1200 words).
        """
        prev = self.memory_system.get_recent_outlines(topic, limit=1, kind="rolling_summary")
        prev_text = prev[-1].get("content", "") if prev else ""

        prompt = (f"""Merge the previous rolling summary with the new chapter summary into an updated rolling summary:
Requirements: 1) Keep global key character states, critical settings, main-plot progress, unresolved foreshadowing; 2) Drop overly fine details; 3) Target 800-1200 words.

Previous rolling summary:
{prev_text}

New chapter summary:
{new_chapter_summary}
""")
        messages = [
            {"role": "system", "content": "You are a rolling-summary assistant for long-form fiction, maintaining global consistency."},
            {"role": "user", "content": prompt}
        ]
        try:
            return self.llm_client.chat(messages, temperature=0.3, max_tokens=min(self.config.max_tokens, 2000))
        except Exception:
            return ""

    def _extract_fact_cards(self, text: str, topic: str, chapter_index: int) -> List[str]:
        """
        Extract 'fact cards' (short entries) from chapter content to enforce alignment in subsequent writing and reduce hallucinations.
        """
        prompt = (f"""Extract concise "fact cards" from the chapter text below (each <=120 words):
- Character state changes (including motivation/relationship shifts)
- Key events and causality
- World/setting hard constraints (new/confirmed)
- Foreshadowing and promises (to be fulfilled later)

Rules:
1) Only include facts explicitly stated or directly implied; no fabrication.
2) Output as bullet list (one per line), max 10 items.

Topic: {topic}
Chapter: {chapter_index}
Text:
{text}
""")
        messages = [
            {"role": "system", "content": "You are a fact-extraction assistant; extract only explicit or directly implied facts, no speculation."},
            {"role": "user", "content": prompt}
        ]
        try:
            raw = self.llm_client.chat(messages, temperature=0.2, max_tokens=1200)
        except Exception:
            return []

        lines = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(("-", "*", "•")):
                line = line.lstrip("-*•").strip()
            if len(line) >= 10:
                lines.append(line)
        # Deduplicate while preserving order.
        dedup = []
        seen = set()
        for l in lines:
            if l not in seen:
                dedup.append(l)
                seen.add(l)
        return dedup[:10]

    def _build_consistency_baseline(self, topic: str, chapter_no: int) -> str:
        """
        Build the baseline information for consistency checks: rolling summary + current chapter outline/title + protagonist anchor + recent world-building + recent fact cards + previous chapter summary.
        """
        parts: List[str] = []
        rolling = self.memory_system.get_recent_outlines(topic, limit=1, kind="rolling_summary")
        if rolling and rolling[-1].get("content"):
            parts.append("[Rolling summary]")
            parts.append(rolling[-1]["content"])
        chapter_outline = self._get_chapter_outline_text(topic, chapter_no)
        if chapter_outline:
            parts.append("[Chapter outline]")
            parts.append(chapter_outline)
        title = self._get_chapter_outline_title(topic, chapter_no)
        if title:
            parts.append(f"[Chapter title] {title}")
        if self.main_characters:
            parts.append(f"[Main characters] {', '.join(self.main_characters)}")
        last_summary = self._get_last_chapter_summary(topic, chapter_no - 1)
        if last_summary:
            parts.append("[Previous chapter summary]")
            parts.append(last_summary)
        recent_facts = self._get_recent_fact_cards(topic, limit=4)
        if recent_facts:
            parts.append("[Recent fact cards]")
            parts.extend(recent_facts)
        world_ctx = self._get_recent_world_settings(topic, limit=2)
        if world_ctx:
            parts.append("[World setting]")
            parts.extend(world_ctx)
        return "\n".join(parts)

    def _run_outline_stage_agents(self, outline_text: str, topic: str, genre: Optional[str], style_tags: Optional[List[str]]) -> List[Dict]:
        """
        Pre-generate core character cards, world-building settings, and background retrieval notes based on the global outline, storing them in memory for reference in subsequent chapters.
        """
        results = []
        context = outline_text
        for agent_name in ["character", "world", "retrieval"]:
            if agent_name not in self.agents:
                continue
            agent = self.agents[agent_name]
            # Use stricter prompts tailored for the outlining phase: do not add content outside the outline.
            prompt = (
                f"Book title: {topic}\n"
                f"Overall outline:\n{outline_text}\n\n"
            )
            if agent_name == "character":
                prompt += (
                    "Create core character profiles from the outline (names, traits, backgrounds, goals, conflicts, and relationships). "
                    "Use the names from the outline and stay within its established plot and locations."
                )
            elif agent_name == "world":
                prompt += (
                    "Develop the world from the outline (background, rules, and systems). "
                    "Keep the world consistent with the outline and its rules for supernatural elements."
                )
            else:
                prompt += (
                    "Identify the historical, cultural, and background details relevant to the outline. "
                    "Keep these writing references within the outline's established story and setting."
                )
            try:
                res = agent.generate(prompt, context, topic=topic, genre=genre, style_tags=style_tags)
            except Exception:
                continue
            content = res.get("content", "")
            rtype = res.get("type", agent_name)
            if not content:
                continue
            if rtype == "character":
                name = self._extract_character_name(content)
                if name and self._is_valid_character_name(name):
                    self.memory_system.store_character(name, content, topic)
                    self._append_progress_event(
                        "outline_character",
                        f"{name} | {self._safe_excerpt(content, 260)}",
                    )
            elif rtype == "world":
                # Use fixed names to avoid confusion with the topic.
                setting_name = "Core world setting"
                self.memory_system.store_world_setting(setting_name, content, topic)
                self._append_progress_event(
                    "outline_world",
                    self._safe_excerpt(content, 260),
                )
            elif rtype == "retrieval":
                self.memory_system.store_generated_text(
                    content, topic, {"source": "outline_stage_retrieval"}, store_vector=True
                )
                self._append_progress_event(
                    "outline_retrieval",
                    self._safe_excerpt(content, 260),
                )
            results.append({"type": rtype, "content": content})
        return results

    # =====================
    # Outline splitting and chapter outline retrieval.
    # =====================
    def _precompute_chapter_outlines(self, global_outline: str, topic: str, genre: Optional[str] = None, style_tags: Optional[List[str]] = None) -> List[Dict]:
        """
        Based on the global outline, use an Agent to generate chapter outlines and write them to memory for use in subsequent chapter writing.
        """
        try:
            chapters = self._generate_chapter_outlines_by_agent(global_outline, topic, genre=genre, style_tags=style_tags)
        except Exception:
            chapters = []
        if not chapters:
            self._log("Outline", "Warning: chapter outlines are empty. Please check the outline or the prompt.")
        stored = []
        for chap in chapters:
            chapter_no = chap.get("chapter")
            title = chap.get("title")
            content = chap.get("content", "")
            if chapter_no is None:
                continue
            # Check if the chapter outline already exists to avoid duplicate writes.
            existing = [
                o for o in self.memory_system.memory_index.get("outlines", [])
                if o.get("topic") == topic
                and o.get("structure", {}).get("kind") == "chapter_outline"
                and o.get("structure", {}).get("chapter") == chapter_no
            ]
            if existing:
                continue
            structure = {"kind": "chapter_outline", "chapter": chapter_no, "title": title, "source": "from_global"}
            self.memory_system.store_outline(content, topic, structure=structure)
            stored.append({"chapter": chapter_no, "title": title})
        return stored

    def _split_outline_to_chapters(self, outline_text: str) -> List[Dict]:
        """
        Simply split the outline by 'Chapter X' titles to extract chapter numbers, titles, and main content points.
        """
        lines = outline_text.splitlines()
        chapters = []
        current = None
        buf = []

        def flush():
            nonlocal current, buf
            if current:
                content = "\n".join(buf).strip()
                chapters.append({
                    "chapter": current.get("chapter"),
                    "title": current.get("title"),
                    "content": content
                })
            buf = []

        for line in lines:
            heading = self._parse_chapter_heading(line)
            if heading:
                flush()
                current = heading
            else:
                if current:
                    buf.append(line)
        flush()
        return chapters

    def _generate_chapter_outlines_by_agent(self, outline_text: str, topic: str, genre: Optional[str] = None, style_tags: Optional[List[str]] = None) -> List[Dict]:
        """
        Generate chapter outlines (JSON) via an LLM Agent, strictly based on the global outline, without adding new main plotlines, characters, or settings.
        """
        style = ", ".join(style_tags or [])
        prompt = (f"""Based on the global outline below, generate chapter-level outlines as a JSON array:
[
  {{"chapter":1,"title":"Chapter Title","content":"1-3 key sentences"}},
  ...
]
Requirements:
1) Chapter numbers start at 1 and must match the global outline count; no adding/removing chapters.
2) Titles must match (or lightly adjust) the corresponding titles from the global outline; no renaming/jumping.
3) content should include 1-3 key sentences strictly derived from the global outline; do NOT introduce new plotlines/characters/settings/locations.
4) Preserve logical order and resolution; keep foreshadowing/payoff positions aligned.
5) Output valid JSON; each element has chapter/title/content fields.

Topic: {topic}
Genre: {genre or 'general'}
Style: {style or 'unspecified'}
Global outline:
{outline_text}
""")
        messages = [
            {"role": "system", "content": "You are a chapter-planning assistant for long-form fiction; output executable chapter outlines."},
            {"role": "user", "content": prompt}
        ]
        try:
            raw = self.llm_client.chat(messages, temperature=0.4, max_tokens=min(self.config.max_tokens, 3000))
        except Exception:
            return []
        chapters: List[Dict] = []
        # Attempt to parse the JSON array (extract even if the model output is wrapped in other text).
        try:
            block = self._extract_json_array(raw)
            data = json.loads(block)
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    ch = item.get("chapter")
                    title = item.get("title") or ""
                    content = item.get("content") or ""
                    if ch is None:
                        continue
                    chapters.append({"chapter": int(ch), "title": title, "content": content})
        except Exception:
            # fallback: simple line parsing, format: Chapter X Title: Key Points
            for line in raw.splitlines():
                m = self._parse_chapter_heading(line)
                if m:
                    chapters.append({"chapter": m.get("chapter"), "title": m.get("title"), "content": ""})
        return chapters

    def _parse_chapter_heading(self, line: str) -> Optional[Dict]:
        match = re.match(r"^\s*(?:#{1,6}\s*)?chapter\s+(\d+)\b\s*[:.\-–—]?\s*(.*)$", line, re.I)
        if match:
            return {"chapter": int(match[1]), "title": match[2].strip()}
        return None


    def _get_recent_turning_point_notes(self, topic: str, limit: int = 3) -> List[str]:
        notes = [
            o for o in self.memory_system.memory_index.get("outlines", [])
            if o.get("topic") == topic
            and o.get("structure", {}).get("kind") == "turning_point_issue"
        ]
        tail = notes[-limit:] if limit > 0 else notes
        return [n.get("content", "") for n in tail if n.get("content")]

    def _get_chapter_outline_text(self, topic: str, chapter_no: int) -> str:
        """
        Retrieve the outline text for a specified chapter; return an empty string if none exists.
        """
        outlines = [
            o for o in self.memory_system.memory_index.get("outlines", [])
            if o.get("topic") == topic
            and o.get("structure", {}).get("kind") == "chapter_outline"
            and o.get("structure", {}).get("chapter") == chapter_no
        ]
        if not outlines:
            return ""
        return outlines[-1].get("content", "") or ""

    def _get_chapter_outline_title(self, topic: str, chapter_no: int) -> str:
        """
        Retrieve the outline title for a specified chapter (prioritize structure.title, then parse the content).
        """
        outlines = [
            o for o in self.memory_system.memory_index.get("outlines", [])
            if o.get("topic") == topic
            and o.get("structure", {}).get("kind") == "chapter_outline"
            and o.get("structure", {}).get("chapter") == chapter_no
        ]
        if outlines:
            title = outlines[-1].get("structure", {}).get("title")
            if title:
                return title
            parsed = self._parse_chapter_heading(outlines[-1].get("content", ""))
            if parsed and parsed.get("title"):
                return parsed.get("title")
        return ""

    def _extract_book_title(self, text: str) -> Optional[str]:
        """
        Extract the book title from the outline text; return None if not found.
        """
        try:
            m = re.search(r'《([^》]+)》', text)
            if m:
                return m.group(1).strip()
        except Exception:
            pass
        return None

    def _is_valid_character_name(self, name: str) -> bool:
        return bool(name and len(name) <= 80 and all(c.isalpha() or c in " ’'-." for c in name)
                    and name.casefold() not in {"the protagonist", "protagonist", "unknown", "unnamed"})

    def _extract_main_characters(self, outline_text: str) -> List[str]:
        names = []
        for line in outline_text.splitlines():
            match = re.search(r"(?i)\b(?:protagonist|main character|lead character)\s*:\s*([^,;(\n]+)", line)
            if not match:
                match = re.match(r"(?i)^\s*[-* ]*([^:(\n]+?)\s*\((?:protagonist|main character|lead character)\)", line)
            if match:
                name = match[1].strip().strip("*")
                if self._is_valid_character_name(name) and name not in names:
                    names.append(name)
        return names

    def _get_last_chapter_summary(self, topic: str, chapter_no: int) -> str:
        """
        Retrieve the summary of the previous chapter (kind=chapter_summary).
        """
        if chapter_no <= 0:
            return ""
        summaries = [
            o for o in self.memory_system.memory_index.get("outlines", [])
            if o.get("topic") == topic
            and o.get("structure", {}).get("kind") == "chapter_summary"
            and o.get("structure", {}).get("chapter") == chapter_no
        ]
        if not summaries:
            return ""
        return summaries[-1].get("content", "") or ""

    def _get_recent_fact_cards(self, topic: str, limit: int = 4) -> List[str]:
        """
        Retrieve the content of the most recent fact cards (from memory_index).
        """
        cards = [
            c for c in self.memory_system.memory_index.get("fact_cards", [])
            if c.get("topic") == topic
        ]
        tail = cards[-limit:] if limit > 0 else cards
        return [c.get("content", "") for c in tail if c.get("content")]

    def _get_recent_world_settings(self, topic: str, limit: int = 2) -> List[str]:
        """
        Retrieve the most recent world-building setting text (via vector search, to avoid issues with empty memory_index).
        """
        try:
            memories = self.memory_system.retrieve_memories(
                "World setting",
                memory_types=["world_setting"],
                topic=topic,
                top_k=limit
            )
            return [m.get("text", "") for m in memories if m.get("text")]
        except Exception:
            return []

    def _extract_json_array(self, text: str) -> str:
        """
        Extract the first JSON array fragment from the model output.
        """
        import re
        m = re.search(r'\[[\s\S]*\]', text)
        if m:
            return m.group(0)
        return text

    def _get_run_dir_path(self):
        """Return Path to current run directory, or None if not available."""
        import os
        from pathlib import Path
        run_id = str(getattr(self.config, "run_id", "") or "").strip()
        runs_dir = str(getattr(self.config, "runs_dir", "") or "runs").strip()
        if not run_id:
            return None
        return Path(runs_dir) / run_id

    def _signal_chapter_complete(self, chapter_no: int, chars: int, planned_total: int) -> None:
        """Write claw_chapter_complete.json so the web frontend can show a between-chapter UI."""
        import time, json
        run_dir = self._get_run_dir_path()
        if run_dir is None:
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            (run_dir / "claw_chapter_msg.json").unlink(missing_ok=True)
        except Exception:
            pass
        atomic_write_text(run_dir / "claw_chapter_complete.json",
            json.dumps({
                "chapter": chapter_no,
                "chars": chars,
                "planned_total": planned_total,
                "ts": time.time(),
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        try:
            self._append_progress_event(
                "claw_chapter_checkpoint",
                f"chapter={chapter_no}, planned_total={planned_total}, chars={chars}",
                chapter_no,
            )
        except Exception:
            pass

    def _wait_for_between_chapter_interaction(self, chapter_no: int, planned_total: int) -> None:
        """
        In web mode, pause after each non-final chapter until the chat UI tells us
        to continue. The frontend does that by clearing claw_chapter_complete.json,
        optionally writing claw_chapter_msg.json first.
        """
        import os
        import time

        if chapter_no >= planned_total:
            return
        if not str(os.environ.get("RUN_ID", "") or "").strip():
            return

        run_dir = self._get_run_dir_path()
        if run_dir is None:
            return

        complete_path = run_dir / "claw_chapter_complete.json"
        msg_path = run_dir / "claw_chapter_msg.json"
        if not complete_path.exists():
            return

        print(f"[OpenClaw] Waiting for between-chapter input after chapter {chapter_no}")
        while complete_path.exists():
            if msg_path.exists():
                break
            time.sleep(0.8)
        print(f"[OpenClaw] Between-chapter wait cleared after chapter {chapter_no}")

    def _handle_between_chapter_interaction(self, topic: str) -> None:
        """
        Check for a user instruction left between chapters via file IPC.
        If claw_chapter_msg.json exists, read it, inject into Claw memory,
        and emit a progress event so the trace reflects the user's input.
        """
        import json
        run_dir = self._get_run_dir_path()
        if run_dir is None:
            return
        msg_path = run_dir / "claw_chapter_msg.json"
        if not msg_path.exists():
            return
        try:
            payload = json.loads(msg_path.read_text(encoding="utf-8"))
            message = str(payload.get("message") or "").strip()
            memory_updates = payload.get("memory_updates") if isinstance(payload.get("memory_updates"), list) else []
            if message:
                self.memory_system.store_claw_memory(
                    "decision_log",
                    f"[User instruction between chapters — ch{self.chapter_counter}→{self.chapter_counter+1}]\n{message}",
                    topic,
                    metadata={"chapter": self.chapter_counter, "type": "between_chapter"},
                    store_vector=True,
                )
                self._append_progress_event(
                    "claw_user_chapter_instruction",
                    f"chapter={self.chapter_counter}, instruction={message[:240]}",
                    self.chapter_counter,
                )
                print(f"[OpenClaw] Between-chapter instruction applied: {message[:120]}")
            applied_banks = []
            for item in memory_updates:
                if not isinstance(item, dict):
                    continue
                bank = str(item.get("bank") or "").strip()
                content = str(item.get("content") or "").strip()
                if bank not in self.memory_system.CLAW_BANKS or not content:
                    continue
                self.memory_system.store_claw_memory(
                    bank,
                    content,
                    str(item.get("topic") or "").strip() or topic,
                    metadata={
                        "chapter": self.chapter_counter,
                        "kind": str(item.get("kind") or "between_chapter_note").strip() or "between_chapter_note",
                        "source": str(item.get("source") or "checkpoint_ui").strip() or "checkpoint_ui",
                        "user_supplied": True,
                    },
                    store_vector=False,
                )
                applied_banks.append(bank)
            if applied_banks:
                banks_summary = ",".join(applied_banks[:8])
                self.memory_system.store_claw_memory(
                    "decision_log",
                    f"[User memory update between chapters — ch{self.chapter_counter}→{self.chapter_counter+1}]\nbanks={banks_summary}",
                    topic,
                    metadata={"chapter": self.chapter_counter, "type": "between_chapter_memory"},
                    store_vector=False,
                )
                self._append_progress_event(
                    "claw_user_memory_update",
                    f"chapter={self.chapter_counter}, banks={banks_summary}",
                    self.chapter_counter,
                )
                print(f"[OpenClaw] Between-chapter memory updates applied: {banks_summary}")
        except Exception:
            pass
        finally:
            try:
                msg_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _write_chapter_file(self, chapter_no: int, story_text: str, iteration: int, topic: str):
        """
        Write the final text of the chapter to the runs/<run_id>/chapters directory for traceability.
        """
        import os
        run_dir = os.path.join(self.config.runs_dir, self.config.run_id, "chapters")
        os.makedirs(run_dir, exist_ok=True)
        fname = f"chapter_{chapter_no:02d}_iter_{iteration:02d}_final.txt"
        path = os.path.join(run_dir, fname)
        book_title = self.book_title or topic
        header = f"Book: {book_title}\nChapter: {chapter_no}\n\n"
        save_text_version(Path(path), header + story_text)

    def _safe_excerpt(self, text: str, limit: int = 300) -> str:
        cleaned = " ".join((text or "").split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[:limit] + "...(truncated)"

    def _extract_length_range_from_text(self, text: str) -> Optional[Tuple[int, int]]:
        if not text:
            return None
        # Example: 3200-3600 words / 3200~3600 words / 3200 to 3600 words
        m = re.search(r'(\d{3,6})\s*(?:[-~–—]|to)\s*(\d{3,6})\s*words?\b', text, re.I)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            low, high = (a, b) if a <= b else (b, a)
            return low, high
        # Example: approximately 3300 words per chapter
        m = re.search(r'(?:per chapter|each chapter|chapter target)\D{0,12}(\d{3,6})\s*words?\b', text, re.I)
        if m:
            t = int(m.group(1))
            low = max(800, int(t * 0.9))
            high = max(low + 200, int(t * 1.1))
            return low, high
        return None

    def _get_chapter_length_bounds(self, topic: str, chapter_no: int, chapter_target: int) -> Tuple[int, int, str]:
        chapter_outline = self._get_chapter_outline_text(topic, chapter_no)
        r = self._extract_length_range_from_text(chapter_outline)
        if r:
            return r[0], r[1], "chapter_outline"

        global_outline = ""
        try:
            go = self.memory_system.get_outline_by_topic(topic)
            if go:
                global_outline = go.get("content", "")
        except Exception:
            global_outline = ""
        r = self._extract_length_range_from_text(global_outline)
        if r:
            return r[0], r[1], "global_outline"

        low = max(800, int((chapter_target or 1000) * 0.9))
        high = max(low + 200, int((chapter_target or 1000) * 1.12))
        return low, high, "target_fallback"

    def _measure_text(self, text: str) -> int:
        return text_length(text, self.lang)

    def _truncate_text_soft(self, text: str, limit: int) -> str:
        return truncate_text(text, limit, self.lang)

    def _is_chapter_heading_line(self, line: str, chapter_no: int) -> bool:
        stripped = str(line or "").strip()
        if not stripped:
            return False
        chapter_token = str(chapter_no) if chapter_no > 0 else r"\d+"
        patterns = [
            rf"^(?:#+\s*)?(?:chapter|ch\.)\s*{chapter_token}(?:\s*[:：\-]\s*.*)?$",
        ]
        return any(re.match(pattern, stripped, re.IGNORECASE) for pattern in patterns)

    def _sanitize_chapter_text(self, text: str, chapter_no: int) -> str:
        value = str(text or "").replace("\r\n", "\n").strip()
        if not value:
            return ""
        cleaned_lines: List[str] = []
        for line in value.split("\n"):
            stripped = line.strip()
            if not stripped:
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                continue
            if stripped.strip("()[] ").casefold().rstrip(".") in {"to be continued", "continued", "end of chapter"}:
                continue
            cleaned_lines.append(line.rstrip())
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()
        return "\n".join(cleaned_lines).strip()

    def _sanitize_chapter_continuation(self, text: str, chapter_no: int) -> str:
        value = self._sanitize_chapter_text(text, chapter_no)
        if not value:
            return ""
        lines = value.split("\n")
        while lines and self._is_chapter_heading_line(lines[0], chapter_no):
            lines.pop(0)
        return "\n".join(lines).strip()

    def _enforce_chapter_length_prompt_first(
        self,
        topic: str,
        chapter_no: int,
        story_text: str,
        chapter_target: int,
        chapter_min_required: int,
        chapter_max_allowed: int,
        genre: Optional[str],
        style_tags: Optional[List[str]],
    ) -> str:
        text = self._sanitize_chapter_text(story_text, chapter_no)
        if not text:
            return text

        max_rounds = max(1, int(getattr(self.config, "max_chapter_subrounds", 3) or 3))
        for _ in range(max_rounds):
            cur_len = self._measure_text(text)
            if chapter_min_required <= cur_len <= chapter_max_allowed:
                break

            if cur_len > chapter_max_allowed:
                prompt = (f"""Rewrite this chapter to strictly {chapter_min_required}-{chapter_max_allowed} words.
Requirements:
1) Keep core plot/characters/world rules unchanged, no new main plotline;
2) Remove repetition and verbose passages;
3) Output prose only, no explanations or bullets;
4) Exceeding {chapter_max_allowed} words is considered failure.

Original chapter:
{text}
""")
                try:
                    rewritten = self.agents["writer"].generate(
                        prompt=prompt,
                        context="",
                        topic=topic,
                        genre=genre,
                        style_tags=style_tags,
                        target_length=chapter_target,
                        target_min_chars=chapter_min_required,
                        target_max_chars=chapter_max_allowed,
                    ).get("content", "")
                    if rewritten:
                        text = self._sanitize_chapter_text(rewritten, chapter_no)
                        self._append_progress_event(
                            "chapter_length_fix",
                            f"mode=shrink_to_range, words={self._measure_text(text)}",
                            chapter_no,
                        )
                        continue
                except Exception:
                    pass
                text = self._truncate_text_soft(text, chapter_max_allowed)
                self._append_progress_event(
                    "chapter_length_fix",
                    f"mode=truncate_fallback, words={self._measure_text(text)}",
                    chapter_no,
                )
                continue

            remain = chapter_min_required - cur_len
            append_target = max(600, min(remain + 240, chapter_max_allowed - cur_len))
            if append_target <= 0:
                break
            continue_prompt = (f"""Continue chapter {chapter_no} and bring total length into {chapter_min_required}-{chapter_max_allowed} words.
Requirements:
1) Continue directly from existing text; no restart and no repetition;
2) Keep character/world consistency; no out-of-outline additions;
3) Output continuation only.

Current chapter text:
{text}
""")
            try:
                continuation = self.agents["writer"].generate(
                    prompt=continue_prompt,
                    context=text[-3000:],
                    topic=topic,
                    genre=genre,
                    style_tags=style_tags,
                    target_length=append_target,
                    target_min_chars=min(remain, append_target),
                    target_max_chars=append_target,
                ).get("content", "")
            except Exception:
                continuation = ""
            continuation = self._sanitize_chapter_continuation(continuation, chapter_no)
            if not continuation:
                break
            text = self._sanitize_chapter_text(text + "\n\n" + continuation, chapter_no)
            self._append_progress_event(
                "chapter_length_fix",
                f"mode=expand_to_min, words={self._measure_text(text)}",
                chapter_no,
            )

        if self._measure_text(text) > chapter_max_allowed:
            text = self._truncate_text_soft(text, chapter_max_allowed)
            self._append_progress_event(
                "chapter_length_fix",
                f"mode=final_truncate, words={self._measure_text(text)}",
                chapter_no,
            )
        if self._measure_text(text) < chapter_min_required:
            self._append_progress_event(
                "chapter_length_warning",
                f"still_below_min words={self._measure_text(text)} min={chapter_min_required}",
                chapter_no,
            )
        return self._sanitize_chapter_text(text, chapter_no)

    def _progress_log_path(self) -> Optional[str]:
        if not self.config.run_id:
            return None
        import os
        run_dir = os.path.join(self.config.runs_dir, self.config.run_id)
        os.makedirs(run_dir, exist_ok=True)
        return os.path.join(run_dir, "progress.log")

    def _append_progress_event(self, event: str, detail: str = "", chapter_no: int = 0) -> None:
        path = self._progress_log_path()
        if not path:
            return
        from datetime import datetime
        text = " ".join((detail or "").split())
        if len(text) > 500:
            text = text[:500] + "...(truncated)"
        line = f"[event] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {event}"
        if chapter_no > 0:
            line += f" | chapter {chapter_no}"
        if text:
            line += f" | {text}"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _parse_progress_kv(self, line: str) -> Dict[str, str]:
        data: Dict[str, str] = {}
        for kv in line.split(","):
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            data[k.strip()] = v.strip()
        return data

    def _write_progress_log(
        self,
        chapter_no: int,
        words: int,
        planned_total: int,
        topic: str,
        chapter_target: Optional[int] = None,
        chapter_min_required: Optional[int] = None,
        chapter_max_allowed: Optional[int] = None,
    ):
        """
        Write to progress.log, recording the chapter number, word count, and total planned chapters.
        """
        import os
        run_dir = os.path.join(self.config.runs_dir, self.config.run_id)
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "progress.log")
        # Append a record of the chapter's word count and length constraints.
        with open(path, "a", encoding="utf-8") as f:
            f.write(
                f"chapter={chapter_no}, words={words}, planned_total={planned_total}, "
                f"target={chapter_target or 0}, min={chapter_min_required or 0}, "
                f"max={chapter_max_allowed or 0}, topic={topic}\n"
            )
        mem = self.memory_system.memory_index
        self._append_progress_event(
            "memory_snapshot",
            (
                f"texts={len(mem.get('texts', []))}, outlines={len(mem.get('outlines', []))}, "
                f"characters={len(mem.get('characters', []))}, world_settings={len(mem.get('world_settings', []))}, "
                f"plot_points={len(mem.get('plot_points', []))}, fact_cards={len(mem.get('fact_cards', []))}"
            ),
            chapter_no,
        )
        total_words = self._compute_total_words()
        self._log("Progress", f"Current total words: {total_words} (chapter {chapter_no}/{planned_total})")

    def _stitch_chapters(self, topic: str) -> str:
        """
        Concatenate the final drafts (chapter_final) from memory, sorted by chapter number, to generate the full book text.
        """
        files = latest_chapter_files(Path(self.config.runs_dir) / self.config.run_id)
        if files:
            return "\n\n".join(path.read_text(encoding="utf-8").strip() for path in files)
        texts = [
            t for t in self.memory_system.memory_index.get("texts", [])
            if t.get("topic") == topic and t.get("metadata", {}).get("source") == "chapter_final"
        ]
        if not texts:
            return ""
        try:
            sorted_texts = sorted(
                texts,
                key=lambda x: int(x.get("metadata", {}).get("chapter", 0))
            )
        except Exception:
            sorted_texts = texts
        contents = []
        for t in sorted_texts:
            content = t.get("content", "")
            contents.append(content)
        return "\n\n".join(contents)

    def _summarize_progress(self) -> str:
        """
        Accumulate the word count for each chapter and return a string summary (chapter=, words=).
        """
        if not self.config.run_id:
            return ""
        import os
        path = os.path.join(self.config.runs_dir, self.config.run_id, "progress.log")
        if not os.path.exists(path):
            return ""
        chapter_lengths = {}
        lines = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                lines.append(line)
                try:
                    parts = self._parse_progress_kv(line)
                    if line.startswith("chapter=") and parts.get("chapter"):
                        chapter_lengths[int(parts["chapter"])] = int(parts.get("words", 0))
                except Exception:
                    pass
        lines.append(f"total_words={sum(chapter_lengths.values())}")
        return "\n".join(lines)

    def _compute_total_words(self) -> int:
        """
        Calculate the current cumulative word count via progress.log.
        """
        if not self.config.run_id:
            return 0
        import os
        path = os.path.join(self.config.runs_dir, self.config.run_id, "progress.log")
        if not os.path.exists(path):
            return 0
        total = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    parts = self._parse_progress_kv(line)
                    total += int(parts.get("words", 0))
                except Exception:
                    pass
        # If the log is empty or parsing fails, attempt to accumulate the length of chapter_final from memory to avoid displaying 0.
        if total == 0:
            try:
                texts = [
                    t for t in self.memory_system.memory_index.get("texts", [])
                    if t.get("topic")
                    and t.get("metadata", {}).get("source") == "chapter_final"
                ]
                for t in texts:
                    total += len(t.get("content", "") or "")
            except Exception:
                pass
        return total

    def _plan_current_chapter_by_agent(self, topic: str, chapter_no: int, genre: Optional[str], style_tags: Optional[List[str]]) -> str:
        """
        Use an Agent to plan the chapter based on global/chapter outlines, generating a brief plan (for memory and consistency).
        """
        if "plot" not in self.agents:
            return ""
        outline_text = self._get_chapter_outline_text(topic, chapter_no)
        title = self._get_chapter_outline_title(topic, chapter_no)
        global_outline = ""
        try:
            go = self.memory_system.get_outline_by_topic(topic)
            if go:
                global_outline = go.get("content", "")
        except Exception:
            global_outline = ""
        style = ", ".join(style_tags or [])
        prompt = (f"""Book: {topic}
Current chapter: {chapter_no}{(' - ' + title) if title else ''}
Global outline:
{global_outline}

Chapter outline:
{outline_text}

Without adding beyond the outlines, provide a concise writing plan (150-250 words) including:
1) Core events/turning points for this chapter;
2) Required key characters (keep names consistent with outline);
3) World/setting rules or foreshadowing to respect;
4) Ending direction (avoid spoiling later chapters).""")
        agent = self.agents["plot"]
        try:
            res = agent.generate(prompt, "", topic=topic, genre=genre, style_tags=style_tags)
            return res.get("content", "") or ""
        except Exception:
            return ""

