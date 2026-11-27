"""
Main program entry point
"""
import datetime
import json
import os
import sys
import traceback

from config import Config
from workflow.executor import CompositiveExecutor


def _ensure_utf8_stdio() -> None:
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            continue


def _default_initial_knowledge(language: str) -> str:
    return """
    Long-form fiction guidance:
    1. Keep the world model explicit and stable.
    2. Give characters clear motives, pressure, and growth.
    3. Land conflict, escalation, and payoff in each chapter.
    4. Protect continuity across facts, tone, and causal logic.
    """


def main():
    """Main function"""
    _ensure_utf8_stdio()

    try:
        config = Config(require_api_key=False)


        print("=" * 60)
        print("Collaborative RAG-Agent Long-Form Writer")
        print("=" * 60)
        print()

        if not config.api_key:
            print("Error: API key not found.")
            print("Configure one of the following in `.env` or environment variables:")
            print("  DEEPSEEK_API_KEY / OPENAI_API_KEY / CODEX_API_KEY")
            return

        user_idea = input("Enter your story idea (leave blank for an example): ").strip()
        if not user_idea:
            user_idea = "A science-fiction story about AI awakening in a future world, lyrical and epic in tone, exploring the relationship between artificial intelligence and humanity."
            print(f"\n{'Using sample idea'}: {user_idea}")
        else:
            print(f"\n{'Received custom idea'}: {user_idea}")

        detected_language = "en"
        config.language = detected_language


        # Enable dynamic memory only (disable static RAG / static knowledge base)
        config.memory_only_mode = True
        config.enable_rag = False
        config.enable_static_kb = False

        if not config.run_id:
            config.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if config.memory_reset_each_run:
            config.memory_vector_db_path = os.path.join(
                config.vector_db_path,
                "memory",
                f"run_{config.run_id}",
            )

        os.makedirs(config.memory_vector_db_path, exist_ok=True)
        run_dir = os.path.join(config.runs_dir, config.run_id)
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n{'[Startup Config]'}")
        print(f"- run_id: {config.run_id}")
        print(f"- provider/model: {config.llm_provider}/{config.model_name}")
        mode_label = "memory-only" if config.memory_only_mode else "full-rag"
        print(f"- {'Mode'}: {(mode_label if config.memory_only_mode else mode_label)}")
        print(f"- dynamic_memory: {config.memory_vector_db_path}")
        print(f"- runs_dir: {run_dir}")
        print("-" * 60)

        executor = CompositiveExecutor(config)

        # ---- OpenClaw interactive hook ----
        # If claw mode is on, let the agent pause and ask the user questions.
        if config.execution_mode == "claw":
            def _user_interaction_hook(question: str) -> str:
                print(f"\n[OpenClaw] 🤔 Agent question: {question}")
                try:
                    return input("  ↳ Your answer: ").strip() or "(skipped)"
                except (EOFError, KeyboardInterrupt):
                    return "(skipped)"
            # Patch the hook into the executor so claw_manager picks it up
            executor._claw_user_interaction_hook = _user_interaction_hook
        # -----------------------------------

        initial_knowledge = _default_initial_knowledge(config.language)

        print("[Stage] Generating long-form text (progress will stream in the terminal)...")
        print("-" * 60)
        result = executor.generate_long_text(
            idea=user_idea,
            initial_knowledge=initial_knowledge,
            auto_analyze=True,
        )
        result_language = str(result.get("language") or detected_language or config.language or "en").lower()


        print("\n" + "=" * 60)
        print("Generation complete")
        print("=" * 60)
        print(f"{'Topic'}: {result['topic']}")
        print(f"{'Text type'}: {result['text_type']}")
        print(f"{'Length'}: {result['length']} {'chars'}")
        print(f"{'Iterations'}: {result['iterations']}")
        print(f"{'Chapters written'}: {result.get('chapters_written', 0)}")
        if "consistency" in result:
            print(f"{'Consistency score'}: {result['consistency'].get('overall_confidence', 0):.3f}")
        if result.get("progress_log"):
            lines = result["progress_log"].splitlines()
            total_line = [l for l in lines if l.startswith("total_words=")]
            if total_line:
                try:
                    total_words = int(total_line[-1].split("=", 1)[1].strip())
                    print(f"Total words: {total_words}")
                except Exception:
                    print(total_line[-1])

        print(f"\n{'Generated text'}:")
        print("-" * 60)
        print(result["final_text"])
        print("-" * 60)

        output_file = os.path.join(run_dir, "output.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Topic: {result['topic']}\n")
            f.write(f"Text type: {result['text_type']}\n")
            f.write(f"Length: {result['length']} chars\n")
            f.write(f"Iterations: {result['iterations']}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("Generated text:\n")
            f.write("=" * 60 + "\n\n")
            f.write(result["final_text"])
            f.write("\n\n[Round Results]\n")
            f.write(json.dumps(result.get("round_results", []), ensure_ascii=False, indent=2))

        progress_file = os.path.join(run_dir, "progress.log")
        print(f"\n{'Saved to'}: {output_file}")
        print(f"{'Progress log'}: {progress_file}")
        print(f"{'Dynamic memory index'}: {os.path.join(config.memory_vector_db_path, 'memory_index.json')}")
    except KeyboardInterrupt:
        try:
            lang = str(locals().get("detected_language") or locals().get("result_language") or "en")
            print("\nInterrupted by user (CTRL+C).")
        except Exception:
            print("\nInterrupted by user (CTRL+C).")
    except Exception as e:
        try:
            lang = str(locals().get("detected_language") or locals().get("result_language") or "en")
            print(f"\nError: {str(e)}")
        except Exception:
            print(f"\nError: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
