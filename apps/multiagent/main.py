"""
Main program entry point
"""
import datetime
import json
import os
import traceback

from config import Config
from workflow.executor import CompositiveExecutor


def main():
    """Main function"""
    print("=" * 60)
    print("Collaborative RAG-Agent long-form writer")
    print("=" * 60)
    print()

    try:
        config = Config(require_api_key=False)

        if not config.api_key:
            print("Missing API key.")
            print("Configure one of the following in .env or in your environment:")
            print("  DEEPSEEK_API_KEY / OPENAI_API_KEY / CODEX_API_KEY")
            return

        user_idea = input("Enter your story idea (leave blank for an example): ").strip()
        if not user_idea:
            user_idea = "A science fiction story about an AI awakening and its relationship with people in a future world"
            print(f"\nUsing example idea: {user_idea}")
        else:
            print(f"\nReceived idea: {user_idea}")

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

        print("\n[Startup settings]")
        print(f"- run_id: {config.run_id}")
        print(f"- provider/model: {config.llm_provider}/{config.model_name}")
        print(f"- mode: {'memory-only' if config.memory_only_mode else 'full-rag'}")
        print(f"- dynamic_memory: {config.memory_vector_db_path}")
        print(f"- runs_dir: {run_dir}")
        print("-" * 60)

        executor = CompositiveExecutor(config)

        initial_knowledge = """
        Science fiction writing guidance:
        1. Establish the world's technology, society, and historical context
        2. Give each character depth and a clear motive
        3. Build conflict and give it a payoff
        4. Keep events logically consistent
        """

        print("[Generation] Writing (progress appears in the terminal)...")
        print("-" * 60)
        result = executor.generate_long_text(
            idea=user_idea,
            initial_knowledge=initial_knowledge,
            auto_analyze=True,
        )

        print("\n" + "=" * 60)
        print("Generation complete")
        print("=" * 60)
        print(f"Topic: {result['topic']}")
        print(f"Text type: {result['text_type']}")
        print(f"Length: {result['length']} characters")
        print(f"Iterations: {result['iterations']}")
        print(f"Chapters written: {result.get('chapters_written', 0)}")
        if "consistency" in result:
            print(f"Consistency score: {result['consistency'].get('overall_confidence', 0):.3f}")
        if result.get("progress_log"):
            lines = result["progress_log"].splitlines()
            total_line = [l for l in lines if l.startswith("total_words=")]
            if total_line:
                print(total_line[-1])

        print("\nGenerated text:")
        print("-" * 60)
        print(result["final_text"])
        print("-" * 60)

        output_file = os.path.join(run_dir, "output.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Topic: {result['topic']}\n")
            f.write(f"Text type: {result['text_type']}\n")
            f.write(f"Length: {result['length']} characters\n")
            f.write(f"Iterations: {result['iterations']}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("Generated text:\n")
            f.write("=" * 60 + "\n\n")
            f.write(result["final_text"])
            f.write("\n\n[Round summaries]\n")
            f.write(json.dumps(result.get("round_results", []), ensure_ascii=False, indent=2))

        progress_file = os.path.join(run_dir, "progress.log")
        print(f"\nSaved to: {output_file}")
        print(f"Progress log: {progress_file}")
        print(f"Memory index: {os.path.join(config.memory_vector_db_path, 'memory_index.json')}")
    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C).")
    except Exception as e:
        print(f"\nError: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
