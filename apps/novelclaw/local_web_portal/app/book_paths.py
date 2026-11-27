from pathlib import Path
import re


def book_memory_dir(config, key):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", key):
        raise ValueError("Invalid book memory identifier")
    app_dir = Path(__file__).resolve().parents[2]
    return (app_dir / Path(config.vector_db_path) / "memory" / f"run_{key}").resolve()
