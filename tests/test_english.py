"""English application output and read compatibility for pre-English Books."""
import copy
import importlib
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_repository_text_has_no_han_characters():
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        pytest.skip("The source file inventory is available in a Git checkout")
    inventory = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    matches = []
    for raw in inventory.decode("utf-8").split("\0"):
        path = root / raw
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        for number, line in enumerate(data.decode("utf-8-sig").splitlines(), 1):
            if re.search(r"[\u3400-\u9fff\uf900-\ufaff\U00020000-\U000323af]", line):
                matches.append(f"{raw}:{number}")
    assert not matches, matches


def test_chapter_review_sends_english_labels_and_source_details(workspace):
    executor_type = importlib.import_module("workflow.executor").CompositiveExecutor
    checker_type = importlib.import_module("rag.consistency_checker").ConsistencyChecker
    executor = executor_type.__new__(executor_type)
    executor.main_characters = ["Daniel Turner"]
    executor.memory_system = SimpleNamespace(
        get_recent_outlines=lambda *args, **kwargs: [{"content": "Daniel found the jar."}]
    )
    executor._get_chapter_outline_text = lambda *args: "Return the jar."
    executor._get_chapter_outline_title = lambda *args: "The return"
    executor._get_last_chapter_summary = lambda *args: "He found a key."
    executor._get_recent_fact_cards = lambda *args, **kwargs: ["Height at dawn: 5 feet 11 inches."]
    executor._get_recent_world_settings = lambda *args, **kwargs: ["One wish per day."]
    client = Mock()
    client.chat.return_value = '{"consistent": true, "confidence": 1}'
    checker = checker_type(SimpleNamespace(), client, executor.memory_system)
    baseline = executor._build_consistency_baseline("Test Book", 2)
    checker.check_plot_consistency("Daniel returns the jar.", "Test Book", baseline)
    messages = client.chat.call_args.args[0]
    prompt = messages[-1]["content"]
    for label, value in [
        ("Rolling summary", "Daniel found the jar."),
        ("Chapter outline", "Return the jar."),
        ("Chapter title", "The return"),
        ("Main characters", "Daniel Turner"),
        ("Previous chapter summary", "He found a key."),
        ("Recent fact cards", "Height at dawn: 5 feet 11 inches."),
        ("World setting", "One wish per day."),
    ]:
        assert f"[{label}]" in prompt and value in prompt
    assert "Daniel returns the jar." in prompt
    assert all(message["content"].isascii() for message in messages)


def test_older_book_field_names_are_read_without_editing_the_book(workspace):
    if workspace.name != "novelclaw":
        pytest.skip("NovelClaw character and world cards")
    # Simulate the serialized field identifiers used by early Book files.
    assets = {
        "characters": [{
            "attributes": {"\u59d3\u540d": "Daniel Turner", "\u804c\u4e1a": "Keeper"},
            "character_info": "\u76ee\u6807\uff1aReturn the jar.",
        }],
        "world_settings": [{"setting_info": "\u540d\u79f0\uff1aRainfall\n\u89c4\u5219\uff1aOne wish."}],
    }
    original = copy.deepcopy(assets)
    character = workspace.main._build_character_cards(assets, "en")[0]
    assert character["name"] == "Daniel Turner"
    assert {"label": "Role", "value": "Keeper"} in character["fields"]
    assert {"label": "Goal", "value": "Return the jar."} in character["fields"]
    world = workspace.main._build_world_cards(assets, "en")["settings"][0]
    assert world["title"] == "Rainfall"
    assert {"label": "Rule", "value": "One wish."} in world["fields"]
    assert character["raw_text"] == original["characters"][0]["character_info"]
    assert assets == original
