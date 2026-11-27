from uuid import uuid4
from urllib.parse import urlsplit

import pytest


def test_daniel_scene_revisions_and_isolation(signed_in):
    a = signed_in
    if a.name != "novelclaw":
        pytest.skip("NovelClaw chapter editor")
    paths = []
    for name in ("Scene fixture", "Other book"):
        res = a.client.post("/books", data={"title": name}, follow_redirects=False)
        paths.append(
            urlsplit(res.headers["location"]).path.removesuffix("/console/chat")
        )
    read = paths[0] + "/console/manuscript/read"
    res = a.client.post(paths[0] + "/console/manuscript/new-chapter", data={"after": 0})
    assert res.status_code == 200, res.text
    run = res.context["scene_run"]
    old = res.context["scene_doc"]
    api = paths[0] + f"/api/runs/{run}/chapters/1/scenes"

    def scene(text, time, value=None, phase="entry", day=1):
        facts = (
            [{"character": "Daniel", "trait": "Height", "value": value, "phase": phase}]
            if value
            else []
        )
        return {
            "id": uuid4().hex,
            "text": text,
            "day": day,
            "time": time,
            "facts": facts,
        }

    morning = scene("Morning 🫙\n\n", "09:00", "5′11″")
    after = scene("Transformation.\n\n", "16:00", "4′2″", "change")
    later = scene("Evening.", "18:00")
    doc = {**old, "scenes": [morning, after, later]}
    saved = a.client.put(api, json=doc)
    assert saved.status_code == 200, saved.text
    rev = saved.json()["revision"]
    assert a.client.put(api, json=doc).status_code == 409
    assert (
        a.client.put(
            api, json={"revision": rev, "scenes": [morning, morning]}
        ).status_code
        == 422
    )
    assert (
        a.client.put(
            api, json={"revision": rev, "scenes": [{**morning, "time": "29:19"}]}
        ).status_code
        == 422
    )
    assert (
        a.client.get(
            paths[1] + "/console/manuscript/read", params={"manuscript_run": run}
        ).status_code
        == 404
    )
    assert (
        a.client.post(
            api.removesuffix("/scenes") + "/content",
            json={"content": "Stale plain editor"},
        ).status_code
        == 409
    )
    assert a.client.put(api.replace(paths[0], paths[1]), json=doc).status_code == 404

    def context(item):
        res = a.client.get(
            api + "/" + item["id"] + "/context", params={"revision": rev}
        )
        assert res.status_code == 200, res.text
        return res.json()["context"]

    assert "4′2″" not in context(morning) and "5′11″" in context(morning)
    assert "Evening." not in context(morning)
    assert "5′11″" in context(after) and "4′2″" in context(after)
    assert "4′2″" in context(later) and "5′11″" not in context(later)
    assert a.client.get(read).context["workspace_bootstrap"]["chapters"][0][
        "content"
    ] == "".join(s["text"] for s in doc["scenes"])
    second = a.client.post(
        paths[0] + "/console/manuscript/new-chapter", data={"run": run, "after": 1}
    )
    assert second.status_code == 200
    flashback = scene("Flashback.", "10:00")
    api2 = api.replace("/chapters/1/", "/chapters/2/")
    newer = a.client.put(
        api2, json={**second.context["scene_doc"], "scenes": [flashback]}
    )
    result = a.client.get(
        api2 + "/" + flashback["id"] + "/context", params=newer.json()
    )
    assert (
        "5′11″" in result.json()["context"] and "4′2″" not in result.json()["context"]
    )
    assert (
        a.client.post(
            paths[0] + "/console/manuscript/new-chapter", data={"run": run, "after": 1}
        ).status_code
        == 409
    )


def test_interrupted_scene_save_preserves_old_boundary_map(
    workspace, tmp_path, monkeypatch
):
    if workspace.name != "novelclaw":
        pytest.skip("NovelClaw chapter editor")
    mod = workspace.main.scene_manuscript
    path = tmp_path / "chapter_01_iter_1_final.txt"
    path.write_text("Before", encoding="utf8")
    first = mod.document(path)
    first["scenes"][0]["day"] = 1
    rev = mod.save(path, mod.ChapterEdit(**first))["revision"]
    draft = mod.document(path)
    draft["scenes"][0].update(text="After", day=2)
    write = mod.atomic_write_text

    def cut(target, text):
        if target == path:
            raise OSError("Simulated interrupted text replacement")
        write(target, text)

    monkeypatch.setattr(mod, "atomic_write_text", cut)
    with pytest.raises(OSError):
        mod.save(path, mod.ChapterEdit(**draft))
    assert mod.document(path)["revision"] == rev
    assert mod.document(path)["scenes"] == first["scenes"]
