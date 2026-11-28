(() => {
  const el = id => document.getElementById(id);
  const root = el("scene-editor");
  const doc = JSON.parse(el("scene-data").textContent);
  const scenes = doc.scenes;
  const prose = el("chapter-prose");
  const jump = el("scene-jump");
  const status = el("scene-status");
  const context = el("scene-context");
  const answer = el("scene-answer");
  const endpoint = window.__appPath("/api/runs/" + encodeURIComponent(root.dataset.run) + "/chapters/" + root.dataset.chapter + "/scenes");
  let divider = "\n──────── Scene break ────────\n";
  while (scenes.some(s => s.text.includes(divider))) divider = divider.replace("── Scene", "─── Scene");
  let active = 0, dirty = false, saving = false, editCount = 0, previewCount = 0;
  const replies = new Map();
  const questions = new Map();
  const candidates = new Map();
  const selected = () => scenes[active];

  function notice(message, error = false) {
    status.textContent = message;
    status.dataset.error = String(error);
  }
  function modified() {
    dirty = true;
    candidates.clear(); el("use-scene").hidden = true;
    el("entry-state").textContent = "Save chapter to update the on-entry state.";
    editCount++;
    previewCount++;
    context.textContent = "Save chapter to refresh the scene context.";
    notice("Unsaved changes");
  }
  async function request(url, options) {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) {
      const detail = data.detail;
      throw new Error(typeof detail === "string" ? detail : detail?.[0]?.msg || "The request failed.");
    }
    return data;
  }
  async function preview() {
    if (dirty) return;
    const ticket = ++previewCount;
    context.textContent = "Loading scene context…";
    el("entry-state").textContent = "";
    try {
      const result = await request(endpoint + "/" + selected().id + "/context?revision=" + doc.revision);
      if (ticket === previewCount) {
        context.textContent = result.context;
        el("entry-state").textContent = "On entry · " + result.on_entry;
      }
    } catch (error) {
      if (ticket === previewCount) el("entry-state").textContent = context.textContent = error.message;
    }
  }
  function renderOptions() {
    jump.replaceChildren(...scenes.map((scene, i) => {
      const when = [scene.day && "Day " + scene.day, scene.time].filter(Boolean).join(" · ");
      return new Option("Scene " + (i + 1) + (when ? " · " + when : ""), String(i));
    }));
    jump.value = String(active);
  }
  function renderFacts() {
    const rows = el("state-rows");
    rows.replaceChildren();
    for (const [index, fact] of selected().facts.entries()) {
      const row = document.createElement("div");
      row.className = "scene-fact";
      for (const [key, title] of [["character","Character"],["trait","Trait"],["value","Value"],["phase","Applies"]]) {
        const label = document.createElement("label");
        label.textContent = title;
        const input = document.createElement(key === "phase" ? "select" : "input");
        if (key === "phase") input.append(new Option("On entry","entry"), new Option("Changes during scene","change"));
        else { input.maxLength = key === "value" ? 1200 : 120; input.required = true; }
        if (key === "character") input.setAttribute("list", "book-character-names");
        input.value = fact[key];
        input.addEventListener("input", () => { fact[key] = input.value; modified(); });
        label.append(input);
        row.append(label);
      }
      const privateLabel = document.createElement("label");
      const check = document.createElement("input");
      check.type = "checkbox"; check.checked = !!fact.intimate;
      check.addEventListener("change", () => { fact.intimate = check.checked; modified(); });
      privateLabel.append(check, "Intimate fact"); row.append(privateLabel);
      const remove = document.createElement("button");
      remove.type = "button"; remove.className = "ws-btn"; remove.textContent = "Remove";
      remove.addEventListener("click", () => { selected().facts.splice(index, 1); modified(); renderFacts(); });
      row.append(remove);
      rows.append(row);
    }
  }
  function refreshIntimate() {
    const level = selected().prose_level || el("scene-level").dataset.default;
    const enabled = ["sensual", "explicit"].includes(level);
    el("scene-private").disabled = !enabled;
    el("scene-private").checked = enabled && !!selected().include_intimate;
  }
  function renderScene() {
    renderOptions();
    el("state-title").textContent = "Character state · Scene " + (active + 1);
    el("scene-day").value = selected().day ?? "";
    el("scene-time").value = selected().time;
    el("scene-pov").value = selected().pov || "";
    el("scene-cast").value = selected().participants || "";
    el("scene-level").value = selected().prose_level || "";
    refreshIntimate();
    el("use-scene").hidden = !candidates.has(selected().id);
    answer.textContent = replies.get(selected().id) || "";
    el("scene-question").value = questions.get(selected().id) || "";
    renderFacts();
    if (dirty) context.textContent = "Save chapter to refresh the scene context.";
    else preview();
  }
  function startOf(index) {
    return scenes.slice(0, index).reduce((n, s) => n + s.text.length + divider.length, 0);
  }
  function locate(position) {
    let start = 0;
    for (let i = 0; i < scenes.length; i++) {
      const end = start + scenes[i].text.length;
      if (position < end + divider.length || i === scenes.length - 1) return { index:i, start, end };
      start = end + divider.length;
    }
  }
  function trackCursor() {
    const next = locate(prose.selectionStart).index;
    if (next !== active) { active = next; renderScene(); }
  }
  function goToScene(index) {
    active = index;
    prose.focus();
    const offset = startOf(index);
    prose.setSelectionRange(offset, offset);
    renderScene();
  }
  function assemble() { return scenes.map(s => s.text).join(divider); }
  prose.value = assemble();
  prose.addEventListener("input", () => {
    const parts = prose.value.split(divider);
    if (parts.length !== scenes.length) {
      const pos = prose.selectionStart;
      prose.value = assemble();
      prose.setSelectionRange(pos, pos);
      notice("Keep scene dividers intact when editing across scenes.", true);
      return;
    }
    parts.forEach((part, i) => { scenes[i].text = part; });
    modified();
    trackCursor();
  });
  for (const event of ["click", "keyup", "select"]) prose.addEventListener(event, trackCursor);
  jump.addEventListener("change", () => goToScene(Number(jump.value)));
  el("scene-split").addEventListener("click", () => {
    const pos = prose.selectionStart;
    const { index, start, end } = locate(pos);
    if (pos !== prose.selectionEnd || pos > end) { notice("Place the cursor in the prose where the new scene starts.", true); return; }
    const source = scenes[index];
    const id = Array.from(crypto.getRandomValues(new Uint8Array(16)), n => n.toString(16).padStart(2,"0")).join("");
    const offset = pos - start;
    const next = {id, text:source.text.slice(offset), day:source.day, time:"", facts:[]};
    source.text = source.text.slice(0, offset);
    // Preserve paragraph separation in the plain manuscript, including a split
    // at the end where the author will type the next scene later.
    if (source.text && !source.text.endsWith("\n\n") && !next.text.startsWith("\n\n")) {
      const existingBreaks = (source.text.match(/\n+$/)?.[0].length || 0) + (next.text.match(/^\n+/)?.[0].length || 0);
      source.text += "\n".repeat(Math.max(0, 2 - existingBreaks));
    }
    scenes.splice(index + 1, 0, next);
    prose.value = assemble(); modified(); goToScene(index + 1);
  });
  el("scene-day").addEventListener("input", e => { selected().day = e.target.value === "" ? null : Number(e.target.value); modified(); renderOptions(); });
  el("scene-time").addEventListener("input", e => { selected().time = e.target.value; modified(); renderOptions(); });
  el("scene-pov").addEventListener("input", e => { selected().pov = e.target.value; modified(); });
  el("scene-cast").addEventListener("input", e => { selected().participants = e.target.value; modified(); });
  el("scene-level").addEventListener("change", e => {
    selected().prose_level = e.target.value || null;
    refreshIntimate();
    if (el("scene-private").disabled) selected().include_intimate = false;
    modified();
  });
  el("scene-private").addEventListener("change", e => { selected().include_intimate = e.target.checked; modified(); });
  el("scene-action").addEventListener("change", e => {
    el("scene-question").required = e.target.value === "ask";
    el("scene-send").textContent = e.target.value === "ask" ? "Ask Claw" : e.target.selectedOptions[0].textContent;
  });
  el("state-add").addEventListener("click", () => {
    selected().facts.push({character:"",trait:"",value:"",phase:"entry"}); modified(); renderFacts();
    el("state-rows").lastElementChild.querySelector("input").focus();
  });
  async function save() {
    if (saving) return false;
    if (!dirty) return true;
    for (let i = 0; i < scenes.length; i++) {
      const s = scenes[i];
      if (s.day !== null && (!Number.isInteger(s.day) || s.day < 1) ||
          s.facts.some(f => !f.character.trim() || !f.trait.trim() || !f.value.trim())) {
        goToScene(i); notice("Enter a positive story day and fill in each character state row, or remove the empty row.", true); return false;
      }
    }
    const savedCount = editCount;
    saving = true;
    el("chapter-save").disabled = true;
    try {
      const result = await request(endpoint, {method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify(doc)});
      doc.revision = result.revision;
      dirty = savedCount !== editCount;
      notice(dirty ? "More changes to save" : "Chapter saved");
      if (!dirty) preview();
      return !dirty;
    } catch (error) { notice(error.message, true); return false; }
    finally { saving = false; el("chapter-save").disabled = false; }
  }
  el("chapter-save").addEventListener("click", save);
  el("scene-preview").addEventListener("toggle", preview);
  el("scene-question").addEventListener("input", e => questions.set(selected().id, e.target.value));
  el("ask-form").addEventListener("submit", async e => {
    e.preventDefault();
    const id = selected().id, question = el("scene-question").value, provider = el("scene-provider").value, model = e.currentTarget.elements.model.value;
    const action = el("scene-action").value;
    const button = el("scene-send");
    button.disabled = true;
    try {
      if (!await save() || id !== selected().id) return;
      const version = editCount;
      candidates.delete(id); el("use-scene").hidden = true;
      replies.set(id, "Asking Claw…"); answer.textContent = replies.get(id);
      const result = await request(endpoint + "/" + id + "/ask", {method:"POST",body:new URLSearchParams({revision:doc.revision, question, provider, model, action})});
      replies.set(id, result.reply);
      if (["draft", "revise"].includes(action) && version === editCount && result.reply.trim()) {
        candidates.set(id, result.reply);
        if (selected().id === id) el("use-scene").hidden = false;
      }
    } catch (error) { replies.set(id, error.message); }
    finally { if (id === selected().id) answer.textContent = replies.get(id) || ""; button.disabled = false; }
  });
  el("use-scene").addEventListener("click", () => {
    const candidate = candidates.get(selected().id);
    if (!candidate) return;
    selected().text = candidate.trim() + (active + 1 < scenes.length ? "\n\n" : "");
    prose.value = assemble(); modified(); goToScene(active);
    notice("Scene prose updated. Save chapter to keep it.");
  });
  el("chapter-jump").addEventListener("change", async e => {
    const chapter = e.target.value;
    e.target.value = root.dataset.chapter;
    if (await save()) location.href = window.__appPath("/console/manuscript/edit/" + encodeURIComponent(root.dataset.run) + "/" + chapter);
  });
  window.addEventListener("beforeunload", e => { if (dirty) { e.preventDefault(); e.returnValue = ""; } });
  renderScene();
})();

