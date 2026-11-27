for (const picker of document.querySelectorAll("[data-model-picker]")) {
  const provider = picker.querySelector("select");
  const model = picker.querySelector("input");
  const options = picker.querySelector("datalist");
  const status = picker.querySelector("[data-model-info]");
  const favorite = picker.querySelector("[data-favorite]");
  let catalog = [], message = "", epoch = 0;
  const storageKey = () => "novelclaw.favorites." + picker.dataset.user + "." + provider.value;
  function favorites() {
    try { return JSON.parse(localStorage.getItem(storageKey()) || "[]"); }
    catch { return []; }
  }
  function show() {
    const row = catalog.find(item => item.id === model.value);
    const parts = [message];
    if (row?.context) parts.push(Number(row.context).toLocaleString() + " context tokens");
    if (row?.input_price != null && row?.output_price != null)
      parts.push("$" + row.input_price + " input / $" + row.output_price + " output per million tokens");
    if (row?.tools != null) parts.push(row.tools ? "Tool calling advertised" : "Text only");
    status.textContent = parts.filter(Boolean).join(" · ");
    const active = favorites().includes(model.value);
    favorite.textContent = active ? "★" : "☆";
    favorite.setAttribute("aria-pressed", String(active));
    favorite.disabled = !model.value.trim();
  }
  function renderList() {
    const items = new Set([...favorites(), ...catalog.map(item => item.id)]);
    options.replaceChildren(...Array.from(items, id => new Option(id, id)));
    show();
  }
  async function load(force = false) {
    const ticket = ++epoch;
    catalog = [];
    message = "";
    renderList();
    const kind = provider.selectedOptions[0]?.dataset.kind;
    if (!force && !["local", "openrouter"].includes(kind)) return;
    message = "Checking model list…"; show();
    try {
      const url = window.__appPath("/providers/" + encodeURIComponent(provider.value) + "/models");
      const response = await fetch(url + (force ? "?refresh=true" : ""));
      const data = await response.json();
      if (ticket !== epoch) return;
      if (!response.ok) throw new Error(data.detail || "Could not reach the model server.");
      catalog = data.models;
      message = catalog.length ? (force ? "Connected · " : "") + catalog.length + (catalog.length === 1 ? " model available" : " models available") : "Connected · no models served";
      renderList();
    } catch (err) {
      if (ticket === epoch) { message = err.message; show(); }
    }
  }
  provider.addEventListener("change", () => {
    model.value = provider.selectedOptions[0]?.dataset.defaultModel || "";
    load();
  });
  model.addEventListener("input", show);
  favorite.addEventListener("click", () => {
    const id = model.value.trim();
    const saved = favorites();
    const next = saved.includes(id) ? saved.filter(v => v !== id) : [id,...saved];
    try { localStorage.setItem(storageKey(), JSON.stringify(next)); }
    catch { message = "Favorites need browser storage."; }
    renderList();
  });
  picker.querySelector("[data-refresh]")?.addEventListener("click", () => load(true));
  load();
}
