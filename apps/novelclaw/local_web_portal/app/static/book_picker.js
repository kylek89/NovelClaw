(() => {
  const picker = document.getElementById("active-book");
  if (!picker) return;
  const initial = picker.value;
  const dirty = new Set();
  const fields = [...document.querySelectorAll("main textarea")];
  const chat = document.querySelector("[data-idea-copilot-root]");
  const key = "novelclaw.switch-draft:" + window.__APP_BASE_PATH__ + ":" + (chat?.dataset.sessionId || "new");
  // Cache only an unsent chat composer. Other editors may refer to a selected
  // record within the page, so keep the author on that record until they save.
  const composer = document.querySelector(".chat-input-form textarea");
  if (composer) {
    try {
      const draft = sessionStorage.getItem(key);
      if (draft && !composer.value) {
        composer.value = draft;
        composer.dispatchEvent(new Event("input", { bubbles: true }));
      }
      sessionStorage.removeItem(key);
    } catch (_) {}
  }
  for (const field of fields) field.addEventListener("input", () => dirty.add(field));
  function preserve() {
    if ([...dirty].some(field => field !== composer && field.isConnected) &&
        !window.confirm("Leave this page and discard any unsaved editor changes?")) return false;
    if (composer && composer.value) {
      try { sessionStorage.setItem(key, composer.value); }
      catch (_) { return window.confirm("Your chat draft could not be saved. Switch books anyway?"); }
    }
    return true;
  }
  picker.addEventListener("change", () => {
    if (picker.value && preserve()) location.assign(picker.value);
    else picker.value = initial;
  });
  document.querySelector(".book-picker form")?.addEventListener("submit", e => {
    if (!preserve()) e.preventDefault();
  });
})();
