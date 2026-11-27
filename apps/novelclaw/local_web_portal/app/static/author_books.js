(() => {
  const root = document.getElementById("author-book");
  const status = document.getElementById("book-status");
  const changed = new Set();
  let sending = false;
  let leaving = false;
  root.addEventListener("input", event => {
    const form = event.target.form;
    if (form && (form.hasAttribute("data-edit") || form.hasAttribute("data-ask") || form.hasAttribute("data-new"))) changed.add(form);
  });
  window.addEventListener("beforeunload", event => {
    if (!leaving && (changed.size || sending)) {event.preventDefault();event.returnValue = "";}
  });
  async function post(path, data) {
    const response = await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
    const payload = await response.json();
    if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "Check the fields and try again.");
    return payload;
  }
  root.addEventListener("submit", async event => {
    const form = event.target;
    if (!["edit","ask","new","restore","delete"].some(k => Object.hasOwn(form.dataset,k))) return;
    event.preventDefault();
    if (sending) return;
    if ([...changed].some(other => other !== form)) {
      status.textContent = "Save the other open edit on this page first."; return;
    }
    if (form.hasAttribute("data-delete")) {
      if (!window.confirm('Permanently delete “' + form.dataset.title + '” and all its chapters, cast and chats?')) return;
      sending = true;
      try {
        await post(root.dataset.api + "/" + form.dataset.book + "/delete", {
          revision: Number(form.dataset.revision), confirm: true
        });
        form.closest(".ab-library-row").remove();
        status.textContent = "Book deleted.";
        if (!root.querySelector(".ab-library-row")) location.reload();
      } catch (error) {
        status.textContent = error.message;
      } finally {
        sending = false;
      }
      return;
    }
    const data = new FormData(form);
    const values = Object.fromEntries(data);
    const target = new URL(location.href);
    sending = true;
    root.inert = true;
    const button = event.submitter;
    if (button) button.disabled = true;
    try {
      const api = root.dataset.api;
      const bookApi = api+"/"+root.dataset.id;
      if (form.hasAttribute("data-new") || form.hasAttribute("data-restore")) {
        const restoring = form.hasAttribute("data-restore");
        let payload=values;
        if (restoring) {
          const file=form.querySelector("input[type=file]").files[0];
          if (!file || file.size>21_000_000) throw new Error("Choose a Book backup smaller than 21 MB.");
          payload=JSON.parse(await file.text());
        }
        const result=await post(api+(restoring ? "/restore" : ""),payload);
        target.pathname=root.dataset.library+"/"+result.id;target.search="";
      } else {
        values.revision=Number(root.dataset.revision);
        if (form.hasAttribute("data-ask")) {
          values.token=form.dataset.token || crypto.randomUUID().replaceAll("-","");
          form.dataset.token=values.token;
          values.replace=data.has("replace");
          status.textContent="Claw is writing…";
          const result=await post(bookApi+"/ask",values);
          if(result.status==="started") throw new Error("This request was already started. Reload to check its saved chat before retrying.");
          if(result.status==="provider_error") throw new Error("This request stopped at the provider. Reload before sending another request.");
          target.searchParams.set("lane",values.lane);
          if(values.mode==="discover" || result.status==="format_error") {
            target.searchParams.set("view","chat");target.searchParams.set("chat",values.chat);
            if(result.status==="format_error") target.searchParams.set("notice","format_error");
          }
          if(result.status==="chat_only") {
            target.searchParams.set("view","chat");target.searchParams.set("chat",values.chat);
            target.searchParams.set("notice","chat_only");
          }
        } else {
          values.action=form.dataset.edit;
          if(button?.hasAttribute("data-split")) {
            values.action="split";
            const field=form.elements.namedItem("text");
            values.offset=Array.from(field.value.slice(0,field.selectionStart)).length;
          }
          if(data.has("source")) [values.chapter,values.scene]=values.source.split("/");
          if(values.action==="save_candidates") {
            values.picks=[...form.querySelectorAll("[data-pick]")].filter(el=>el.querySelector("input[type=checkbox]").checked)
                .map(el=>({index:Number(el.dataset.pick),name:el.querySelector("[data-name]").value,profile:el.querySelector("[data-profile]").value}));
            if(!values.picks.length) throw new Error("Select at least one character to save.");
          }
          if(values.action==="profile_capture") values.selected=data.getAll("selected").map(Number);
          const result=await post(bookApi+"/edit",values);
          for(const key of ["chat","character","scene","chapter"]) if (result[key]) target.searchParams.set(key,result[key]);
          if(form.dataset.next) target.searchParams.set("view",form.dataset.next);
        }
      }
      leaving=true;location.assign(target.href);
    } catch(error) {
      root.inert=false;
      status.textContent=error.message; status.scrollIntoView({block:"nearest"});
      sending=false;if(button) button.disabled=false;
    }
  });
  if(new URL(location.href).searchParams.get("notice")==="format_error") status.textContent="Claw returned an incomplete proposal. Its reply is saved here for review.";
  if(new URL(location.href).searchParams.get("notice")==="chat_only") status.textContent="The Book changed during generation. Claw saved its reply in this chat.";
})();
