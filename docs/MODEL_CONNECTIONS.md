# Models in NovelClaw

The provider and model selectors appear in the chat composer and in **Ask Claw about this scene**. Type to search a catalog or enter an exact model ID. The star keeps a favorite for that provider in the current browser. In a saved conversation, expand **Model** to change what the next turn uses. Chapter runs inherit the chat selection at the time they are queued. Earlier conversations whose model was never recorded keep that history unknown.

## Lenovo PGX

The local installation starts new chats with **Lenovo PGX · Qwen**. Its vLLM service on `http://pgx01:8000/v1` serves `qwen3.6-35b-a3b-nvfp4`. **Lenovo PGX · Dolphin** uses `http://pgx01:8001/v1` and `dolphin-venice-24b-q6`. Docker needs network access to the Lenovo through Tailscale. **Models → Test connection** reads the selected server’s model list without generating text. An unavailable local model reports an error. Local requests use only the selected server.

Use **Add or update provider → Local server** for another OpenAI-compatible endpoint. Supply its server URL including `/v1`, a default model ID, and an optional provider-specific key. Each selector lists the models currently advertised by that service. Model loading and switching on the Lenovo remain under the author’s control.

The default host can be set with `PGX_API_HOST`. The NovelClaw app environment also supports `PGX_BASE_URL`, `PGX_MODEL`, `PGX_PROSE_BASE_URL`, and `PGX_PROSE_MODEL`. Keep profile URLs free of credentials. Save private tokens in **Models → Save API key**.

## OpenRouter

In **Models**, save an OpenRouter key for the **OpenRouter** provider. In chat or in a scene, choose **OpenRouter** and select a model, such as one found by typing its publisher name. Catalog lookup works without a key. The picker shows advertised context length, tool support, and per-million-token prices when available.

**Models → Book model activity** shows this Book’s requests, reported model identity, token usage and reported dollar cost when supplied by the provider. Missing costs are shown separately for OpenRouter. Provider billing and retries can differ from the values NovelClaw receives. No spending cap is enforced by this view.

## Agent API

Authenticated callers can fetch `GET /api/v1/providers/{provider}/models` and submit both `provider` and `model` to `POST /api/v1/stories`. Omitting the model uses the connection default when one exists. The returned story includes its pinned model. The original endpoint and protocol are retained with the job, while credentials are resolved at use time.

## Verification

The test suite covers saved model selections, local operation with no cloud key, cross-Book isolation, offline handling, reported usage, and an OpenRouter-compatible recording server. Run `python scripts/browser-smoke.py --models --books --scenes --channel chrome` for the disposable browser journey. The opt-in `NOVELCLAW_TEST_PGX=1` cases in `tests/test_model_connections.py` exercise the actual Lenovo hardware with disposable Books.

On November 27, 2026, both Lenovo models returned Daniel’s morning height of 5′11″ with a later 4′2″ change present. Qwen completed a disposable 153-word chapter in 24 local requests and saved its prose, outline and memory. The live OpenRouter catalog and a simulated hosted generation were verified. No paid OpenRouter generation was run.
