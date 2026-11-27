"""Observe completed transport attempts without retaining prompts or credentials."""

from contextvars import ContextVar

listener = ContextVar("model_usage_listener", default=None)


def observe(provider, requested, response=None, error=None):
    record = listener.get()
    if record is None:
        return
    usage = getattr(response, "usage", None)
    cost = getattr(usage, "cost", None)
    record(
        dict(
            provider=provider,
            requested=requested,
            reported=getattr(response, "model", None),
            upstream=getattr(response, "provider", None),
            input_tokens=getattr(
                usage, "prompt_tokens", getattr(usage, "input_tokens", None)
            ),
            output_tokens=getattr(
                usage, "completion_tokens", getattr(usage, "output_tokens", None)
            ),
            cost_usd=str(cost) if cost is not None else None,
            outcome="received" if response is not None else "error",
        )
    )
