from contextlib import contextmanager
from decimal import Decimal
import logging
from sqlalchemy import select
from utils.model_usage import listener
from .db import SessionLocal
from .models import ModelCall


@contextmanager
def recording(book_id, operation):
    def save(fields):
        if book_id is None:
            return
        try:
            with SessionLocal.begin() as db:
                db.add(ModelCall(book_id=book_id, operation=operation, **fields))
        except Exception:
            logging.getLogger(__name__).exception("Could not save model usage")

    token = listener.set(save)
    try:
        yield
    finally:
        listener.reset(token)


def book_usage(db, book_id):
    if not book_id:
        return {}
    calls = list(
        db.scalars(
            select(ModelCall)
            .where(ModelCall.book_id == book_id)
            .order_by(ModelCall.id.desc())
        )
    )
    costs = [Decimal(c.cost_usd) for c in calls if c.cost_usd is not None]
    return {
        "calls": len(calls),
        "cost": str(sum(costs)) if costs else None,
        "unreported": sum(
            c.provider == "openrouter" and c.cost_usd is None for c in calls
        ),
        "recent": calls[:12],
    }
