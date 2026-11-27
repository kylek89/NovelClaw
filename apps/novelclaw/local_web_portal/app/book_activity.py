"""Coordinate a confirmed deletion with the local workspace's requests and workers."""

from collections import Counter
from contextlib import contextmanager
from threading import Lock, Thread

from fastapi import HTTPException

_lock = Lock()
_writers = Counter()
_deleting = set()


@contextmanager
def book_activity(book_id, *, deleting=False):
    if book_id is None:
        yield
        return
    with _lock:
        if book_id in _deleting or (deleting and _writers[book_id]):
            raise HTTPException(
                409,
                "This book is busy. Let the current operation finish, then try again.",
            )
        if deleting:
            _deleting.add(book_id)
        else:
            _writers[book_id] += 1
    try:
        yield
    finally:
        with _lock:
            if deleting:
                _deleting.remove(book_id)
            else:
                _writers[book_id] -= 1
                if not _writers[book_id]:
                    del _writers[book_id]


def start_book_thread(book_id, target, item_id):
    # Reserve before starting the thread. Generation can hand off to another worker.
    reservation = book_activity(book_id)
    reservation.__enter__()

    def run():
        try:
            target(item_id)
        finally:
            reservation.__exit__(None, None, None)

    try:
        Thread(target=run, daemon=True).start()
    except BaseException:
        reservation.__exit__(None, None, None)
        raise
