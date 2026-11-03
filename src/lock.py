"""One post at a time, across every running copy of the agent.

Two terminals could both check "is invoice-01 in the ledger?", both see "no",
and both post it. So the final check and ledger.post happen while holding this
lock: the second terminal waits, then its check sees the first one's entry.

It is an operating-system file lock on ledger.lock. The OS releases it when the
process ends, even with kill -9, so a crash can never leave it stuck.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path

LOCK_PATH = Path(__file__).resolve().parent.parent / "ledger.lock"

if os.name == "nt":  # Windows
    import msvcrt

    def _try_lock(f) -> bool:
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock(f) -> None:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)

else:  # macOS, Linux
    import fcntl

    def _try_lock(f) -> bool:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    def _unlock(f) -> None:
        fcntl.flock(f, fcntl.LOCK_UN)


class LedgerBusy(Exception):
    """Another session held the lock for too long."""


@contextmanager
def ledger_lock(timeout: float = 30.0, on_wait=None, check_stop=None):
    """Hold the posting lock. Waits up to `timeout` seconds for another session.

    on_wait() is called once if we have to wait. check_stop() is called while
    waiting and may raise to give up (nothing has been posted yet, so that is safe).
    """
    with open(LOCK_PATH, "a+") as f:
        deadline = time.monotonic() + timeout
        waited = False
        while not _try_lock(f):
            if check_stop:
                check_stop()
            if time.monotonic() > deadline:
                raise LedgerBusy("another session is still posting; try /approve again in a moment")
            if on_wait and not waited:
                on_wait()
                waited = True
            time.sleep(0.1)
        try:
            yield
        finally:
            _unlock(f)
