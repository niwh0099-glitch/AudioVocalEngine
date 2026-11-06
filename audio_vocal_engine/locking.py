"""OS-owned lock shared by every client of this engine installation."""

import os
from pathlib import Path
import time


DEFAULT_STATE_DIR = Path(__file__).resolve().parents[1] / ".state"


class EngineLock:
    def __init__(self, path):
        self.path = Path(path)
        self._file = None

    def try_acquire(self):
        if self._file is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self._file = handle
        return True

    def acquire(self, on_wait=None, poll_interval=0.25, heartbeat_interval=2.0):
        next_event = 0.0
        while not self.try_acquire():
            now = time.monotonic()
            if on_wait is not None and now >= next_event:
                on_wait()
                next_event = now + heartbeat_interval
            time.sleep(poll_interval)
        return self

    def release(self):
        if self._file is None:
            return
        handle, self._file = self._file, None
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_args):
        self.release()
