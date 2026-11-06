"""Short private native workspaces and strictly owned stale-work cleanup."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from .locking import DEFAULT_STATE_DIR


ENGINE_ROOT = Path(__file__).resolve().parents[1]


def filesystem_path(path):
    """Use Windows extended paths only at standard-library IO boundaries."""
    value = os.path.abspath(str(path))
    if os.name != "nt" or value.startswith("\\\\?\\"):
        return value
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value[2:]
    return "\\\\?\\" + value


def process_identity(pid):
    """Return an OS creation identity, None for an exited PID; errors stay errors."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            error = ctypes.get_last_error()
            if error in (87, 1168):  # invalid / exited PID
                return None
            raise OSError(error, "Cannot safely identify the workspace owner process.")
        try:
            exit_code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                raise ctypes.WinError(ctypes.get_last_error())
            if exit_code.value != 259:  # STILL_ACTIVE
                return None
            times = [wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in times)):
                raise ctypes.WinError(ctypes.get_last_error())
            return f"windows:{(times[0].dwHighDateTime << 32) | times[0].dwLowDateTime}"
        finally:
            kernel.CloseHandle(handle)
    try:
        # Linux start ticks + boot identity distinguish a reused PID.
        fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return f"linux:{boot}:{fields[19]}"
    except FileNotFoundError:
        return None


def _is_reparse(path):
    try:
        stat = path.lstat()
    except FileNotFoundError:
        return False
    return path.is_symlink() or bool(getattr(stat, "st_file_attributes", 0) & 0x400)


def _work_root(state_dir):
    state = Path(state_dir or DEFAULT_STATE_DIR)
    intended = state / "work"
    root = intended.resolve()
    if _is_reparse(state) or _is_reparse(intended) or not root.is_relative_to(state.resolve()):
        raise ValueError("Private workspace root must remain inside the engine state directory without reparse links.")
    return root


def _remove_owned_directory(directory, root):
    directory, root = Path(directory), Path(root).resolve()
    resolved = directory.resolve()
    if resolved.parent != root or _is_reparse(directory):
        raise ValueError("Refusing to remove a workspace outside its exact private root.")
    if not directory.exists():
        return
    # Native output should contain only regular files/directories. Preserve
    # unexpected links rather than traversing anything with unclear ownership.
    if any(_is_reparse(path) for path in directory.rglob("*")):
        raise ValueError("Refusing to remove a private workspace containing reparse links.")
    shutil.rmtree(filesystem_path(resolved))


def cleanup_work(job_id=None, state_dir=None):
    """Remove only registered private work belonging to an exited original owner."""
    root = _work_root(state_dir)
    if not root.is_dir():
        return []
    removed = []
    for directory in root.iterdir():
        if not re.fullmatch(r"w-[0-9a-f]{12}", directory.name) or not directory.is_dir() or _is_reparse(directory):
            continue
        try:
            owner = json.loads((directory / "owner.json").read_text(encoding="utf-8"))
            if (owner.get("schema_version") != 1 or owner.get("directory") != directory.name
                    or not isinstance(owner.get("engine_root"), str) or not owner["engine_root"]
                    or Path(owner.get("engine_root", "")).resolve() != ENGINE_ROOT
                    or isinstance(owner.get("pid"), bool) or not isinstance(owner.get("pid"), int) or not 0 < owner["pid"] <= 0xFFFFFFFF
                    or not isinstance(owner.get("create_time"), str)
                    or not re.fullmatch(r"windows:[0-9]+|linux:[0-9a-f-]+:[0-9]+", owner["create_time"])
                    or not isinstance(owner.get("job_id"), str) or not owner["job_id"]
                    or (job_id is not None and owner["job_id"] != job_id)):
                continue
            identity = process_identity(owner["pid"])
            if identity == owner["create_time"]:
                continue  # The original owner still lives.
            # None means exited; a different creation identity means PID reuse.
            _remove_owned_directory(directory, root)
            removed.append(str(directory))
        except (OSError, ValueError, TypeError, AttributeError):
            # An inaccessible or ambiguous owner/workspace must be preserved.
            continue
    return removed


@contextmanager
def private_workspace(job_id, state_dir=None):
    root = _work_root(state_dir)
    root.mkdir(parents=True, exist_ok=True)
    directory = root / f"w-{uuid.uuid4().hex[:12]}"
    directory.mkdir(exist_ok=False)
    try:
        identity = process_identity(os.getpid())
        if identity is None:
            raise RuntimeError("Cannot record this workspace's owner process.")
        owner = {"schema_version": 1, "directory": directory.name, "engine_root": str(ENGINE_ROOT),
                 "job_id": job_id, "pid": os.getpid(), "create_time": identity}
        (directory / "owner.json").write_text(json.dumps(owner), encoding="utf-8")
        yield directory
    finally:
        _remove_owned_directory(directory, root)
