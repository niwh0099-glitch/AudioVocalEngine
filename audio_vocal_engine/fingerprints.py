"""Standard-library content hashes, cached across catalog subprocesses."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .locking import DEFAULT_STATE_DIR


_cache = None


def _stat_key(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_dev]


def content_hash(path):
    global _cache
    path = Path(path).resolve()
    signature = _stat_key(path)
    cache_path = DEFAULT_STATE_DIR / "model-fingerprints.json"
    if _cache is None:
        try:
            saved = json.loads(cache_path.read_text(encoding="utf-8"))
            _cache = saved if isinstance(saved, dict) else {}
        except (OSError, ValueError):
            _cache = {}
    key = str(path)
    entry = _cache.get(key, {})
    if isinstance(entry, dict) and entry.get("stat") == signature and isinstance(entry.get("sha256"), str) and len(entry["sha256"]) == 64:
        return entry["sha256"]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if _stat_key(path) != signature:
        raise ValueError(f"File changed during fingerprint verification: {path}")
    result = digest.hexdigest()
    _cache[key] = {"stat": signature, "sha256": result}
    temporary = None
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_path.parent, prefix=".fingerprints-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(_cache, handle)
        os.replace(temporary, cache_path)
    except OSError:
        # Read-only installations can still inspect and execute models.
        pass
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return result


@lru_cache(maxsize=4)
def _hash_sources(signature):
    digest = hashlib.sha256()
    for path, _size, _mtime, _ctime in signature:
        digest.update(Path(path).relative_to(Path(__file__).parent).as_posix().encode("utf-8"))
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def engine_fingerprint():
    root = Path(__file__).parent
    paths = sorted(set(root.rglob("*.py")) | set(root.rglob("*.json")))
    signature = tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns) for path in paths)
    return _hash_sources(signature)
