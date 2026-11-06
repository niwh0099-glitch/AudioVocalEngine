"""Request validation, serialized execution, and atomic per-item checkpoints."""

import json
import math
import os
from pathlib import Path
import tempfile

from .adapters import create_backend
from .audio import prepare_input, publish_output, wav_metadata
from .catalog import get_model, model_kind, primary_role
from .locking import DEFAULT_STATE_DIR, EngineLock
from .work import cleanup_work, filesystem_path, private_workspace


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_request(request):
    if not isinstance(request, dict) or request.get("schema_version") != 1:
        raise ValueError("A request with schema_version=1 is required.")
    for field in ("job_id", "caller", "model_root", "model_id"):
        if not isinstance(request.get(field), str) or not request[field].strip():
            raise ValueError(f"A nonempty {field} is required.")
    if request.get("device", "auto") not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda.")
    if request.get("precision", "auto") not in ("auto", "fp32", "fp16"):
        raise ValueError("precision must be auto, fp32, or fp16.")
    parameters = request.get("parameters", {})
    supported = {"aggressiveness"} if model_kind(request["model_id"]) in ("vr", "deecho") else set()
    if not isinstance(parameters, dict) or set(parameters) - supported:
        raise ValueError(f"Unsupported parameters for {request['model_id']}.")
    aggressiveness = parameters.get("aggressiveness", 10)
    if isinstance(aggressiveness, bool) or not isinstance(aggressiveness, int) or not 0 <= aggressiveness <= 20:
        raise ValueError("aggressiveness must be an integer from 0 to 20.")
    items = request.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("At least one item is required.")
    identifiers = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Every item must be an object.")
        for field in ("id", "input", "output_dir"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"Every item requires a nonempty {field}.")
        if item["id"] in identifiers:
            raise ValueError(f"Duplicate item id: {item['id']}")
        identifiers.add(item["id"])
        for field in ("source_start", "source_end"):
            value = item.get(field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                raise ValueError(f"{field} must be a finite nonnegative number.")
        if item.get("source_start") is not None and item.get("source_end") is not None and item["source_end"] < item["source_start"]:
            raise ValueError("source_end must be at least source_start.")
    return {"aggressiveness": aggressiveness} if supported else {}


def run_job(request, result_path, event, backend_factory=None, state_dir=None):
    parameters = validate_request(request)
    model = get_model(request["model_root"], request["model_id"])
    result_path = Path(result_path).resolve()
    for item in request["items"]:
        if result_path == Path(item["input"]).resolve():
            raise ValueError("The result path cannot replace an input audio file.")
    result = {"schema_version": 1, "job_id": request["job_id"], "caller": request["caller"],
              "model": model, "parameters": parameters, "runtime": {}, "items": []}

    def emit(stage, message, progress=0.0, indeterminate=False, **extra):
        event({"progress": progress * 100, "message": message, "stage": stage,
               "indeterminate": indeterminate, **extra})

    # The caller supplies a new result path for each job. Refuse to overwrite
    # unrelated or previous results, while later checkpoints replace our file.
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False)
    lock = EngineLock(Path(state_dir or DEFAULT_STATE_DIR) / "engine.lock")
    backend = None
    try:
        if not model["available"]:
            raise ValueError(f"Model unavailable ({model['status']}): {model['path']}. {model.get('status_detail', '')}")
        lock.acquire(on_wait=lambda: emit("waiting", "等待其他音频任务完成", indeterminate=True))
        cleanup_work(state_dir=state_dir)
        # A job can wait while another caller maintains model files. Validate
        # again after waiting so replaced weights are never silently loaded.
        model = get_model(request["model_root"], request["model_id"])
        result["model"] = model
        if not model["available"]:
            raise ValueError(f"Model unavailable ({model['status']}): {model['path']}. {model.get('status_detail', '')}")
        emit("loading", "正在加载人声模型", indeterminate=True)
        backend = (backend_factory or create_backend)(model, request.get("device", "auto"), request.get("precision", "auto"), parameters)
        result["runtime"] = dict(getattr(backend, "runtime", {}))
        atomic_json(result_path, result)
        total = len(request["items"])
        for index, item in enumerate(request["items"]):
            item_result = {key: item[key] for key in ("id", "clip_id", "input_version_id") if key in item}
            item_result.update(primary_role=primary_role(model["id"]), outputs=[], runtime=dict(result["runtime"]), model=dict(model))
            try:
                source = Path(item["input"]).expanduser().resolve()
                if not Path(filesystem_path(source)).is_file():
                    raise FileNotFoundError(f"Input audio not found: {source}")
                base = Path(item["output_dir"]).expanduser().resolve()
                Path(filesystem_path(base)).mkdir(parents=True, exist_ok=True)
                # libsndfile/FFmpeg native paths remain short regardless of the
                # caller's project and clip nesting. Published files stay in base.
                with private_workspace(request["job_id"], state_dir) as output_dir:
                    emit("preparing", f"正在准备音频 {index + 1}/{total}", index / total, True)
                    prepared = prepare_input(source, output_dir / "i.wav")
                    input_metadata = wav_metadata(prepared)
                    source_start = item.get("source_start")
                    if source_start is None:
                        source_start = 0.0
                    source_end = item.get("source_end")
                    if source_end is None:
                        source_end = source_start + input_metadata["duration"]
                    elif abs(source_end - source_start - input_metadata["duration"]) > 0.15:
                        raise ValueError("Source range duration differs from the prepared input by more than 0.15 seconds.")
                    emit("processing", f"正在处理音频 {index + 1}/{total}", index / total, True)
                    outputs = backend.process(prepared, output_dir)
                    if not outputs:
                        raise RuntimeError("The model returned no audio outputs.")
                    measured = []
                    for output in outputs:
                        path = Path(output["path"]).resolve()
                        if not path.is_relative_to(output_dir.resolve()) or path == source or path == Path(prepared).resolve():
                            raise ValueError("Model output must remain inside its unique private workspace.")
                        metadata = wav_metadata(path)
                        if metadata["sample_rate"] != 44100 or metadata["channels"] != 2:
                            raise ValueError(f"Expected native 44100 Hz stereo WAV output: {path}")
                        if abs(metadata["duration"] - input_metadata["duration"]) > 0.15:
                            raise ValueError(f"Output duration differs from the prepared input by more than 0.15 seconds: {path}")
                        measured.append({"role": output["role"], **metadata,
                                         "source_start": source_start, "source_end": source_end})
                    if item_result["primary_role"] not in [output["role"] for output in measured]:
                        raise ValueError("The required primary audio output is missing.")
                    for output in measured:
                        output["path"] = str(publish_output(output["path"], base, output["role"]))
                    item_result["outputs"] = measured
            except Exception as exc:
                item_result["error"] = f"{type(exc).__name__}: {exc}"
            result["items"].append(item_result)
            atomic_json(result_path, result)
            emit("item_completed", f"已处理音频 {index + 1}/{total}", (index + 1) / total, item=item_result)
        failures = sum("error" in item for item in result["items"])
        if failures:
            result["error"] = f"{failures} of {total} audio items failed."
        atomic_json(result_path, result)
        emit("completed", "音频处理完成" if not failures else "音频处理完成，部分项目失败", 1.0)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        atomic_json(result_path, result)
        emit("failed", result["error"])
    finally:
        try:
            if backend is not None and hasattr(backend, "close"):
                backend.close()
        finally:
            lock.release()
    return result
