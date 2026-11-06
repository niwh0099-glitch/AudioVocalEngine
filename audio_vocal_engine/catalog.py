"""The seven supported weights; independent of the inference environment."""

import hashlib
import json
from pathlib import Path

from . import __version__
from .fingerprints import content_hash, engine_fingerprint


MODEL_SPECS = (
    ("HP2_all_vocals", "HP2 全部人声", "HP2_all_vocals.pth", "vr", "separate", "分离人声与伴奏，保留较完整的人声。"),
    ("HP5_only_main_vocal", "HP5 主人声", "HP5_only_main_vocal.pth", "vr", "separate", "提取主人声、减少和声，可能削弱部分人声细节。"),
    ("model_bs_roformer_ep_317_sdr_12.9755", "BS Roformer ep_317", "model_bs_roformer_ep_317_sdr_12.9755.ckpt", "roformer", "separate", "分离人声与背景音乐，可与 HP2 对比试听。"),
    ("onnx_dereverb_By_FoxJoy", "FoxJoy 去混响", "onnx_dereverb_By_FoxJoy/vocals.onnx", "mdx", "dereverb", "去混响，主要用于双声道音频。"),
    ("VR-DeEchoAggressive", "DeEcho 强力去回声", "VR-DeEchoAggressive.pth", "deecho", "deecho", "更强地去除回声、延迟效果。"),
    ("VR-DeEchoDeReverb", "DeEcho 去回声与混响", "VR-DeEchoDeReverb.pth", "deecho", "dereverb", "同时处理回声和混响。"),
    ("VR-DeEchoNormal", "DeEcho 常规去回声", "VR-DeEchoNormal.pth", "deecho", "deecho", "去除回声、延迟效果。"),
)


def load_manifest():
    try:
        manifest = json.loads((Path(__file__).parent / "configs/model_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") == 1 and isinstance(manifest.get("models"), dict):
            return manifest["models"]
    except (OSError, ValueError, AttributeError):
        pass
    return {}


def model_config_path(model_id, kind):
    root = Path(__file__).parent
    if kind == "roformer":
        return root / "configs" / f"{model_id}.json"
    if kind in ("vr", "deecho"):
        return root / "vendor/lib/lib_v5/modelparams" / ("4band_v2.json" if kind == "vr" else "4band_v3.json")
    return root / "vendor/mdxnet.py"


def inspect_weight(path, expected, relative):
    if not path.is_file():
        return False, "missing", "Model file was not found.", "missing"
    try:
        stat = path.stat()
        if not isinstance(expected, dict) or expected.get("path") != relative or not isinstance(expected.get("bytes"), int) or not isinstance(expected.get("sha256"), str):
            return False, "invalid", "Verified model manifest is missing or invalid.", f"unverified:{stat.st_size}:{stat.st_mtime_ns}"
        if stat.st_size != expected["bytes"] or stat.st_size == 0:
            return False, "invalid", "Model size does not match the verified weight.", f"invalid:{stat.st_size}:{stat.st_mtime_ns}"
        digest = content_hash(path)
        if digest != expected["sha256"]:
            return False, "invalid", "Model SHA256 does not match the verified weight.", digest
        return True, "ready", "Verified model file.", digest
    except (OSError, ValueError) as exc:
        return False, "invalid", str(exc), "unreadable"


def list_models(model_root):
    root = Path(model_root).expanduser().resolve()
    manifest = load_manifest()
    engine_digest = engine_fingerprint()
    models = []
    for model_id, name, relative, kind, task, description in MODEL_SPECS:
        path = root / relative
        available, status, detail, weight_digest = inspect_weight(path, manifest.get(model_id), relative)
        try:
            config_digest = hashlib.sha256(model_config_path(model_id, kind).read_bytes()).hexdigest()
        except OSError:
            available, status, detail, config_digest = False, "invalid", "Native model configuration is missing.", "missing"
        fingerprint = hashlib.sha256(f"{__version__}:{engine_digest}:{weight_digest}:{config_digest}".encode("utf-8")).hexdigest()
        models.append({
            "id": model_id, "name": name, "type": "vocal", "architecture": "uvr5",
            "task": task, "description": description,
            "parameters": {"aggressiveness": {"minimum": 0, "maximum": 20, "default": 10}} if kind in ("vr", "deecho") else {},
            "precisions": ["auto", "fp32"] if kind == "mdx" else ["auto", "fp32", "fp16"],
            "available": available, "status": status, "status_detail": detail,
            "version": f"{__version__}:{fingerprint}", "path": str(path),
        })
    return models


def get_model(model_root, model_id):
    for model in list_models(model_root):
        if model["id"] == model_id:
            return model
    raise ValueError(f"Unknown model_id: {model_id}")


def model_kind(model_id):
    for spec in MODEL_SPECS:
        if spec[0] == model_id:
            return spec[3]
    raise ValueError(f"Unknown model_id: {model_id}")


def primary_role(model_id):
    return "cleaned" if model_kind(model_id) in ("deecho", "mdx") else "vocals"
