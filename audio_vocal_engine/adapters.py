"""Explicit native output routing; filenames never determine an output's role."""

from pathlib import Path

from .catalog import model_kind


class NativeAdapter:
    def __init__(self, native, kind, runtime=None):
        self.native = native
        self.kind = kind
        self.runtime = runtime or {}

    def process(self, source, output_dir):
        output_dir = Path(output_dir)
        primary = "cleaned" if self.kind in ("deecho", "mdx") else "vocals"
        secondary = "removed" if primary == "cleaned" else "background"
        primary_dir, secondary_dir = output_dir / primary, output_dir / secondary
        primary_dir.mkdir(parents=True, exist_ok=False)
        secondary_dir.mkdir(parents=True, exist_ok=False)
        if self.kind == "vr":
            self.native._path_audio_(str(source), ins_root=str(secondary_dir), vocal_root=str(primary_dir), format="wav")
        elif self.kind == "deecho":
            # AudioPreDeEcho predicts the cleaned signal into ins_root. Its
            # vocal_root contains the removed residual, unlike AudioPre.
            self.native._path_audio_(str(source), ins_root=str(primary_dir), vocal_root=str(secondary_dir), format="wav")
        elif self.kind in ("mdx", "roformer"):
            self.native._path_audio_(str(source), others_root=str(secondary_dir), vocal_root=str(primary_dir), format="wav")
        else:
            raise ValueError(f"Unknown adapter: {self.kind}")
        outputs = []
        for role, directory in ((primary, primary_dir), (secondary, secondary_dir)):
            files = sorted(directory.glob("*.wav"))
            if len(files) != 1:
                raise RuntimeError(f"Expected one WAV for {role}, found {len(files)}.")
            outputs.append({"role": role, "path": str(files[0].resolve())})
        return outputs

    def close(self):
        """Free model memory while the process still owns the engine lock."""
        import gc
        import torch

        self.native = None
        gc.collect()
        if self.runtime.get("device") == "cuda":
            torch.cuda.empty_cache()


def resolve_runtime(device, precision, kind, model_id=None):
    import torch

    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda.")
    if precision not in ("auto", "fp32", "fp16"):
        raise ValueError("precision must be auto, fp32, or fp16.")
    if kind not in ("vr", "deecho", "mdx", "roformer"):
        raise ValueError(f"Unknown adapter: {kind}")
    if kind == "mdx" and precision == "fp16":
        raise ValueError("FoxJoy's ONNX graph requires fp32; fp16 is unsupported.")

    cpu_threads = min(4, torch.get_num_threads())
    torch.set_num_threads(cpu_threads)
    gib = 1024 ** 3
    required_free_gpu_bytes = (4 if kind in ("mdx", "roformer") else 3 if model_id == "VR-DeEchoDeReverb" else 2) * gib
    free_gpu_bytes = None
    total_gpu_bytes = None
    cuda_available = torch.cuda.is_available()
    if kind == "mdx":
        import onnxruntime
        cuda_available = cuda_available and "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    if device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested, but the selected model's CUDA runtime is unavailable.")

    actual_device = "cpu"
    selection_reason = "按请求使用 CPU。" if device == "cpu" else "CUDA 运行环境不可用，自动使用 CPU。"
    if device != "cpu" and cuda_available:
        try:
            free_gpu_bytes, total_gpu_bytes = map(int, torch.cuda.mem_get_info())
            if free_gpu_bytes < 0 or total_gpu_bytes <= 0:
                raise ValueError("CUDA returned invalid memory information.")
        except Exception as exc:
            free_gpu_bytes, total_gpu_bytes = None, None
            if device == "cuda":
                raise RuntimeError(f"Cannot verify available GPU memory for the requested CUDA device: {exc}") from exc
            selection_reason = "无法确认可用显存，保守选择 CPU。"
        else:
            if free_gpu_bytes < required_free_gpu_bytes:
                detail = f"可用显存 {free_gpu_bytes / gib:.1f} GiB 低于模型保守预留 {required_free_gpu_bytes / gib:.1f} GiB"
                if device == "cuda":
                    raise RuntimeError(f"{detail}；显式 CUDA 请求无法安全执行，请释放显存或选择 CPU。")
                selection_reason = f"{detail}，自动使用 CPU。"
            else:
                actual_device = "cuda"
                selection_reason = "CUDA 可用且满足模型的保守显存预留。"
    if kind == "mdx":
        actual_precision = "fp32"
    else:
        actual_precision = "fp16" if precision == "auto" and actual_device == "cuda" else precision
        if actual_precision == "auto":
            actual_precision = "fp32"
        if actual_precision == "fp16" and actual_device == "cpu":
            raise ValueError("fp16 requires a CUDA device; use fp32 or auto on CPU.")
    return {"device": actual_device, "precision": actual_precision,
            "selection_reason": selection_reason, "free_gpu_bytes": free_gpu_bytes,
            "total_gpu_bytes": total_gpu_bytes, "required_free_gpu_bytes": required_free_gpu_bytes,
            "torch_num_threads": cpu_threads}


def create_backend(model, device="auto", precision="auto", parameters=None):
    kind = model_kind(model["id"])
    runtime = resolve_runtime(device, precision, kind, model["id"])
    is_half = runtime["precision"] == "fp16"
    parameters = parameters or {"aggressiveness": 10}
    if kind in ("vr", "deecho"):
        from .vendor.vr import AudioPre, AudioPreDeEcho
        cls = AudioPre if kind == "vr" else AudioPreDeEcho
        native = cls(parameters["aggressiveness"], model["path"], runtime["device"], is_half)
    elif kind == "roformer":
        from .vendor.bsroformer import Roformer_Loader
        config_path = Path(__file__).parent / "configs" / f"{model['id']}.json"
        native = Roformer_Loader(model["path"], str(config_path), runtime["device"], is_half)
    else:
        from .vendor.mdxnet import MDXNetDereverb
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if runtime["device"] == "cuda" else ["CPUExecutionProvider"]
        native = MDXNetDereverb(15, onnx_dir=str(Path(model["path"]).parent), providers=providers)
        active = native.pred.model.get_providers()
        actual_device = "cuda" if "CUDAExecutionProvider" in active else "cpu"
        if device == "cuda" and actual_device != "cuda":
            raise RuntimeError("ONNX Runtime could not activate the requested CUDA provider.")
        if runtime["device"] == "cuda" and actual_device == "cpu":
            runtime["selection_reason"] = "ONNX CUDA 提供程序未能启用，自动使用 CPU。"
        runtime["device"] = actual_device
        runtime["providers"] = active
        runtime["ort_intra_op_num_threads"] = 4
    return NativeAdapter(native, kind, runtime)
