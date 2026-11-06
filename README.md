# AudioVocalEngine

独立的 UVR 人声处理引擎。代码、Python 环境和模型根目录均由本引擎自己管理；运行时不导入 GPT-SoVITS，不修改原音频或原项目模型。

## 源码发布与安装

本仓库发布 **1.0.0 源码基线**，提供人声分离、去回声、去混响、逐项进度和任务清理接口，供 GPT-SoVITS 等调用方复用。保留原模型算法与第三方许可，不包含环境、权重、用户音频或任务记录。

Windows 使用 Python 3.11，在 PowerShell 中执行：

```powershell
git clone https://github.com/niwh0099-glitch/AudioVocalEngine.git
cd AudioVocalEngine
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install --no-deps .
.venv/Scripts/python.exe -m audio_vocal_engine models --model-root ./models
```

当前依赖清单对应 CUDA 12.8 / Torch 2.8 的独立 GPU 环境，需要兼容的 NVIDIA 驱动；不要求修改其他 AI 项目的环境。引擎支持 CPU 执行，但本次没有提供或验证独立 CPU-only 安装锁。最后一条命令仅检查模型清单，未导入模型时显示缺失是正常现象。按下方清单自行准备有权使用的权重，或由调用方配置其他模型目录；源码下载不会自动获得权重。本次源码发布不等于已在全新电脑完成安装或音质验收。

## 调用

在引擎目录中使用该目录专属的 Python：

```powershell
.venv/Scripts/python.exe -m audio_vocal_engine models --model-root ./models
.venv/Scripts/python.exe -m audio_vocal_engine run request.json result.json
.venv/Scripts/python.exe -m audio_vocal_engine cleanup --job-id example-001
```

`models` 仅依赖 Python 标准库，不加载 Torch，也不加载模型。输出为 UTF-8 JSON，包含固定的七个模型及各自的文件可用状态。模型的字节数和 SHA256 必须匹配本引擎已验证的 `configs/model_manifest.json`；空文件、占位文件和未经确认的替换权重显示 `invalid`。清单缺失时仍返回七条模型状态，不中断调用方启动。确需更换权重时，应先明确验证兼容性，再维护清单。

模型 `version` 包含权重、原生配置和引擎源码版本的内容指纹。权重哈希按路径、文件大小、修改和创建时间、文件标识缓存到 `.state/model-fingerprints.json`，连续扫描不重复读取全部权重。`run` 的标准输出是 UTF-8 JSON Lines 进度事件，诊断信息写到标准错误；全部成功返回 0，否则返回 1。为每次任务提供一个新的结果文件路径。

## 请求

```json
{
  "schema_version": 1,
  "job_id": "example-001",
  "caller": "MyAudioTool",
  "model_root": "./models",
  "model_id": "HP2_all_vocals",
  "device": "auto",
  "precision": "auto",
  "parameters": {"aggressiveness": 10},
  "items": [{
    "id": "item-001",
    "clip_id": "clip-001",
    "input_version_id": "original",
    "input": "./my-audio/voice.wav",
    "output_dir": "./outputs/example-001",
    "source_start": 0.0,
    "source_end": null
  }]
}
```

`items` 没有 12 项上限，ID 必须唯一。输入应为调用方选定的完整音频或已经裁好的片段；`source_start` / `source_end` 记录片段在原音频中的位置，不会再次裁剪。缺少开始位置时使用 0；结束位置为 null 时，按实际输入时长补全；显式区间与实际输入时长相差超过 0.15 秒时报错。

`aggressiveness` 为 0 到 20 的整数，默认 10，仅 HP2、HP5 和三种 VR DeEcho 支持；FoxJoy 和 Roformer 使用 `parameters: {}`。设备支持 `auto` / `cpu` / `cuda`，精度支持 `auto` / `fp32` / `fp16`。VR / Roformer 自动精度在 CUDA 下使用 fp16，在 CPU 下使用 fp32；FoxJoy 固定 fp32，其 ONNX 执行提供程序可使用 CPU 或 CUDA。显式请求不可用的 CUDA 或不支持的 fp16 会报错；实际设备和精度记录在结果顶层及每项的 `runtime` 中。

取得引擎锁后、模型加载前，通过 `torch.cuda.mem_get_info()` 检查当前空闲显存。FoxJoy / BS Roformer 保守要求至少 4 GiB，VR-DeEchoDeReverb 至少 3 GiB，其余 VR 模型至少 2 GiB。`device=auto` 在余量不足或无法确认余量时选择 CPU，并记录 `runtime.selection_reason`、`free_gpu_bytes` 和 `required_free_gpu_bytes`；显式 `cuda` 请求则清楚报错，不自动换设备。显式 `cpu` 不查询或占用 CUDA 上下文。此规则是针对 Windows WDDM 及同时运行其他应用的保守资源预算，无法保证检查之后的显存状态，也不代表模型质量变化；不会改模型、分块或请求精度。显式 fp16 在 CPU 上仍按原规则报错。

引擎把 Torch 的 CPU 运算线程限制为最多 4，FoxJoy 的 ONNX Runtime 会话使用 `intra_op_num_threads=4`，并随 `runtime` 记录实际线程设置。资源选择不关闭或调整其他应用。

## 输出与排队

| 模型 | 主输出 `primary_role` | 第二输出 |
| --- | --- | --- |
| HP2、HP5、BS Roformer ep_317 | `vocals` | `background` |
| 三种 VR DeEcho、FoxJoy | `cleaned` | `removed` |

DeEcho 的净化输出明确对应原生 `ins_root`，FoxJoy 的净化输出对应 `vocal_root`。输出角色由适配器参数定义，不从文件名猜测。

所有调用方共用引擎安装目录下 `.state/engine.lock` 的操作系统文件锁。取得锁之后才导入和加载模型；等待期间每约 2 秒发送 `stage=waiting` 的不定进度事件。进程退出后由操作系统自动释放锁。模型释放后才主动解锁，避免下一位调用方同时占用模型显存。

事件阶段为 `waiting`、`loading`、`preparing`、`processing`、`item_completed`、`completed`；整体失败使用 `failed`。`progress` 范围为 0 到 100，`indeterminate` 表示当前阶段无法估算内部进度。只有 `item_completed` 事件携带完整 `item`，且该项已经先保存到结果文件。

每一项在本引擎 `.state/work/w-<唯一标识>` 私有短目录中准备输入副本和原生输出，避免调用方项目层级超过 Windows 原生音频库的路径限制。FFmpeg 将副本转换为 44100 Hz、双声道 PCM WAV，然后调用模型。全部输出通过 WAV 校验后，先复制到调用方 `output_dir` 内的 `.tmp` 文件，再原子发布为短文件名 `<role>-<唯一标识>.wav`；拒绝覆盖已有文件。结果路径仍为调用方目录内的普通绝对路径。

输出包含 `role`、`path`、`sample_rate`、`channels`、`frames`、`duration`、`source_start`、`source_end`。时长来自 WAV 的实际帧数，检查数据未截断、声道和采样率正确，且与准备后的输入时长差不超过 0.15 秒。一个项目失败会记录该项的 `error`，继续后续项目。每项完成后原子替换结果文件，保留此前已完成结果。

正常完成、异常或可处理的取消都会清理该项的私有工作目录。强制结束进程后，调用方可运行 `cleanup --job-id ID` 立即清理；关闭应用时无需再启动进程，下次持锁任务会清理已退出进程留下的工作目录。清理只接受私有目录 `owner.json` 中精确匹配的任务、进程 ID 和创建时间；原进程仍在运行、无法确认身份或存在重解析链接时保留目录。已发布音频不在私有目录中，不会被清理。命令仅依赖标准库，输出 `{"job_id":"ID","removed":["已清理的私有目录"]}`，正常返回 0。

若强制结束恰逢向调用方目录复制文件，可能留下未发布的 `.tmp`；引擎清理严格限定于 `.state/work`，此类项目临时文件由调用方后续定向回收。

结果顶层包含 `schema_version`、`job_id`、`caller`、`model`、`parameters`、`runtime`、`items`，失败时包含 `error`。每项保留请求的 `id`、`clip_id`、`input_version_id`，并提供实际 `model`、`runtime`、`primary_role` 和 `outputs`。

## 环境

推理环境需要 Torch、NumPy、librosa、soundfile、tqdm、PyYAML、einops、rotary-embedding-torch、packaging，以及 ONNX Runtime（使用 CUDA 时需要对应 GPU 运行环境）；精确版本记录在 `requirements.txt`。FFmpeg 默认只使用当前环境内 `imageio-ffmpeg` 的配套程序，不从继承的 PATH 或其他项目查找；只有显式设置 `AUDIO_VOCAL_ENGINE_FFMPEG` 时才使用指定外部程序。

模型文件只从 `--model-root` / `model_root` 读取：

```text
HP2_all_vocals.pth
HP5_only_main_vocal.pth
model_bs_roformer_ep_317_sdr_12.9755.ckpt
onnx_dereverb_By_FoxJoy/vocals.onnx
VR-DeEchoAggressive.pth
VR-DeEchoDeReverb.pth
VR-DeEchoNormal.pth
```

Roformer 使用本引擎内的显式 `audio_vocal_engine/configs/model_bs_roformer_ep_317_sdr_12.9755.json`，内容对应已复制 ep_317 权重的原默认配置。

## 验证与来源

标准库核心测试：`python -m unittest discover -s tests -v`。这些测试检查模型目录、输出方向、逐项保存、13 项批处理和跨进程锁释放，不代表模型分离质量验收。

`audio_vocal_engine/vendor` 保留了迁移时的 GPT-SoVITS UVR 核心和许可证，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。模型权重单独管理，不打包进代码。

安装步骤见本文开头。依赖安装与模型准备分离；不会自动下载权重，也不引用 GPT 环境。`local-runtime-copy.json` 是旧机器的环境准备记录，不属于公开发行内容，安装不依赖它。

资源保护采用保守显存余量：FoxJoy/BS Roformer4GiB、DeEchoDeReverb3GiB、其他VR2GiB。自动模式不足或无法查询时使用CPU并记录原因；显式CUDA不足时失败，保持显式精度要求。Torch/ORT线程池限制最多4，不更改原生模型或分块。阈值用于避免并发程序占用下显存争用，不是模型质量或通用性能承诺。
