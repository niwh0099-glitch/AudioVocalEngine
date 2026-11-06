# Third-party code

The files in `audio_vocal_engine/vendor` were copied from the local GPT-SoVITS checkout's `tools/uvr5` core (`vr.py`, `mdxnet.py`, `bsroformer.py`, `lib/`, and `bs_roformer/`). The upstream checkout is GPT-SoVITS-v2pro-20250604. Upstream project: https://github.com/RVC-Boss/GPT-SoVITS.

The original MIT license, copyright (c) 2024 RVC-Boss, is retained verbatim in `audio_vocal_engine/vendor/LICENSE-GPT-SoVITS.txt`. Existing source attribution comments, including the Roformer attribution to ZFTurbo, remain in the copied source.

Local adaptations use package-relative imports, accept the FoxJoy ONNX directory and execution providers explicitly, cap its ONNX intra-op thread pool at four, honor the requested Roformer precision in autocast, and remove an obsolete configuration error referring users to the original project's model directory. VR inverse STFT calls explicitly supply the spectrogram's real dtype to preserve precision and bypass librosa 0.9's removed NumPy alias. Roformer ep_317's original default configuration is also stored explicitly under `audio_vocal_engine/configs`.

No model weights are included in this source package. Model files and their applicable terms remain separate.
