"""Safe input conversion and measured PCM WAV output metadata."""

import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import uuid
import wave

from .work import filesystem_path


def ffmpeg_path():
    configured = os.environ.get("AUDIO_VOCAL_ENGINE_FFMPEG")
    if configured:
        if not Path(configured).is_file():
            raise FileNotFoundError(f"Configured FFmpeg was not found: {configured}")
        return configured
    try:
        import imageio_ffmpeg
        from imageio_ffmpeg._definitions import FNAME_PER_PLATFORM, get_platform

        # Select this environment's wheel-bundled binary directly. The public
        # auto-discovery also consults inherited environment/PATH entries.
        executable = Path(imageio_ffmpeg.__file__).parent / "binaries" / FNAME_PER_PLATFORM.get(get_platform(), "")
        if not executable.is_file():
            raise FileNotFoundError("The engine environment's bundled FFmpeg binary is missing.")
        return str(executable.resolve())
    except ImportError as exc:
        raise RuntimeError("Install imageio-ffmpeg in this engine environment or set AUDIO_VOCAL_ENGINE_FFMPEG.") from exc


def prepare_input(source, destination):
    """Convert a copy to the native 44100 Hz stereo input; never change source."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("Input conversion destination must differ from the source.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [ffmpeg_path(), "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
         "-i", filesystem_path(source), "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", filesystem_path(destination)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode:
        raise RuntimeError(f"Input conversion failed: {completed.stderr.strip()}")
    metadata = wav_metadata(destination)
    if metadata["sample_rate"] != 44100 or metadata["channels"] != 2:
        raise ValueError("Converted input is not 44100 Hz stereo PCM WAV.")
    return destination


def wav_metadata(path):
    path = Path(path).resolve()
    with wave.open(filesystem_path(path), "rb") as audio:
        channels, rate, frames = audio.getnchannels(), audio.getframerate(), audio.getnframes()
        frame_bytes = channels * audio.getsampwidth()
        actual_bytes = 0
        while True:
            block = audio.readframes(65536)
            if not block:
                break
            actual_bytes += len(block)
    if rate <= 0 or channels <= 0 or frames <= 0:
        raise ValueError(f"WAV output is empty or invalid: {path}")
    if actual_bytes != frames * frame_bytes:
        raise ValueError(f"WAV payload is truncated: {path}")
    return {"path": str(path), "sample_rate": rate, "channels": channels,
            "frames": frames, "duration": frames / rate}


def publish_output(source, output_dir, role):
    """Copy a validated WAV beside its final path, then publish without overwrite."""
    if role not in ("vocals", "cleaned", "background", "removed"):
        raise ValueError(f"Unknown output role: {role}")
    output_dir = Path(output_dir).resolve()
    Path(filesystem_path(output_dir)).mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{role}-", suffix=".tmp", dir=filesystem_path(output_dir), delete=False) as handle:
            temporary = handle.name
            with open(filesystem_path(source), "rb") as original:
                shutil.copyfileobj(original, handle)
            handle.flush()
            os.fsync(handle.fileno())
        while True:
            destination = output_dir / f"{role}-{uuid.uuid4().hex[:12]}.wav"
            try:
                if os.name == "nt":
                    os.rename(temporary, filesystem_path(destination))  # Windows refuses an existing destination.
                else:
                    os.link(temporary, destination)  # Atomic exclusive publication on POSIX.
                    os.unlink(temporary)
                temporary = None
                return destination
            except FileExistsError:
                continue
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
