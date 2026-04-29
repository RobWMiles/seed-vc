"""
Seed-VC RunPod worker.

One serverless endpoint that wraps Plachtaa/seed-vc's singing-voice
conversion pipeline. We pre-load the SVC model + RMVPE + Whisper +
BigVGAN at module import so each job only pays the inference cost,
not the model-load cost (which would be ~10 s otherwise).

Job input shape (set by the Firebase Function `v2generateMusic`):

    {
        "input": {
            "source_url": "https://…/song.mp3",     # song to convert
            "target_url": "https://…/voice.wav",    # singer reference (≤30 s)
            "diffusion_steps": 30,                  # optional
            "semi_tone_shift": 0,                   # optional
            "auto_f0_adjust": true                  # optional
        },
        "webhook": "https://…/v1seedVcCallback?…"
    }

Returns `{"audio_base64": "...", "mime_type": "audio/wav"}`. RunPod
forwards both that JSON and the webhook URL to the configured
callback endpoint.
"""

from __future__ import annotations

import base64
import logging
import os
import subprocess
import sys
import tempfile
import traceback
from typing import Any

import requests
import runpod

LOGGER = logging.getLogger("seedvc")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Model preload — see inference.py for the upstream definitions.
# We import lazily to keep cold-start logs clean on import errors.
# ---------------------------------------------------------------------------
_models: dict[str, Any] | None = None


def _ensure_models_loaded() -> dict[str, Any]:
    """Load Seed-VC's DiT + F0 + Whisper + BigVGAN once and cache them."""
    global _models
    if _models is not None:
        return _models

    LOGGER.info("loading Seed-VC models (one-shot)…")
    import argparse

    import inference  # type: ignore[import-not-found]

    args = argparse.Namespace(
        source="",
        target="",
        output=tempfile.gettempdir(),
        diffusion_steps=30,
        length_adjust=1.0,
        inference_cfg_rate=0.7,
        f0_condition=True,
        auto_f0_adjust=True,
        semi_tone_shift=0,
        checkpoint=None,
        config=None,
        fp16=True,
    )
    (
        model,
        semantic_fn,
        f0_fn,
        vocoder_fn,
        campplus_model,
        mel_fn,
        mel_fn_args,
    ) = inference.load_models(args)
    _models = {
        "model": model,
        "semantic_fn": semantic_fn,
        "f0_fn": f0_fn,
        "vocoder_fn": vocoder_fn,
        "campplus_model": campplus_model,
        "mel_fn": mel_fn,
        "mel_fn_args": mel_fn_args,
    }
    LOGGER.info("Seed-VC models loaded ✓")
    return _models


def _download(url: str, dest: str) -> str:
    """Stream a URL to disk; raise on non-2xx."""
    LOGGER.info("downloading %s → %s", url, dest)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)
    return dest


def _ffmpeg(args: list[str], step_name: str) -> None:
    """Run ffmpeg and surface stderr on failure — handler bubbles the
    output up to RunPod so the dashboard shows what actually broke
    instead of a generic non-zero exit."""
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg {step_name} failed (rc={proc.returncode}): {proc.stderr.strip()[:500]}"
        )


def _trim_to_30s(input_path: str, output_path: str) -> str:
    """Cut <input_path> to its first 30 s for use as the SVC reference.
    Seed-VC uses 1-30 s of reference, anything past that is wasted GPU
    work on the encoder."""
    _ffmpeg(
        [
            "-i", input_path,
            "-t", "30",
            "-ar", "44100",
            "-ac", "1",
            output_path,
        ],
        "trim-30s",
    )
    return output_path


def _normalize_source(input_path: str, output_path: str) -> str:
    """Re-encode the source song to a known WAV format. The download
    helper saves it with no real extension (we can't trust the upload
    URL's content-type), and Seed-VC's inference.py uses librosa
    which is mostly forgiving but occasionally chokes on
    extension-less files or odd codec variants. A round-trip through
    ffmpeg gives us a clean 44.1 kHz mono WAV that librosa accepts
    every time."""
    _ffmpeg(
        [
            "-i", input_path,
            "-ar", "44100",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            output_path,
        ],
        "normalize-source",
    )
    return output_path


def _run_inference(
    *,
    source_path: str,
    target_path: str,
    output_dir: str,
    diffusion_steps: int = 30,
    semi_tone_shift: int = 0,
    auto_f0_adjust: bool = True,
) -> str:
    """Run Seed-VC and return the path to the produced WAV."""
    _ensure_models_loaded()

    # We invoke the upstream `inference.main` via subprocess for now —
    # the function is one big script body and forking a child keeps the
    # parent's loaded weights warm for the next job. Subprocess
    # cold-loads the weights again (~10 s) but that's offset by Seed-VC
    # caching the rmvpe / whisper / bigvgan downloads on disk.
    #
    # TODO: refactor inference.main into a `convert(...)` callable so we
    # can share the loaded models with no subprocess. For MVP this
    # path is reliable and the inference time dominates anyway.
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "inference.py"),
        "--source",
        source_path,
        "--target",
        target_path,
        "--output",
        output_dir,
        "--diffusion-steps",
        str(diffusion_steps),
        "--length-adjust",
        "1.0",
        "--inference-cfg-rate",
        "0.7",
        "--f0-condition",
        "True",
        "--auto-f0-adjust",
        "True" if auto_f0_adjust else "False",
        "--semi-tone-shift",
        str(semi_tone_shift),
        "--fp16",
        "True",
    ]
    LOGGER.info("running inference: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    # Capture stdout + stderr on failure so the RunPod dashboard
    # shows the real Python traceback instead of a generic exit code.
    if proc.returncode != 0:
        tail_stdout = (proc.stdout or "").strip().splitlines()[-30:]
        tail_stderr = (proc.stderr or "").strip().splitlines()[-30:]
        details = "\n".join(["[stdout]", *tail_stdout, "[stderr]", *tail_stderr])
        LOGGER.error("inference subprocess failed (rc=%s):\n%s", proc.returncode, details)
        raise RuntimeError(
            f"inference.py failed (rc={proc.returncode}): "
            f"{(proc.stderr or '').strip()[-400:]}"
        )

    # inference.py writes vc_<source>_<target>_<la>_<ds>_<cfg>.wav into
    # output_dir. Pick the most recently modified file.
    candidates = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.lower().endswith(".wav")
    ]
    if not candidates:
        # Surface inference logs even on "no output" — sometimes
        # inference.py exits 0 but silently skipped writing the file.
        tail_stdout = (proc.stdout or "").strip().splitlines()[-20:]
        LOGGER.error("Seed-VC produced no .wav. stdout tail:\n%s", "\n".join(tail_stdout))
        raise RuntimeError(
            f"Seed-VC produced no .wav output. "
            f"Last stdout: {(proc.stdout or '').strip()[-300:]}"
        )
    return max(candidates, key=os.path.getmtime)


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """RunPod entry-point."""
    job_input = job.get("input", {}) or {}
    source_url = job_input.get("source_url")
    target_url = job_input.get("target_url")
    if not source_url or not target_url:
        return {"error": "source_url and target_url are required"}

    diffusion_steps = int(job_input.get("diffusion_steps", 30))
    semi_tone_shift = int(job_input.get("semi_tone_shift", 0))
    auto_f0_adjust = bool(job_input.get("auto_f0_adjust", True))

    work_dir = tempfile.mkdtemp(prefix="seedvc_")
    try:
        raw_source = _download(source_url, os.path.join(work_dir, "source_raw.audio"))
        # Round-trip the source through ffmpeg first — file from the
        # signed-storage download has no proper extension, and librosa
        # (used by inference.py) occasionally fails to identify the
        # codec without one. A normalized 44.1k mono WAV always loads.
        source_path = _normalize_source(raw_source, os.path.join(work_dir, "source.wav"))

        raw_target = _download(target_url, os.path.join(work_dir, "target_raw.audio"))
        target_path = _trim_to_30s(raw_target, os.path.join(work_dir, "target.wav"))

        output_dir = os.path.join(work_dir, "out")
        os.makedirs(output_dir, exist_ok=True)
        result_path = _run_inference(
            source_path=source_path,
            target_path=target_path,
            output_dir=output_dir,
            diffusion_steps=diffusion_steps,
            semi_tone_shift=semi_tone_shift,
            auto_f0_adjust=auto_f0_adjust,
        )

        with open(result_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("ascii")

        LOGGER.info("✓ converted (%d bytes audio)", len(audio_b64))
        return {
            "audio_base64": audio_b64,
            "mime_type": "audio/wav",
        }
    except Exception as e:
        # Trim the traceback so the JSON output stays compact in
        # RunPod's dashboard but keep the message + last few frames
        # so we can diagnose without SSH'ing into the worker.
        tb_tail = "\n".join(traceback.format_exc().strip().splitlines()[-12:])
        LOGGER.error("job failed: %s\n%s", e, tb_tail)
        return {
            "error": f"seedvc_failed: {e}",
            "traceback": tb_tail,
        }


if __name__ == "__main__":
    # Warm the models eagerly so the first real job is fast. Failures
    # here are non-fatal — we'd rather start serving than crash-loop.
    try:
        _ensure_models_loaded()
    except Exception as e:
        LOGGER.warning("preload failed (will retry per-job): %s", e)
    runpod.serverless.start({"handler": handler})
