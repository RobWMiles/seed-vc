# Seed-VC singing-voice-conversion runtime for the RunPod hub.
# Builds on a CUDA runtime image; pre-downloads the SVC checkpoints
# at build time so cold-start workers don't pay the HuggingFace
# download cost on the first job.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 as runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv \
        ffmpeg wget git ca-certificates curl && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /app

# Pip + wheel up first so error messages on the heavy installs below
# point at the actual failing dep, not at pip itself.
RUN pip install --upgrade pip wheel

# PyTorch from the cu121 wheel index. Separate RUN so a failure here
# is unambiguous and so the layer is cached independently of the
# generic-deps layer.
RUN pip install \
        "torch==2.4.0" \
        "torchvision==0.19.0" \
        "torchaudio==2.4.0" \
        --index-url https://download.pytorch.org/whl/cu121

# Seed-VC inference deps only. ALL specs quoted so the shell doesn't
# try to interpret `>=` as a stdout redirection.
#
# The upstream requirements.txt also lists `funasr`, `modelscope`,
# `resemblyzer`, and `jiwer` — these are only imported by Seed-VC's
# eval / real-time-GUI scripts, NOT by `inference.py`. funasr in
# particular pins torch < 2.4 which conflicts with our torch==2.4.0
# layer above; pulling those four out fixes the build and shaves
# ~2 GB off the image. If you ever need eval.py inside the worker,
# add them back to a separate optional layer.
RUN pip install \
        "runpod" \
        "requests" \
        "scipy==1.13.1" \
        "librosa==0.10.2" \
        "huggingface-hub==0.25.2" \
        "munch==4.0.0" \
        "einops==0.8.0" \
        "descript-audio-codec==1.0.0" \
        "pydub==0.25.1" \
        "transformers==4.46.3" \
        "soundfile==0.12.1" \
        "numpy==1.26.4" \
        "hydra-core==1.3.2" \
        "pyyaml" \
        "python-dotenv" \
        "accelerate"

# huggingface-hub 0.26+ removed `proxies` + `resume_download` from
# the ModelHubMixin._from_pretrained call path, but BigVGAN (used by
# Seed-VC for vocoding) declares those as required keyword-only args
# in its `_from_pretrained` override. Result with the upstream
# `>=0.28.1` pin: `TypeError: BigVGAN._from_pretrained() missing 2
# required keyword-only arguments`, every job fails. 0.25.2 is the
# last release that still passes them and is compatible with our
# transformers==4.46.3 (which requires >=0.23.2).

# Pre-fetch SVC weights (~2 GB) so the first job doesn't stall behind
# a HuggingFace download. Path layout matches Seed-VC's hf_utils helper
# (./checkpoints/hf_cache → resolved by `cache_dir='/app/checkpoints'`).
# Note: hf_transfer extra is installed AT THE PINNED 0.25.2 — using
# `pip install --upgrade` here would silently reinstall hub at latest
# and bring back the BigVGAN incompatibility.
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN pip install "huggingface_hub[hf_transfer]==0.25.2" && \
    python -c "from huggingface_hub import hf_hub_download; \
        hf_hub_download('Plachta/Seed-VC', 'DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth', cache_dir='/app/checkpoints'); \
        hf_hub_download('Plachta/Seed-VC', 'config_dit_mel_seed_uvit_whisper_base_f0_44k.yml', cache_dir='/app/checkpoints'); \
        hf_hub_download('lj1995/VoiceConversionWebUI', 'rmvpe.pt', cache_dir='/app/checkpoints')" && \
    python -c "from transformers import AutoModel; AutoModel.from_pretrained('openai/whisper-small')" || true

# Source last so the model layers above stay cached across handler edits.
COPY . /app

RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
