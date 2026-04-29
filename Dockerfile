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

# Seed-VC's general Python deps. ALL package specs are quoted so the
# shell doesn't try to interpret `>=` as a stdout redirection — that
# is what blew up the previous build (`huggingface-hub>=0.28.1`
# silently became `huggingface-hub` + a redirect to file `0.28.1`,
# and pip then errored on the truncated request).
RUN pip install \
        "runpod" \
        "requests" \
        "scipy==1.13.1" \
        "librosa==0.10.2" \
        "huggingface-hub>=0.28.1" \
        "munch==4.0.0" \
        "einops==0.8.0" \
        "descript-audio-codec==1.0.0" \
        "pydub==0.25.1" \
        "transformers==4.46.3" \
        "soundfile==0.12.1" \
        "modelscope==1.18.1" \
        "funasr==1.1.5" \
        "numpy==1.26.4" \
        "hydra-core==1.3.2" \
        "pyyaml" \
        "python-dotenv" \
        "accelerate" \
        "resemblyzer" \
        "jiwer==3.0.3"

# Pre-fetch SVC weights (~2 GB) so the first job doesn't stall behind
# a HuggingFace download. Path layout matches Seed-VC's hf_utils helper
# (./checkpoints/hf_cache → resolved by `cache_dir='/app/checkpoints'`).
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN pip install --upgrade "huggingface_hub[hf_transfer]" && \
    python -c "from huggingface_hub import hf_hub_download; \
        hf_hub_download('Plachta/Seed-VC', 'DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth', cache_dir='/app/checkpoints'); \
        hf_hub_download('Plachta/Seed-VC', 'config_dit_mel_seed_uvit_whisper_base_f0_44k.yml', cache_dir='/app/checkpoints'); \
        hf_hub_download('lj1995/VoiceConversionWebUI', 'rmvpe.pt', cache_dir='/app/checkpoints')" && \
    python -c "from transformers import AutoModel; AutoModel.from_pretrained('openai/whisper-small')" || true

# Source last so the model layers above stay cached across handler edits.
COPY . /app

RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
