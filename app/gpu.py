import os
import subprocess

from loguru import logger

from app.config import settings


def _gpu_free_gb_from_nvidia_smi() -> list[tuple[int, float]]:
    """Return (gpu_index, free_vram_gb) for each visible GPU."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        logger.warning(f"nvidia-smi failed ({error}); GPU auto-pick skipped")
        return []

    gpus: list[tuple[int, float]] = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        index, free_mb = (part.strip() for part in line.split(",", 1))
        gpus.append((int(index), int(free_mb) / 1024))
    return gpus


def configure_cuda_devices() -> str:
    """
    Pick GPUs before torch initializes CUDA:
      - GPU0 free > threshold -> use GPU 0 only
      - else GPU1 free > threshold -> use GPU 1 only
      - else -> use GPU 0 and 1 (device_map=auto will split)
    """
    threshold_gb = settings.gpu_free_gb_threshold
    preset = os.environ.get("CUDA_VISIBLE_DEVICES")

    if preset is not None and preset.strip() != "":
        logger.info(f"CUDA_VISIBLE_DEVICES preset (unchanged): {preset!r}")
        return f"preset:{preset}"

    gpus = _gpu_free_gb_from_nvidia_smi()
    if not gpus:
        logger.info("No GPU info; leaving CUDA_VISIBLE_DEVICES unset")
        return "unset"

    if len(gpus) == 1:
        index = str(gpus[0][0])
        os.environ["CUDA_VISIBLE_DEVICES"] = index
        logger.info(
            f"Single GPU visible: index={index}, free={gpus[0][1]:.1f}GB "
            f"-> CUDA_VISIBLE_DEVICES={index}"
        )
        return f"single:{index}"

    free = {index: gb for index, gb in gpus}
    free0 = free.get(0, 0.0)
    free1 = free.get(1, 0.0)

    if free0 > threshold_gb:
        devices = "0"
        reason = f"gpu0 free {free0:.1f}GB > {threshold_gb}GB"
    elif free1 > threshold_gb:
        devices = "1"
        reason = f"gpu1 free {free1:.1f}GB > {threshold_gb}GB"
    else:
        devices = "0,1"
        reason = (
            f"both <= {threshold_gb}GB (gpu0={free0:.1f}GB, gpu1={free1:.1f}GB) -> split"
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    logger.info(f"GPU pick: {reason} -> CUDA_VISIBLE_DEVICES={devices}")
    return reason
