"""Environment detection and path resolution.

The same code runs on a local Windows box, a Kaggle notebook, or Colab.
Nothing else in the project should hardcode a path or a device.
"""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path


def detect_platform() -> str:
    """Return one of: 'kaggle', 'colab', 'local'."""
    if os.path.exists("/kaggle/working"):
        return "kaggle"
    if "COLAB_GPU" in os.environ or os.path.exists("/content"):
        return "colab"
    return "local"


@dataclass(frozen=True)
class Paths:
    root: Path
    data_raw: Path
    data_processed: Path
    results: Path
    checkpoints: Path
    logs: Path
    figures: Path
    cache: Path

    def mkdirs(self) -> None:
        for p in (self.data_raw, self.data_processed, self.results,
                  self.checkpoints, self.logs, self.figures, self.cache):
            p.mkdir(parents=True, exist_ok=True)


def get_paths() -> Paths:
    """Resolve project paths for the current platform.

    On Kaggle only /kaggle/working persists between sessions (20 GB), so
    outputs go there while the HF cache goes to /kaggle/temp to avoid
    burning that quota on model weights we can re-download.
    """
    plat = detect_platform()

    if plat == "kaggle":
        root = Path("/kaggle/working")
        cache = Path("/kaggle/temp/hf_cache")
    elif plat == "colab":
        root = Path("/content/bill-kie")
        cache = root / ".cache"
    else:
        root = Path(__file__).resolve().parents[2]
        cache = root / ".cache"

    results = root / "results"
    paths = Paths(
        root=root,
        data_raw=root / "data" / "raw",
        data_processed=root / "data" / "processed",
        results=results,
        checkpoints=results / "checkpoints",
        logs=results / "logs",
        figures=results / "figures",
        cache=cache,
    )
    paths.mkdirs()
    return paths


def get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def gpu_memory_gb() -> float:
    """Total VRAM of the current GPU in GB, or 0.0 if there is none."""
    import torch
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3


def recommended_batch_config(vram_gb: float | None = None) -> dict:
    """Pick batch size and sequence length from the available VRAM.

    Values for the <6 GB tier are measured, not estimated - see
    results/logs/vram_benchmark.json and scripts/benchmark_vram.py.

    Two hard constraints on small GPUs:

    * LayoutLMv3 does not support gradient checkpointing, so the usual
      activation-memory escape hatch is unavailable.
    * On Windows/WDDM, exceeding VRAM does not raise OutOfMemoryError; CUDA
      spills into system RAM and the step time collapses (measured: 11x
      slower going from batch 4 to batch 8). Configs here stay well under
      the physical limit for that reason.

    Effective batch size is held at 16 on every tier via gradient
    accumulation, so a 4 GB laptop run and a 16 GB cloud run remain
    directly comparable - changing it would change the optimisation
    trajectory and invalidate cross-machine comparison.
    """
    if vram_gb is None:
        vram_gb = gpu_memory_gb()

    target_effective = 16

    if vram_gb == 0:
        cfg = {"per_device_batch_size": 1, "max_seq_length": 256, "fp16": False}
    elif vram_gb < 6:
        # RTX 3050 4 GB: measured 2.64 GB peak, 9.4 samples/s.
        # batch 4 reaches the same throughput at 3.65 GB, leaving no
        # headroom for longer real batches, so batch 2 is the safer pick.
        cfg = {"per_device_batch_size": 2, "max_seq_length": 256, "fp16": True}
    elif vram_gb < 12:
        cfg = {"per_device_batch_size": 8, "max_seq_length": 512, "fp16": True}
    else:
        cfg = {"per_device_batch_size": 16, "max_seq_length": 512, "fp16": True}

    cfg["gradient_accumulation_steps"] = max(
        1, target_effective // cfg["per_device_batch_size"]
    )
    cfg["effective_batch_size"] = (
        cfg["per_device_batch_size"] * cfg["gradient_accumulation_steps"]
    )
    # Not a tunable: the architecture does not support it.
    cfg["gradient_checkpointing"] = False
    return cfg


def summary() -> dict:
    info = {
        "platform": detect_platform(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(gpu_memory_gb(), 2)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        info["torch"] = "not installed"
    return info


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=2))
    print("\nPaths:")
    for k, v in get_paths().__dict__.items():
        print(f"  {k:16s} {v}")
    print("\nRecommended batch config:")
    for k, v in recommended_batch_config().items():
        print(f"  {k:30s} {v}")
