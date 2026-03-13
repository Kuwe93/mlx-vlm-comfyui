try:
    from .MLX_VLM.MLX_VLM import (
        HAS_VLM,
        MfluxVLMLoader,
        MfluxVLMRun,
        MfluxVLMRunMulti,
    )
except ImportError:
    HAS_VLM = False
    print("[MLX-VLM] MLX_VLM.py not found or mlx-vlm not installed – nodes disabled. "
          "Install with: pip install mlx-vlm")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if HAS_VLM:
    NODE_CLASS_MAPPINGS = {
        "MfluxVLMLoader":   MfluxVLMLoader,
        "MfluxVLMRun":      MfluxVLMRun,
        "MfluxVLMRunMulti": MfluxVLMRunMulti,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        "MfluxVLMLoader":   "MLX VLM Loader",
        "MfluxVLMRun":      "MLX VLM Run",
        "MfluxVLMRunMulti": "MLX VLM Run (Multi-Image)",
    }
