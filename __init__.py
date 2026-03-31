try:
    from .MLX_VLM.MLX_VLM import (
        HAS_VLM,
        MfluxVLMLoader,
        MfluxVLMRun,
        MfluxVLMRunMulti,
        MfluxVLMConverter,
        MfluxVLMBatchCaption,
        VLMBatchCaptionCharacterLoRA,
        VLMDatasetCurator,
        VLMCornerInpainter,
        VLMCuratorReview,
        VLMModelUnloader,
        VLMCornerInpainterBatch,
        VLMPromptBuilder,
        # Neue Nodes
        VLMTextAnalyzer,
        VLMImageCompare,
        VLMQualityScorer,
        VLMBatchQualityFilter,
        VLMFaceDetector,
        VLMCaptionRefiner,
        VLMModelInfo,
    )
except ImportError:
    HAS_VLM = False
    print("[MLX-VLM] MLX_VLM.py not found or mlx-vlm not installed – nodes disabled. "
          "Install with: pip install mlx-vlm")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if HAS_VLM:
    NODE_CLASS_MAPPINGS = {
        # ── Core ────────────────────────────────────────────────────────────
        "MfluxVLMLoader":               MfluxVLMLoader,
        "MfluxVLMRun":                  MfluxVLMRun,
        "MfluxVLMRunMulti":             MfluxVLMRunMulti,
        "MfluxVLMConverter":            MfluxVLMConverter,
        # ── Batch Captioning ────────────────────────────────────────────────
        "MfluxVLMBatchCaption":         MfluxVLMBatchCaption,
        "VLMBatchCaptionCharacterLoRA": VLMBatchCaptionCharacterLoRA,
        # ── Dataset Curation ────────────────────────────────────────────────
        "VLMDatasetCurator":            VLMDatasetCurator,
        "VLMCuratorReview":             VLMCuratorReview,
        # ── Corner Inpainting ───────────────────────────────────────────────
        "VLMCornerInpainter":           VLMCornerInpainter,
        "VLMCornerInpainterBatch":      VLMCornerInpainterBatch,
        # ── Utilities ───────────────────────────────────────────────────────
        "VLMModelUnloader":             VLMModelUnloader,
        "VLMPromptBuilder":             VLMPromptBuilder,
        # ── Neue Nodes ──────────────────────────────────────────────────────
        "VLMTextAnalyzer":              VLMTextAnalyzer,
        "VLMImageCompare":              VLMImageCompare,
        "VLMQualityScorer":             VLMQualityScorer,
        "VLMBatchQualityFilter":        VLMBatchQualityFilter,
        "VLMFaceDetector":              VLMFaceDetector,
        "VLMCaptionRefiner":            VLMCaptionRefiner,
        "VLMModelInfo":                 VLMModelInfo,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        # ── Core ────────────────────────────────────────────────────────────
        "MfluxVLMLoader":               "MLX VLM Loader",
        "MfluxVLMRun":                  "MLX VLM Run",
        "MfluxVLMRunMulti":             "MLX VLM Run (Multi-Image)",
        "MfluxVLMConverter":            "MLX VLM Converter",
        # ── Batch Captioning ────────────────────────────────────────────────
        "MfluxVLMBatchCaption":         "VLM Batch Caption",
        "VLMBatchCaptionCharacterLoRA": "VLM Batch Caption (LoRA Character)",
        # ── Dataset Curation ────────────────────────────────────────────────
        "VLMDatasetCurator":            "VLM Dataset Curator",
        "VLMCuratorReview":             "VLM Curator Review (Second Pass)",
        # ── Corner Inpainting ───────────────────────────────────────────────
        "VLMCornerInpainter":           "VLM Corner Inpainter",
        "VLMCornerInpainterBatch":      "VLM Corner Inpainter Batch",
        # ── Utilities ───────────────────────────────────────────────────────
        "VLMModelUnloader":             "VLM Model Unloader",
        "VLMPromptBuilder":             "VLM Prompt Builder",
        # ── Neue Nodes ──────────────────────────────────────────────────────
        "VLMTextAnalyzer":              "VLM Text Analyzer (OCR+)",
        "VLMImageCompare":              "VLM Image Compare",
        "VLMQualityScorer":             "VLM Quality Scorer",
        "VLMBatchQualityFilter":        "VLM Batch Quality Filter",
        "VLMFaceDetector":              "VLM Face Detector",
        "VLMCaptionRefiner":            "VLM Caption Refiner",
        "VLMModelInfo":                 "VLM Model Info",
    }