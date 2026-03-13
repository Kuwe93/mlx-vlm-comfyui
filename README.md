<h1 align="center">MLX-VLM ComfyUI</h1>

<p align="center">
  ComfyUI nodes for Vision Language Models via <a href="https://github.com/Blaizzy/mlx-vlm">mlx-vlm</a> —
  natively optimized for Apple Silicon.
</p>

**Only for macOS (Apple Silicon).** Brings Florence2, Qwen2-VL, Qwen2.5-VL, SmolVLM, PaliGemma and more into ComfyUI — running natively on MLX without PyTorch overhead.

> NOTE: This extension is independent of mflux. It can be used standalone or together with [mflux-comfyui-2](https://github.com/Kuwe93/mflux-comfyui-2) for captioning-to-generation workflows.

---

## ✨ Features

- **Florence2 task support** — all `<TASK>` prompts from the Florence2 spec, identical to [kijai/ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2), including MiaoshouAI PromptGen tasks (`mixed_caption`, `generate_tags`)
- **General VLM presets** — ready-made prompts for Qwen2-VL, SmolVLM, PaliGemma and others
- **Multi-image support** — pass up to 3 images simultaneously (for models that support it, e.g. Qwen2-VL)
- **Model caching** — models stay in memory between runs, no reloading on each generation
- **Fully optional** — if `mlx-vlm` is not installed, ComfyUI starts normally without errors

---

## Installation

```bash
cd /path/to/your_ComfyUI/custom_nodes
git clone https://github.com/Kuwe93/mlx-vlm-comfyui.git
pip install mlx-vlm
# Restart ComfyUI
```

Or search for **"MLX-VLM"** in ComfyUI-Manager.

---

## Node overview

All nodes can be found by double-clicking the canvas and searching for **"MLX VLM"**.

### MLX/VLM

| Node | Description |
|---|---|
| **MLX VLM Loader** | Loads a VLM model from mlx-community or a custom HuggingFace repo / local path. Auto-detects Florence2 models. |
| **MLX VLM Run** | Analyzes an image and returns a text string. Florence2: `task` dropdown with `<TASK>` syntax. Other VLMs: `preset` dropdown with free-text prompts. |
| **MLX VLM Run (Multi-Image)** | Like MLX VLM Run, but accepts up to 3 images simultaneously. |

---

## Supported models

The **MLX VLM Loader** includes a dropdown with tested models from mlx-community. Any other mlx-vlm compatible model can be loaded via the `custom_model_path` field.

| Model | Size | Notes |
|---|---|---|
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | ~1.5 GB | Fast, good quality |
| `mlx-community/Qwen2-VL-7B-Instruct-4bit` | ~5 GB | Higher quality, multilingual |
| `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` | ~2 GB | Newer generation |
| `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` | ~5 GB | Newer generation, high quality |
| `mlx-community/SmolVLM-Instruct` | ~0.5 GB | Very lightweight, fast |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx` | ~1.5 GB | Improved SmolVLM |
| `mlx-community/Florence-2-large-ft` | ~1.5 GB | Specialized for image analysis tasks |
| `mlx-community/paligemma2-3b-mix-448-8bit` | ~3 GB | Google's compact VLM |
| `vikhyatk/moondream2` | ~0.5 GB | Ultra-lightweight |

Models are downloaded automatically from HuggingFace on first use and cached locally.

---

## Florence2 tasks

When a Florence2 model is loaded, the **MLX VLM Run** node uses the `task` dropdown instead of the `preset` dropdown. All standard Florence2 tasks are supported:

| Task | Description |
|---|---|
| `more_detailed_caption` | Detailed image description (recommended for img2img workflows) |
| `detailed_caption` | Medium-detail description |
| `caption` | Short one-sentence description |
| `ocr` | Extract all text from the image |
| `ocr_with_region` | Extract text with bounding box coordinates |
| `object_detection` | Detect and localize objects |
| `dense_region_caption` | Caption for every detected region |
| `region_proposal` | Propose regions of interest |
| `caption_to_phrase_grounding` | Localize phrases from a caption (`text_input` required) |
| `open_vocabulary_detection` | Detect arbitrary objects by name (`text_input` required) |
| `referring_expression_segmentation` | Segment a region described in text (`text_input` required) |
| `region_to_segmentation` | Segmentation mask for a region |
| `region_to_category` | Classify a region |
| `region_to_description` | Describe a region |
| `region_to_ocr` | Extract text from a region |
| `mixed_caption` *(PromptGen)* | Optimized caption for image generation prompts |
| `generate_tags` *(PromptGen)* | Generate comma-separated tags for image generation |

> `mixed_caption` and `generate_tags` require a PromptGen fine-tune such as [MiaoshouAI/Florence-2-large-PromptGen-v2.0](https://huggingface.co/MiaoshouAI/Florence-2-large-PromptGen-v2.0).

Tasks marked with `text_input` required use the optional `text_input` field on the node. For all other Florence2 tasks this field is ignored.

---

## Example workflows

### Caption → MFlux Generation

Load an image → describe it with **MLX VLM Run** → pipe the caption STRING directly into **Quick MFlux Generation** as the prompt. No manual prompt writing needed.

```
[LoadImage] → [MLX VLM Run]  →  caption  →  [Quick MFlux Generation]
                   ↑
          [MLX VLM Loader]
```

A ready-made workflow JSON (`MLX_VLM___Mflux.json`) is included in the `workflows/` folder.

### PromptGen for batch captioning

Use `generate_tags` with a MiaoshouAI PromptGen model to automatically generate Stable Diffusion / FLUX-style tag prompts from reference photos.

---

## Tips

- **Which model to start with?** `Qwen2-VL-2B-Instruct-4bit` is a good default — fast, small (~1.5 GB), and handles most captioning tasks well.
- **For Stable Diffusion / FLUX prompts from photos:** Use Florence2 with `mixed_caption` (requires MiaoshouAI PromptGen fine-tune) or Qwen2-VL with the "Stable Diffusion Style Prompt" preset.
- **Florence2 vs. Qwen2-VL:** Florence2 is faster and more specialized for structured vision tasks (OCR, detection, segmentation). Qwen2-VL gives more natural, detailed descriptions and supports follow-up questions via `text_input`.
- **temperature:** Only relevant for non-Florence2 models. `0.0` gives deterministic, consistent results — useful for batch workflows.
- **Multi-image node:** Useful for comparison captions or when you want one description covering multiple views of the same subject.

---

## Credits

- [@Blaizzy](https://github.com/Blaizzy) for [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)
- [@kijai](https://github.com/kijai) for [ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2) — the Florence2 task list and node design were used as reference
- [@apple/ml-explore](https://github.com/ml-explore) for the [MLX framework](https://github.com/ml-explore/mlx)
