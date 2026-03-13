import os
import tempfile
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# mlx-vlm – defensiver Import
# ---------------------------------------------------------------------------
try:
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    HAS_VLM = True
except ImportError:
    HAS_VLM = False
    print("[MfluxVLM] mlx-vlm not available – VLM nodes disabled. "
          "Install with: pip install mlx-vlm")

# ---------------------------------------------------------------------------
# Bekannte mlx-community Modelle für das Dropdown
# ---------------------------------------------------------------------------
VLM_MODELS = [
    "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    "mlx-community/Qwen2-VL-7B-Instruct-4bit",
    "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    "mlx-community/SmolVLM-Instruct",
    "mlx-community/SmolVLM2-2.2B-Instruct-mlx",
    "mlx-community/Florence-2-large-ft",
    "mlx-community/paligemma2-3b-mix-448-8bit",
    "vikhyatk/moondream2",
]

# ---------------------------------------------------------------------------
# Florence2-Tasks – identisch mit kijai/ComfyUI-Florence2
# ---------------------------------------------------------------------------
FLORENCE2_TASKS = [
    "more_detailed_caption",
    "detailed_caption",
    "caption",
    "ocr",
    "ocr_with_region",
    "object_detection",
    "dense_region_caption",
    "region_proposal",
    "caption_to_phrase_grounding",
    "open_vocabulary_detection",
    "referring_expression_segmentation",
    "region_to_segmentation",
    "region_to_category",
    "region_to_description",
    "region_to_ocr",
    "mixed_caption",
    "generate_tags",
]

FLORENCE2_TASK_MAP = {
    "more_detailed_caption":             "<MORE_DETAILED_CAPTION>",
    "detailed_caption":                  "<DETAILED_CAPTION>",
    "caption":                           "<CAPTION>",
    "ocr":                               "<OCR>",
    "ocr_with_region":                   "<OCR_WITH_REGION>",
    "object_detection":                  "<OD>",
    "dense_region_caption":              "<DENSE_REGION_CAPTION>",
    "region_proposal":                   "<REGION_PROPOSAL>",
    "caption_to_phrase_grounding":       "<CAPTION_TO_PHRASE_GROUNDING>",
    "open_vocabulary_detection":         "<OPEN_VOCABULARY_DETECTION>",
    "referring_expression_segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
    "region_to_segmentation":            "<REGION_TO_SEGMENTATION>",
    "region_to_category":                "<REGION_TO_CATEGORY>",
    "region_to_description":             "<REGION_TO_DESCRIPTION>",
    "region_to_ocr":                     "<REGION_TO_OCR>",
    "mixed_caption":                     "<MIX_CAPTION>",
    "generate_tags":                     "<GENERATE_TAGS>",
}

# Tasks die zusätzlich einen text_input unterstützen
FLORENCE2_TASKS_WITH_TEXT = {
    "caption_to_phrase_grounding",
    "referring_expression_segmentation",
    "open_vocabulary_detection",
    "region_to_segmentation",
    "region_to_category",
    "region_to_description",
    "region_to_ocr",
}

# ---------------------------------------------------------------------------
# Allgemeine Prompt-Presets für alle anderen VLMs
# ---------------------------------------------------------------------------
GENERAL_PRESETS = [
    "Detailed Caption (for img2img)",
    "Stable Diffusion Style Prompt",
    "Short Caption",
    "Object Detection",
    "OCR - Read Text",
    "Custom",
]

GENERAL_PRESET_MAP = {
    "Detailed Caption (for img2img)":
        "Describe this image in detail. Focus on the subject, style, lighting, colors, and composition. "
        "Write it as a single descriptive paragraph suitable as a generation prompt.",
    "Stable Diffusion Style Prompt":
        "Describe this image as a Stable Diffusion prompt. Include subject, art style, lighting, "
        "colors, quality tags. Comma-separated keywords.",
    "Short Caption":      "Describe this image in one sentence.",
    "Object Detection":   "List all objects and their positions visible in this image.",
    "OCR - Read Text":    "Read and transcribe all text visible in this image.",
    "Custom":             "",
}


def _is_florence2(model_path: str) -> bool:
    return "florence" in model_path.lower()


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor.squeeze(0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _tensor_to_temp_path(tensor: torch.Tensor) -> str:
    img = _tensor_to_pil(tensor)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
# Model-Cache
# ---------------------------------------------------------------------------
_vlm_cache: dict = {}


def _load_vlm(model_path: str):
    if model_path in _vlm_cache:
        print(f"[MfluxVLM] Using cached model: {model_path}")
        return _vlm_cache[model_path]
    print(f"[MfluxVLM] Loading model: {model_path}")
    _vlm_cache.clear()
    model, processor = load(model_path)
    config = load_config(model_path)
    _vlm_cache[model_path] = (model, processor, config)
    return model, processor, config


class MfluxVLMPipeline:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.is_florence2 = _is_florence2(model_path)
        self._model = None
        self._processor = None
        self._config = None

    def get(self):
        if self._model is None:
            self._model, self._processor, self._config = _load_vlm(self.model_path)
        return self._model, self._processor, self._config


# ---------------------------------------------------------------------------
# Node: MfluxVLMLoader
# ---------------------------------------------------------------------------
class MfluxVLMLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (VLM_MODELS, {
                    "default": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
                    "tooltip": "VLM model from mlx-community. Downloaded automatically on first use.",
                }),
            },
            "optional": {
                "custom_model_path": ("STRING", {
                    "default": "",
                    "tooltip": "Override: HuggingFace repo ID or local path. Leerlassen um das Dropdown zu nutzen.",
                }),
            },
        }

    RETURN_TYPES = ("MLXVLM_MODEL",)
    RETURN_NAMES = ("vlm_model",)
    CATEGORY = "MFlux/VLM"
    FUNCTION = "load"

    def load(self, model, custom_model_path=""):
        if not HAS_VLM:
            raise RuntimeError("mlx-vlm is not installed. Run: pip install mlx-vlm")
        model_path = custom_model_path.strip() if custom_model_path.strip() else model
        return (MfluxVLMPipeline(model_path),)


# ---------------------------------------------------------------------------
# Node: MfluxVLMRun
# Florence2: task-Dropdown mit <TASK>-Syntax (wie kijai)
# Andere VLMs: preset-Dropdown mit freien Prompts
# ---------------------------------------------------------------------------
class MfluxVLMRun:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vlm_model": ("MLXVLM_MODEL",),
                "image":     ("IMAGE",),
                "task": (FLORENCE2_TASKS, {
                    "default": "more_detailed_caption",
                    "tooltip": "Florence2 task. Nur aktiv wenn ein Florence2-Modell geladen ist.",
                }),
                "preset": (GENERAL_PRESETS, {
                    "default": "Detailed Caption (for img2img)",
                    "tooltip": "Prompt-Preset fuer Qwen2-VL, PaliGemma, SmolVLM usw. Wird ignoriert wenn Florence2 geladen ist.",
                }),
                "max_tokens":  ("INT",   {"default": 300, "min": 50,  "max": 2000, "step": 50}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0,  "step": 0.05,
                                          "tooltip": "0.0 = deterministisch. Fuer Florence2 irrelevant."}),
            },
            "optional": {
                "text_input": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Florence2: Nur fuer Tasks mit Texteingabe (caption_to_phrase_grounding etc.). "
                        "Andere VLMs: Ueberschreibt das Preset komplett."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    CATEGORY = "MFlux/VLM"
    FUNCTION = "run"

    def run(self, vlm_model, image, task, preset, max_tokens, temperature,
            text_input=""):
        if not HAS_VLM:
            raise RuntimeError("mlx-vlm is not installed. Run: pip install mlx-vlm")

        image_path = _tensor_to_temp_path(image)

        try:
            model, processor, config = vlm_model.get()

            if vlm_model.is_florence2:
                task_token = FLORENCE2_TASK_MAP.get(task, "<MORE_DETAILED_CAPTION>")
                if text_input and text_input.strip() and task in FLORENCE2_TASKS_WITH_TEXT:
                    prompt = task_token + " " + text_input.strip()
                else:
                    prompt = task_token
            else:
                if text_input and text_input.strip():
                    prompt = text_input.strip()
                else:
                    prompt = GENERAL_PRESET_MAP.get(preset, "Describe this image in detail.")
                    if not prompt:
                        prompt = "Describe this image in detail."

            print(f"[MfluxVLM] Prompt: {prompt}")

            formatted_prompt = apply_chat_template(
                processor, config, prompt, num_images=1
            )

            output = generate(
                model, processor, formatted_prompt,
                [image_path],
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=False,
            )

            result = output.strip()
            print(f"[MfluxVLM] Result: {result[:120]}{'...' if len(result) > 120 else ''}")
            return (result,)

        finally:
            try:
                os.unlink(image_path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Node: MfluxVLMRunMulti
# ---------------------------------------------------------------------------
class MfluxVLMRunMulti:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vlm_model": ("MLXVLM_MODEL",),
                "image1":    ("IMAGE",),
                "task": (FLORENCE2_TASKS, {
                    "default": "more_detailed_caption",
                }),
                "preset": (GENERAL_PRESETS, {
                    "default": "Detailed Caption (for img2img)",
                }),
                "max_tokens":  ("INT",   {"default": 500, "min": 50,  "max": 2000, "step": 50}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0,  "step": 0.05}),
            },
            "optional": {
                "image2":     ("IMAGE",),
                "image3":     ("IMAGE",),
                "text_input": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    CATEGORY = "MFlux/VLM"
    FUNCTION = "run"

    def run(self, vlm_model, image1, task, preset, max_tokens, temperature,
            image2=None, image3=None, text_input=""):
        if not HAS_VLM:
            raise RuntimeError("mlx-vlm is not installed. Run: pip install mlx-vlm")

        images_tensors = [t for t in [image1, image2, image3] if t is not None]
        image_paths = [_tensor_to_temp_path(t) for t in images_tensors]

        try:
            model, processor, config = vlm_model.get()

            if vlm_model.is_florence2:
                task_token = FLORENCE2_TASK_MAP.get(task, "<MORE_DETAILED_CAPTION>")
                prompt = (task_token + " " + text_input.strip()
                          if text_input and text_input.strip() and task in FLORENCE2_TASKS_WITH_TEXT
                          else task_token)
            else:
                if text_input and text_input.strip():
                    prompt = text_input.strip()
                else:
                    prompt = GENERAL_PRESET_MAP.get(preset, "Describe these images in detail.")
                    if not prompt:
                        prompt = "Describe these images in detail."

            print(f"[MfluxVLM] Multi-image prompt: {prompt} ({len(image_paths)} images)")

            formatted_prompt = apply_chat_template(
                processor, config, prompt, num_images=len(image_paths)
            )

            output = generate(
                model, processor, formatted_prompt,
                image_paths,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=False,
            )

            return (output.strip(),)

        finally:
            for p in image_paths:
                try:
                    os.unlink(p)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Node: MfluxVLMConverter
# Quantisiert ein HuggingFace-Modell und speichert es lokal –
# analog zu MfluxCustomModels in mflux-comfyui-2
# Ruft `python -m mlx_vlm.convert` als Subprozess auf
# ---------------------------------------------------------------------------
import sys
import subprocess

QUANTIZE_BITS = ["4", "8", "3", "6"]


class MfluxVLMConverter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hf_model": ("STRING", {
                    "default": "microsoft/Florence-2-large",
                    "tooltip": "HuggingFace repo ID (z.B. 'Qwen/Qwen2.5-VL-3B-Instruct') oder lokaler Pfad (z.B. '/Volumes/KI/models/Florence-2-large').",
                }),
                "output_path": ("STRING", {
                    "default": "~/models/",
                    "tooltip": "Lokaler Zielordner. Der Modellname wird automatisch angehängt, z.B. ~/models/Florence-2-large-4bit.",
                }),
                "q_bits": (QUANTIZE_BITS, {
                    "default": "4",
                    "tooltip": "Quantisierungstiefe in Bit. 4-bit empfohlen für maximale Kompression, 8-bit für höhere Qualität.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    CATEGORY = "MFlux/VLM"
    FUNCTION = "convert"
    OUTPUT_NODE = True

    def convert(self, hf_model, output_path, q_bits):
        if not HAS_VLM:
            raise RuntimeError("mlx-vlm is not installed. Run: pip install mlx-vlm")

        hf_model = hf_model.strip()
        output_path = output_path.strip()

        if not hf_model:
            raise ValueError("hf_model darf nicht leer sein.")
        if not output_path:
            raise ValueError("output_path darf nicht leer sein.")

        # Zielpfad zusammenbauen: ~/models/ + "Florence-2-large-4bit"
        model_name = hf_model.split("/")[-1]
        save_name = f"{model_name}-{q_bits}bit"
        save_path = os.path.join(os.path.expanduser(output_path), save_name)

        print(f"[MfluxVLM] Converting '{hf_model}' → '{save_path}' ({q_bits}-bit) ...")

        cmd = [
            sys.executable, "-m", "mlx_vlm.convert",
            "--hf-path",  hf_model,
            "--mlx-path", save_path,
            "--quantize",
            "--q-bits",   q_bits,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=False,   # Output live im ComfyUI-Terminal sichtbar
                text=True,
                check=True,
            )
            print(f"[MfluxVLM] Conversion complete. Model saved to: {save_path}")
            return (save_path,)

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"mlx_vlm.convert failed (exit {e.returncode}). "
                f"Check the ComfyUI terminal for details."
            ) from e