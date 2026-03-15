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
# Hilfsfunktion: generate()-Output normalisieren
# mlx-vlm >= 0.3.x gibt GenerationResult zurück, ältere Versionen einen String
# ---------------------------------------------------------------------------
def _extract_text(output) -> str:
    """Extrahiert den Text aus mlx-vlm generate() Output unabhängig vom Typ."""
    if isinstance(output, str):
        return output.strip()
    # GenerationResult oder ähnliches Objekt
    for attr in ("text", "generated_text", "content", "output"):
        if hasattr(output, attr):
            val = getattr(output, attr)
            if isinstance(val, str):
                return val.strip()
    # Fallback: str() Konvertierung
    return str(output).strip()


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

            result = _extract_text(output)
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

            return (_extract_text(output),)

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


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def _vlm_batch_run(tag, vlm_model, folder, prompt, trigger,
                   max_tokens, temperature, overwrite, reload_every):
    image_files = sorted([
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS
        and not os.path.splitext(f)[0].lower().startswith("preview")
    ])
    if not image_files:
        raise ValueError(f"[{tag}] Keine Bilddateien gefunden in: {folder}")
    print(f"[{tag}] {len(image_files)} Bilder gefunden in: {folder}")
    model, processor, config = vlm_model.get()
    processed = 0
    skipped   = 0
    results   = []
    since_reload = 0
    for fname in image_files:
        img_path = os.path.join(folder, fname)
        base     = os.path.splitext(fname)[0]
        txt_path = os.path.join(folder, base + ".txt")
        if os.path.exists(txt_path) and not overwrite:
            print(f"[{tag}] Ueberspringe: {fname}")
            skipped += 1
            continue
        print(f"[{tag}] [{processed + skipped + 1}/{len(image_files)}]: {fname}")
        if since_reload > 0 and since_reload % reload_every == 0:
            print(f"[{tag}] Lade Modell neu ...")
            vlm_model._model = None
            model, processor, config = vlm_model.get()
        try:
            fp = apply_chat_template(processor, config, prompt, num_images=1)
            out = generate(model, processor, fp, [img_path],
                           max_tokens=max_tokens, temperature=temperature, verbose=False)
            caption = _extract_text(out)
            if trigger:
                caption = trigger + ", " + caption
            with open(txt_path, "w", encoding="utf-8") as fh:
                fh.write(caption)
            print(f"[{tag}]   -> {caption[:80]}")
            results.append(fname + ": " + caption[:60] + "...")
            processed += 1
            since_reload += 1
        except Exception as e:
            print(f"[{tag}] FEHLER bei {fname}: {e}")
            results.append(fname + ": FEHLER - " + str(e))
    summary_lines = [
        "Batch: " + str(processed) + " beschriftet, " + str(skipped) + " uebersprungen.",
        "Ordner: " + folder, "",
    ] + results[:20] + (["..."] if len(results) > 20 else [])
    summary = "\n".join(summary_lines)
    print(f"[{tag}] Fertig.")
    return (summary, processed, skipped)


if HAS_VLM:
    class MfluxVLMBatchCaption:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "vlm_model":    ("MLXVLM_MODEL",),
                    "image_folder": ("STRING", {"default": "/path/to/images"}),
                    "task":    (FLORENCE2_TASKS,  {"default": "more_detailed_caption"}),
                    "preset":  (GENERAL_PRESETS,  {"default": "Detailed Caption (for img2img)"}),
                    "max_tokens":   ("INT",   {"default": 300, "min": 50, "max": 2000, "step": 50}),
                    "temperature":  ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                    "overwrite":    ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
                    "reload_every": ("INT", {"default": 10, "min": 1, "max": 100}),
                },
                "optional": {
                    "custom_prompt": ("STRING", {"multiline": True, "default": ""}),
                },
            }
        RETURN_TYPES = ("STRING", "INT", "INT")
        RETURN_NAMES = ("summary", "processed", "skipped")
        CATEGORY     = "MFlux/VLM"
        FUNCTION     = "run_batch"
        OUTPUT_NODE  = True

        def run_batch(self, vlm_model, image_folder, task, preset,
                      max_tokens, temperature, overwrite, reload_every, custom_prompt=""):
            folder = os.path.expanduser(image_folder.strip())
            if not os.path.isdir(folder):
                raise ValueError("[VLMBatch] Ordner nicht gefunden: " + folder)
            if vlm_model.is_florence2:
                prompt = FLORENCE2_TASK_MAP.get(task, "<MORE_DETAILED_CAPTION>")
            else:
                prompt = (custom_prompt.strip() if custom_prompt and custom_prompt.strip()
                          else GENERAL_PRESET_MAP.get(preset, "Describe this image in detail."))
            return _vlm_batch_run("VLMBatch", vlm_model, folder, prompt,
                                  trigger="", max_tokens=max_tokens,
                                  temperature=temperature, overwrite=overwrite,
                                  reload_every=reload_every)


LORA_PROMPT = (
    "You are a captioning assistant for AI image model training data. "
    "Analyze the image and generate a structured training caption. "
    "STRUCTURE: 1. Character name + hair color + facial features "
    "2. Shot type: close-up / upper body / full body "
    "3. Pose, body language, camera angle "
    "4. Clothing with colors and style details "
    "5. Facial expression precisely described "
    "6. Background type and lighting. "
    "RULES: Start with {CHARACTER}. Comma-separated phrases only. "
    "No full sentences, no articles. 50-70 words total."
)

if HAS_VLM:
    class VLMBatchCaptionCharacterLoRA:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "vlm_model":      ("MLXVLM_MODEL",),
                    "image_folder":   ("STRING", {"default": "/path/to/training/images"}),
                    "character_name": ("STRING", {
                        "default": "ohwx person",
                        "tooltip": "Trigger-Token. Ersetzt {CHARACTER} im Prompt.",
                    }),
                    "max_tokens":   ("INT",   {"default": 150, "min": 50, "max": 500, "step": 10}),
                    "temperature":  ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                    "overwrite":    ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
                    "reload_every": ("INT", {"default": 10, "min": 1, "max": 100}),
                },
                "optional": {
                    "custom_prompt": ("STRING", {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Eigener Prompt mit {CHARACTER} Platzhalter. Leer = Standard.",
                    }),
                },
            }
        RETURN_TYPES = ("STRING", "INT", "INT")
        RETURN_NAMES = ("summary", "processed", "skipped")
        CATEGORY     = "MFlux/VLM"
        FUNCTION     = "run_batch"
        OUTPUT_NODE  = True

        def run_batch(self, vlm_model, image_folder, character_name,
                      max_tokens, temperature, overwrite, reload_every, custom_prompt=""):
            folder = os.path.expanduser(image_folder.strip())
            if not os.path.isdir(folder):
                raise ValueError("[VLMBatchCharacterLoRA] Ordner nicht gefunden: " + folder)
            char = character_name.strip()
            if not char:
                raise ValueError("[VLMBatchCharacterLoRA] character_name darf nicht leer sein.")
            base = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else LORA_PROMPT
            prompt = base.replace("{CHARACTER}", char)
            return _vlm_batch_run("VLMBatchCharacterLoRA", vlm_model, folder, prompt,
                                  trigger=char, max_tokens=max_tokens,
                                  temperature=temperature, overwrite=overwrite,
                                  reload_every=reload_every)


# ---------------------------------------------------------------------------
# Node: VLMDatasetCurator
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Node: VLMDatasetCurator
# Analysiert Bilder mit einem VLM und:
# - Erkennt Kategorie (close-up / upper_body / full_body / back_side)
# - Erkennt Emotion
# - Erkennt Blickwinkel
# - Bewertet Qualität (approved / rejected)
# - Sortiert Bilder in Unterordner
# - Trackt Dataset-Slots gegen V1/V2 Schema
# - Generiert JSON-Report
# ---------------------------------------------------------------------------

CURATOR_ANALYSIS_PROMPT = """Analyze this portrait image for AI training dataset curation.
Respond with ONLY a JSON object, no other text:
{
  "approved": true or false,
  "reject_reason": "reason if rejected, else null",
  "category": "close_up" or "upper_body" or "full_body" or "back_side",
  "emotion": "neutral" or "joy" or "anger" or "fear" or "sadness" or "surprise" or "excited" or "thoughtful" or "confident" or "relaxed" or "disgust" or "contempt",
  "angle": "frontal" or "three_quarter_left" or "three_quarter_right" or "side_left" or "side_right" or "from_above" or "from_below" or "back",
  "hair_visible": true or false,
  "face_visible": true or false,
  "face_quality": "sharp" or "blurry" or "partially_occluded",
  "notes": "brief observation or null"
}

APPROVAL CRITERIA (reject if any fail):
- Exactly one person clearly visible
- No heavy blur or motion blur on face
- No strong obstruction of face (hands, objects)
- Image is not heavily cropped or distorted
- Minimum resolution appears acceptable

CATEGORY DEFINITIONS:
- close_up: face and neck only, no shoulders or minimal shoulders
- upper_body: from waist or chest upward including face
- full_body: entire body visible head to toe
- back_side: back view or side view without face"""

# Dataset-Slot-Schema V1 (30 Bilder)
DATASET_SCHEMA_V1 = {
    "close_up": {
        "total": 12,
        "slots": [
            {"angle": "frontal",            "emotion": "neutral",    "required": True},
            {"angle": "frontal",            "emotion": "joy",        "required": True},
            {"angle": "three_quarter_left", "emotion": "anger",      "required": True},
            {"angle": "three_quarter_right","emotion": "fear",       "required": True},
            {"angle": "from_above",         "emotion": "neutral",    "required": False},
            {"angle": "from_below",         "emotion": "confident",  "required": False},
            {"angle": "side_left",          "emotion": "neutral",    "required": False},
            {"angle": "side_right",         "emotion": "sadness",    "required": True},
            {"angle": "frontal",            "emotion": "relaxed",    "required": False},
            {"angle": "three_quarter_right","emotion": "surprise",   "required": False},
            {"angle": "three_quarter_left", "emotion": "excited",    "required": False},
            {"angle": "from_above",         "emotion": "confident",  "required": False},
        ]
    },
    "upper_body": {"total": 8},
    "full_body":  {"total": 6},
    "back_side":  {"total": 4},
}

DATASET_SCHEMA_V2 = {
    "close_up":   {"total": 16},
    "upper_body": {"total": 10},
    "full_body":  {"total": 8},
    "back_side":  {"total": 6},
}


if HAS_VLM:
    class VLMDatasetCurator:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "vlm_model":    ("MLXVLM_MODEL",),
                    "image_folder": ("STRING", {
                        "default": "/path/to/candidate/images",
                        "tooltip": "Ordner mit Kandidatenbildern fuer das Training.",
                    }),
                    "dataset_version": (["V1 (30 Bilder)", "V2 (40 Bilder)", "Nur Qualitaetspruefung"], {
                        "default": "V1 (30 Bilder)",
                        "tooltip": "Schema gegen das geprueft wird.",
                    }),
                    "move_files": ("BOOLEAN", {
                        "default": True,
                        "label_on": "True", "label_off": "False",
                        "tooltip": "True = Bilder in approved/ und rejected/ verschieben. "
                                   "False = nur analysieren ohne Dateien zu bewegen.",
                    }),
                    "overwrite_analysis": ("BOOLEAN", {
                        "default": False,
                        "label_on": "True", "label_off": "False",
                        "tooltip": "Bereits analysierte Bilder (vorhandene .json) erneut analysieren.",
                    }),
                    "reload_every": ("INT", {"default": 5, "min": 1, "max": 50,
                                    "tooltip": "Modell alle N Bilder neu laden gegen Repetition."}),
                },
                "optional": {
                    "custom_criteria": ("STRING", {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Zusaetzliche Kriterien die ans Analyse-Prompt angehaengt werden.",
                    }),
                },
            }

        RETURN_TYPES = ("STRING", "INT", "INT", "INT")
        RETURN_NAMES = ("report", "approved", "rejected", "needs_slots")
        CATEGORY     = "MFlux/VLM"
        FUNCTION     = "curate"
        OUTPUT_NODE  = True

        def curate(self, vlm_model, image_folder, dataset_version,
                   move_files, overwrite_analysis, reload_every,
                   custom_criteria=""):
            import json as _json
            import shutil

            folder = os.path.expanduser(image_folder.strip())
            if not os.path.isdir(folder):
                raise ValueError(f"[Curator] Ordner nicht gefunden: {folder}")

            # Unterordner anlegen
            approved_dir = os.path.join(folder, "approved")
            rejected_dir = os.path.join(folder, "rejected")
            analysis_dir = os.path.join(folder, ".analysis")
            for d in [approved_dir, rejected_dir, analysis_dir]:
                os.makedirs(d, exist_ok=True)

            image_files = sorted([
                f for f in os.listdir(folder)
                if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS
                and not os.path.splitext(f)[0].lower().startswith("preview")
            ])

            if not image_files:
                raise ValueError(f"[Curator] Keine Bilddateien in: {folder}")

            print(f"[Curator] {len(image_files)} Bilder gefunden.")

            # Prompt aufbauen
            prompt = CURATOR_ANALYSIS_PROMPT
            if custom_criteria and custom_criteria.strip():
                prompt += "\n\nADDITIONAL CRITERIA:\n" + custom_criteria.strip()

            model, processor, config = vlm_model.get()
            approved_list = []
            rejected_list = []
            analyses = []
            since_reload = 0

            for i, fname in enumerate(image_files, 1):
                img_path = os.path.join(folder, fname)
                base     = os.path.splitext(fname)[0]
                json_path = os.path.join(analysis_dir, base + ".json")

                # Vorhandene Analyse verwenden wenn vorhanden
                if os.path.exists(json_path) and not overwrite_analysis:
                    with open(json_path) as f:
                        analysis = _json.load(f)
                    print(f"[Curator] [{i}/{len(image_files)}] {fname} (gecacht)")
                else:
                    print(f"[Curator] [{i}/{len(image_files)}] Analysiere: {fname}")

                    if since_reload > 0 and since_reload % reload_every == 0:
                        print("[Curator] Lade Modell neu ...")
                        vlm_model._model = None
                        model, processor, config = vlm_model.get()

                    try:
                        fp = apply_chat_template(processor, config, prompt, num_images=1)
                        out = generate(model, processor, fp, [img_path],
                                      max_tokens=300, temperature=0.0, verbose=False)
                        raw = _extract_text(out)

                        # JSON aus Antwort extrahieren
                        raw = raw.strip()
                        if "```" in raw:
                            raw = raw.split("```")[1]
                            if raw.startswith("json"):
                                raw = raw[4:]
                        analysis = _json.loads(raw.strip())
                        analysis["filename"] = fname

                        # Analyse cachen
                        with open(json_path, "w") as f:
                            _json.dump(analysis, f, indent=2)

                        since_reload += 1

                    except Exception as e:
                        print(f"[Curator] FEHLER bei {fname}: {e}")
                        analysis = {
                            "filename": fname,
                            "approved": False,
                            "reject_reason": f"Analysis error: {str(e)}",
                            "category": "unknown",
                            "emotion": "unknown",
                            "angle": "unknown",
                            "face_visible": False,
                            "face_quality": "unknown",
                            "notes": None,
                        }

                analyses.append(analysis)
                is_approved = analysis.get("approved", False)

                cat      = analysis.get("category", "unknown")
                emotion  = analysis.get("emotion",  "unknown")
                angle    = analysis.get("angle",    "unknown")
                quality  = analysis.get("face_quality", "unknown")
                reason   = analysis.get("reject_reason", "")

                status = "✓" if is_approved else "✗"
                print(f"[Curator]   {status} {cat} | {emotion} | {angle} | {quality}"
                      + (f" | REJECT: {reason}" if not is_approved else ""))

                if is_approved:
                    approved_list.append(analysis)
                    if move_files and os.path.exists(img_path):
                        shutil.move(img_path, os.path.join(approved_dir, fname))
                else:
                    rejected_list.append(analysis)
                    if move_files and os.path.exists(img_path):
                        shutil.move(img_path, os.path.join(rejected_dir, fname))

            # ── Dataset-Slot-Analyse ────────────────────────────────────────
            slot_report = []
            needs_slots = 0

            if dataset_version != "Nur Qualitaetspruefung":
                schema = DATASET_SCHEMA_V1 if "V1" in dataset_version else DATASET_SCHEMA_V2

                # Zähle approved Bilder pro Kategorie
                counts = {"close_up": 0, "upper_body": 0, "full_body": 0, "back_side": 0}
                for a in approved_list:
                    cat = a.get("category", "unknown")
                    if cat in counts:
                        counts[cat] += 1

                slot_report.append(f"\nDataset-Slots ({dataset_version}):")
                for cat, spec in schema.items():
                    total    = spec["total"]
                    have     = counts.get(cat, 0)
                    missing  = max(0, total - have)
                    needs_slots += missing
                    bar = "█" * have + "░" * missing
                    slot_report.append(f"  {cat:12} [{bar}] {have}/{total}"
                                       + (" ✓" if missing == 0 else f" → {missing} fehlen"))

                # Pflicht-Emotionen prüfen (V1 close_up)
                if "V1" in dataset_version:
                    required_slots = [s for s in DATASET_SCHEMA_V1["close_up"]["slots"]
                                      if s.get("required")]
                    slot_report.append(f"\nPflicht-Emotionen (close_up):")
                    for slot in required_slots:
                        found = any(
                            a.get("category") == "close_up"
                            and a.get("emotion") == slot["emotion"]
                            and a.get("angle") == slot["angle"]
                            for a in approved_list
                        )
                        mark = "✓" if found else "✗ FEHLT"
                        slot_report.append(f"  {mark} {slot['angle']:25} {slot['emotion']}")

            # ── Gesamt-Report ───────────────────────────────────────────────
            report_lines = [
                f"Dataset Curation Report",
                f"Ordner: {folder}",
                f"Gesamt: {len(image_files)} | Approved: {len(approved_list)} | Rejected: {len(rejected_list)}",
                "",
                "APPROVED:",
            ]
            for a in approved_list:
                report_lines.append(
                    f"  {a['filename']:30} {a.get('category','?'):12} "
                    f"{a.get('emotion','?'):12} {a.get('angle','?')}"
                )
            report_lines += ["", "REJECTED:"]
            for a in rejected_list:
                report_lines.append(
                    f"  {a['filename']:30} {a.get('reject_reason','?')}"
                )
            report_lines += slot_report

            # JSON-Gesamtreport speichern
            report_path = os.path.join(folder, "curation_report.json")
            with open(report_path, "w") as f:
                _json.dump({
                    "approved": approved_list,
                    "rejected": rejected_list,
                    "summary": {
                        "total": len(image_files),
                        "approved": len(approved_list),
                        "rejected": len(rejected_list),
                        "needs_slots": needs_slots,
                    }
                }, f, indent=2, ensure_ascii=False)

            report = "\n".join(report_lines)
            print(f"\n[Curator] Report gespeichert: {report_path}")
            print(report)
            return (report, len(approved_list), len(rejected_list), needs_slots)


# ---------------------------------------------------------------------------
# Node: VLMCornerInpainter
# Erstellt Ecken-Masken für Eck-Overlay-Entfernung.
# Mit optionalem vlm_model erkennt das VLM automatisch welche Ecke.
# Ohne vlm_model: manueller corners-Parameter oder Fallback all_4.
# ---------------------------------------------------------------------------
import torch
import numpy as np
from PIL import Image as _PILImage

CORNER_DETECT_PROMPT = """Look at this image carefully. Is there a corner overlay, logo, or text overlay in any corner?
Respond with ONLY a JSON object, no other text:
{
  "has_corner overlay": true or false,
  "corner": "top_left" or "top_right" or "bottom_left" or "bottom_right" or "none",
  "confidence": "high" or "medium" or "low"
}
Only report ONE corner (the most prominent corner overlay location)."""

CORNER_REMAP = {
    "top_left":     "top_left",
    "top_right":    "top_right",
    "bottom_left":  "bottom_left",
    "bottom_right": "bottom_right",
    "none":         "all_4",
}

class VLMCornerInpainter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "corner_size_percent": ("FLOAT", {
                    "default": 15.0, "min": 1.0, "max": 40.0, "step": 0.5,
                    "tooltip": "Groesse der Eckmaske in Prozent der Bildgroesse.",
                }),
                "fallback_corners": (["all_4", "top_left", "top_right",
                                      "bottom_left", "bottom_right",
                                      "top_both", "bottom_both"], {
                    "default": "all_4",
                    "tooltip": "Wird verwendet wenn kein VLM verbunden ist "
                               "oder das VLM kein Eck-Overlay erkennt.",
                }),
                "feather": ("INT", {
                    "default": 8, "min": 0, "max": 50,
                    "tooltip": "Weiche Kante der Maske in Pixeln.",
                }),
            },
            "optional": {
                "vlm_model": ("MLXVLM_MODEL", {
                    "tooltip": "Optional: VLM analysiert das Bild und erkennt "
                               "automatisch welche Ecke das Eck-Overlay enthaelt. "
                               "Ohne VLM wird fallback_corners verwendet.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image",  "mask",  "detected_corner")
    CATEGORY     = "MFlux/VLM"
    FUNCTION     = "create_corner_mask"

    def create_corner_mask(self, image, corner_size_percent,
                           fallback_corners, feather, vlm_model=None):
        import json as _json

        b, h, w, c = image.shape
        corner_h = int(h * corner_size_percent / 100)
        corner_w = int(w * corner_size_percent / 100)

        detected = fallback_corners  # Default

        # ── VLM-Analyse wenn verbunden ──────────────────────────────────────
        if vlm_model is not None and HAS_VLM:
            try:
                img_np  = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                img_pil = _PILImage.fromarray(img_np)
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                img_pil.save(tmp.name)
                tmp_path = tmp.name

                model, processor, config = vlm_model.get()
                fp  = apply_chat_template(processor, config,
                                          CORNER_DETECT_PROMPT, num_images=1)
                out = generate(model, processor, fp, [tmp_path],
                               max_tokens=80, temperature=0.0, verbose=False)
                raw = _extract_text(out).strip()

                # JSON extrahieren
                if "```" in raw:
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                result = _json.loads(raw.strip())

                has_wm  = result.get("has_corner overlay", False)
                corner  = result.get("corner", "none")
                conf    = result.get("confidence", "low")

                print(f"[CornerInpainter] VLM: has_corner overlay={has_wm}, "
                      f"corner={corner}, confidence={conf}")

                if has_wm and corner in CORNER_REMAP and corner != "none":
                    detected = CORNER_REMAP[corner]
                    print(f"[CornerInpainter] Erkannte Ecke: {detected}")
                else:
                    detected = fallback_corners
                    print(f"[CornerInpainter] Kein WZ erkannt → Fallback: {detected}")

                os.unlink(tmp_path)

            except Exception as e:
                print(f"[CornerInpainter] VLM-Analyse fehlgeschlagen: {e} "
                      f"→ Fallback: {fallback_corners}")
                detected = fallback_corners
        else:
            print(f"[CornerInpainter] Kein VLM → {fallback_corners}")

        # ── Maske erstellen ─────────────────────────────────────────────────
        mask_np = np.zeros((h, w), dtype=np.float32)

        base_corners = {
            "top_left":     (0,          corner_h, 0,          corner_w),
            "top_right":    (0,          corner_h, w-corner_w, w),
            "bottom_left":  (h-corner_h, h,        0,          corner_w),
            "bottom_right": (h-corner_h, h,        w-corner_w, w),
        }
        region_map = {
            "top_left":     [base_corners["top_left"]],
            "top_right":    [base_corners["top_right"]],
            "bottom_left":  [base_corners["bottom_left"]],
            "bottom_right": [base_corners["bottom_right"]],
            "all_4":        list(base_corners.values()),
            "top_both":     [base_corners["top_left"],    base_corners["top_right"]],
            "bottom_both":  [base_corners["bottom_left"], base_corners["bottom_right"]],
        }

        for (y1, y2, x1, x2) in region_map.get(detected, list(base_corners.values())):
            mask_np[y1:y2, x1:x2] = 1.0

        if feather > 0:
            import cv2
            ks = feather * 2 + 1
            mask_np = cv2.GaussianBlur(mask_np, (ks, ks), feather / 3)
            mask_np = np.clip(mask_np, 0.0, 1.0)

        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        print(f"[CornerInpainter] Maske: {detected}, "
              f"{corner_w}x{corner_h}px ({corner_size_percent}%)")

        return (image, mask_tensor, detected)