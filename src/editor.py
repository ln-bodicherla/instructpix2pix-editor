"""Core InstructPix2Pix image editing engine.

Provides the InstructPix2PixEditor class for text-guided image editing using
the InstructPix2Pix diffusion model. Supports single-image editing, batch
processing, iterative refinement, and style transfer operations.
"""

import itertools
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


class InstructPix2PixEditor:
    """Text-guided image editor built on the InstructPix2Pix diffusion model.

    This class wraps the diffusers StableDiffusionInstructPix2PixPipeline and
    provides high-level methods for common editing workflows including single
    edits, batch processing, iterative refinement, and style transfer.

    Attributes:
        model_id: HuggingFace model identifier.
        device: Torch device string (e.g., "cuda", "cpu").
        dtype: Torch dtype for model weights.
        pipe: The loaded diffusion pipeline, or None if not yet loaded.
    """

    STYLE_PRESETS: Dict[str, Dict[str, float]] = {
        "watercolor": {"guidance_scale": 8.0, "image_guidance_scale": 1.2},
        "oil_painting": {"guidance_scale": 9.0, "image_guidance_scale": 1.3},
        "pencil_sketch": {"guidance_scale": 7.0, "image_guidance_scale": 1.1},
        "anime": {"guidance_scale": 8.5, "image_guidance_scale": 1.4},
        "photorealistic": {"guidance_scale": 6.0, "image_guidance_scale": 1.8},
        "cinematic": {"guidance_scale": 7.5, "image_guidance_scale": 1.5},
        "vintage": {"guidance_scale": 7.0, "image_guidance_scale": 1.4},
        "pop_art": {"guidance_scale": 9.0, "image_guidance_scale": 1.2},
    }

    def __init__(
        self,
        config_path: Optional[str] = None,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> None:
        """Initialize the editor with configuration.

        Args:
            config_path: Path to a YAML configuration file. Falls back to
                default config location if not provided.
            model_id: Override for the HuggingFace model ID.
            device: Override for the compute device.
            dtype: Override for the model weight dtype ("float16" or "float32").
        """
        self.config = self._load_config(config_path)

        model_cfg = self.config.get("model", {})
        self.model_id = model_id or model_cfg.get("model_id", "timbrooks/instruct-pix2pix")
        self.device = device or model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        dtype_str = dtype or model_cfg.get("dtype", "float16")
        self.dtype = torch.float16 if dtype_str == "float16" else torch.float32

        editing_cfg = self.config.get("editing", {})
        self.default_steps = editing_cfg.get("num_inference_steps", 20)
        self.default_image_guidance = editing_cfg.get("image_guidance_scale", 1.5)
        self.default_guidance = editing_cfg.get("guidance_scale", 7.5)

        self.pipe = None
        logger.info(
            "Editor initialized: model=%s device=%s dtype=%s",
            self.model_id,
            self.device,
            self.dtype,
        )

    @staticmethod
    def _load_config(config_path: Optional[str] = None) -> dict:
        """Load YAML configuration from disk."""
        if config_path is None:
            default_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
            config_path = str(default_path)

        path = Path(config_path)
        if path.exists():
            with open(path, "r") as fh:
                return yaml.safe_load(fh) or {}
        logger.warning("Config file not found at %s, using defaults", config_path)
        return {}

    def load_model(self) -> None:
        """Download and load the InstructPix2Pix pipeline.

        The model weights are loaded onto the configured device with the
        configured dtype. Safety checker is disabled for unrestricted editing.
        """
        from diffusers import StableDiffusionInstructPix2PixPipeline

        logger.info("Loading model %s onto %s ...", self.model_id, self.device)
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
        )
        self.pipe = self.pipe.to(self.device)
        self.pipe.enable_attention_slicing()

        if self.device == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("xformers memory-efficient attention enabled")
            except Exception:
                logger.debug("xformers not available, using default attention")

        logger.info("Model loaded successfully")

    def _ensure_loaded(self) -> None:
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

    def _open_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Open an image from a path or return the PIL Image directly."""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        return image.convert("RGB")

    def edit_image(
        self,
        image: Union[str, Path, Image.Image],
        instruction: str,
        num_inference_steps: Optional[int] = None,
        image_guidance_scale: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Apply a text instruction to edit an image.

        Args:
            image: Input image as a file path or PIL Image.
            instruction: Natural language editing instruction.
            num_inference_steps: Number of diffusion denoising steps.
            image_guidance_scale: Fidelity to the input image.
            guidance_scale: Adherence to the text instruction.
            seed: Random seed for reproducibility.

        Returns:
            The edited PIL Image.
        """
        self._ensure_loaded()
        pil_image = self._open_image(image)

        steps = num_inference_steps or self.default_steps
        img_guidance = image_guidance_scale or self.default_image_guidance
        txt_guidance = guidance_scale or self.default_guidance

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.info(
            "Editing image: instruction='%s' steps=%d img_gs=%.2f txt_gs=%.2f",
            instruction,
            steps,
            img_guidance,
            txt_guidance,
        )

        with torch.no_grad(), torch.autocast(self.device, dtype=self.dtype, enabled=(self.device == "cuda")):
            result = self.pipe(
                prompt=instruction,
                image=pil_image,
                num_inference_steps=steps,
                image_guidance_scale=img_guidance,
                guidance_scale=txt_guidance,
                generator=generator,
            )

        return result.images[0]

    def batch_edit(
        self,
        images: List[Union[str, Path, Image.Image]],
        instruction: str,
        num_inference_steps: Optional[int] = None,
        image_guidance_scale: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """Apply the same instruction to multiple images.

        Args:
            images: List of input images (paths or PIL Images).
            instruction: Natural language editing instruction.
            num_inference_steps: Number of diffusion denoising steps.
            image_guidance_scale: Fidelity to each input image.
            guidance_scale: Adherence to the text instruction.
            seed: Random seed applied to each edit.

        Returns:
            List of edited PIL Images in the same order as the inputs.
        """
        results: List[Image.Image] = []
        total = len(images)
        for idx, img in enumerate(images):
            logger.info("Batch edit %d/%d", idx + 1, total)
            edited = self.edit_image(
                image=img,
                instruction=instruction,
                num_inference_steps=num_inference_steps,
                image_guidance_scale=image_guidance_scale,
                guidance_scale=guidance_scale,
                seed=seed,
            )
            results.append(edited)
        return results

    def iterative_edit(
        self,
        image: Union[str, Path, Image.Image],
        instructions: List[str],
        num_inference_steps: Optional[int] = None,
        image_guidance_scale: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        save_intermediates: bool = False,
        output_dir: Optional[str] = None,
    ) -> Image.Image:
        """Apply a sequence of editing instructions one after another.

        Each instruction is applied to the output of the previous step,
        enabling complex multi-step transformations.

        Args:
            image: Starting image.
            instructions: Ordered list of editing instructions.
            num_inference_steps: Diffusion steps per edit.
            image_guidance_scale: Image guidance per edit.
            guidance_scale: Text guidance per edit.
            seed: Random seed for each step.
            save_intermediates: If True, save each intermediate result.
            output_dir: Directory for intermediate results.

        Returns:
            The final edited PIL Image.
        """
        current = self._open_image(image)

        if save_intermediates and output_dir:
            os.makedirs(output_dir, exist_ok=True)

        for step_idx, instruction in enumerate(instructions):
            logger.info("Iterative step %d/%d: '%s'", step_idx + 1, len(instructions), instruction)
            current = self.edit_image(
                image=current,
                instruction=instruction,
                num_inference_steps=num_inference_steps,
                image_guidance_scale=image_guidance_scale,
                guidance_scale=guidance_scale,
                seed=seed,
            )
            if save_intermediates and output_dir:
                intermediate_path = os.path.join(output_dir, f"step_{step_idx + 1:03d}.png")
                current.save(intermediate_path)
                logger.info("Saved intermediate: %s", intermediate_path)

        return current

    def style_transfer(
        self,
        image: Union[str, Path, Image.Image],
        style: str,
        strength: float = 1.0,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Apply a style transfer to an image using preset or custom style.

        Args:
            image: Input image.
            style: Either a preset name (see STYLE_PRESETS) or a freeform
                style description string.
            strength: Multiplier for the guidance scale (0.5 = subtle, 2.0 = extreme).
            seed: Random seed for reproducibility.

        Returns:
            The stylized PIL Image.
        """
        if style.lower().replace(" ", "_") in self.STYLE_PRESETS:
            preset = self.STYLE_PRESETS[style.lower().replace(" ", "_")]
            guidance = preset["guidance_scale"] * strength
            img_guidance = preset["image_guidance_scale"]
            instruction = f"make it look like a {style.replace('_', ' ')}"
        else:
            guidance = self.default_guidance * strength
            img_guidance = self.default_image_guidance
            instruction = style

        logger.info("Style transfer: style='%s' strength=%.2f", style, strength)
        return self.edit_image(
            image=image,
            instruction=instruction,
            image_guidance_scale=img_guidance,
            guidance_scale=guidance,
            seed=seed,
        )

    def adjust_parameters(
        self,
        image: Union[str, Path, Image.Image],
        instruction: str,
        image_guidance_range: Tuple[float, ...] = (1.0, 1.5, 2.0),
        guidance_range: Tuple[float, ...] = (5.0, 7.5, 10.0),
        seed: int = 42,
        output_dir: Optional[str] = None,
    ) -> List[Dict]:
        """Grid search over guidance scales to find optimal parameters.

        Produces one edited image for every combination of
        image_guidance_scale and guidance_scale, returning metadata
        for each result.

        Args:
            image: Input image for the parameter sweep.
            instruction: Text instruction to apply.
            image_guidance_range: Values to try for image_guidance_scale.
            guidance_range: Values to try for guidance_scale.
            seed: Fixed seed for fair comparison.
            output_dir: Directory to save all grid outputs.

        Returns:
            List of dicts with keys: image_guidance_scale, guidance_scale,
            result (PIL Image), and path (if output_dir is set).
        """
        results: List[Dict] = []
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        combos = list(itertools.product(image_guidance_range, guidance_range))
        total = len(combos)

        for idx, (img_gs, txt_gs) in enumerate(combos):
            logger.info("Grid search %d/%d: img_gs=%.2f txt_gs=%.2f", idx + 1, total, img_gs, txt_gs)
            edited = self.edit_image(
                image=image,
                instruction=instruction,
                image_guidance_scale=img_gs,
                guidance_scale=txt_gs,
                seed=seed,
            )
            entry = {
                "image_guidance_scale": img_gs,
                "guidance_scale": txt_gs,
                "result": edited,
            }
            if output_dir:
                fname = f"igs_{img_gs:.1f}_gs_{txt_gs:.1f}.png"
                path = os.path.join(output_dir, fname)
                edited.save(path)
                entry["path"] = path
            results.append(entry)

        return results

    def save_comparison(
        self,
        original: Union[str, Path, Image.Image],
        edited: Image.Image,
        output_path: str,
        instruction: Optional[str] = None,
        label_height: int = 40,
    ) -> str:
        """Create and save a side-by-side before/after comparison image.

        Args:
            original: The original input image.
            edited: The edited output image.
            output_path: File path for the comparison image.
            instruction: Optional instruction text to render at the top.
            label_height: Pixel height reserved for text labels.

        Returns:
            The output file path.
        """
        orig = self._open_image(original)

        target_h = max(orig.height, edited.height)
        orig_resized = orig.resize((int(orig.width * target_h / orig.height), target_h), Image.LANCZOS)
        edit_resized = edited.resize((int(edited.width * target_h / edited.height), target_h), Image.LANCZOS)

        gap = 10
        header = label_height if instruction else 0
        canvas_w = orig_resized.width + gap + edit_resized.width
        canvas_h = target_h + header
        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

        draw = ImageDraw.Draw(canvas)
        if instruction:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except OSError:
                font = ImageFont.load_default()
            text = f'Instruction: "{instruction}"'
            draw.text((10, 10), text, fill=(0, 0, 0), font=font)

        canvas.paste(orig_resized, (0, header))
        canvas.paste(edit_resized, (orig_resized.width + gap, header))

        draw.text((10, header + 5), "Original", fill=(255, 255, 255))
        draw.text((orig_resized.width + gap + 10, header + 5), "Edited", fill=(255, 255, 255))

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        canvas.save(output_path)
        logger.info("Comparison saved to %s", output_path)
        return output_path
