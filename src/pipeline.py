"""End-to-end editing pipeline with pre/post processing and masking support.

Provides the EditingPipeline class that wraps InstructPix2PixEditor with
image preprocessing (resize, pad), postprocessing (restore dimensions,
color correction), masked editing, blending, and directory-level batch
processing.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from PIL import Image, ImageFilter

from src.editor import InstructPix2PixEditor

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


class EditingPipeline:
    """Full editing pipeline with pre/post processing and masked editing.

    Wraps InstructPix2PixEditor and adds image preprocessing (resize,
    padding to model-friendly dimensions), postprocessing (dimension
    restoration, color correction), masked editing for region-specific
    modifications, and directory-level batch processing.

    Attributes:
        editor: Underlying InstructPix2PixEditor instance.
        max_size: Maximum image dimension before downscaling.
        output_format: Default output image format.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        editor: Optional[InstructPix2PixEditor] = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            config_path: Path to YAML configuration.
            editor: Pre-configured editor instance. If None, one is created.
        """
        self.config = self._load_config(config_path)
        processing_cfg = self.config.get("processing", {})
        self.max_size = processing_cfg.get("max_image_size", 1024)
        self.output_format = processing_cfg.get("output_format", "png")

        if editor is not None:
            self.editor = editor
        else:
            self.editor = InstructPix2PixEditor(config_path=config_path)

    @staticmethod
    def _load_config(config_path: Optional[str] = None) -> dict:
        if config_path is None:
            default = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
            config_path = str(default)
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as fh:
                return yaml.safe_load(fh) or {}
        return {}

    def create_pipeline(
        self,
        instruction: str,
        mask: Optional[Union[str, Path, Image.Image]] = None,
        num_inference_steps: Optional[int] = None,
        image_guidance_scale: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        color_correct: bool = True,
        seed: Optional[int] = None,
    ) -> "PipelineConfig":
        """Create a reusable pipeline configuration.

        Returns a PipelineConfig object that can be applied to one or more
        images, carrying the full set of processing parameters.

        Args:
            instruction: Text editing instruction.
            mask: Optional mask image (white = edit, black = preserve).
            num_inference_steps: Diffusion steps.
            image_guidance_scale: Image guidance.
            guidance_scale: Text guidance.
            color_correct: Whether to apply post-edit color correction.
            seed: Random seed.

        Returns:
            A PipelineConfig dataclass with all parameters.
        """
        mask_img = None
        if mask is not None:
            if isinstance(mask, (str, Path)):
                mask_img = Image.open(mask).convert("L")
            else:
                mask_img = mask.convert("L")

        return PipelineConfig(
            instruction=instruction,
            mask=mask_img,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
            color_correct=color_correct,
            seed=seed,
        )

    def preprocess_image(
        self,
        image: Union[str, Path, Image.Image],
        target_multiple: int = 8,
    ) -> Tuple[Image.Image, Dict]:
        """Resize and pad an image to model-compatible dimensions.

        The image is downscaled so that neither dimension exceeds
        ``max_size``, then padded so both dimensions are multiples of
        ``target_multiple``.

        Args:
            image: Input image.
            target_multiple: Both dimensions will be padded to a multiple
                of this value.

        Returns:
            A tuple of (processed PIL Image, metadata dict) where metadata
            contains the original size and padding offsets needed to restore
            the original dimensions.
        """
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")

        original_size = img.size
        metadata = {"original_size": original_size}

        w, h = img.size
        if max(w, h) > self.max_size:
            scale = self.max_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            logger.info("Downscaled from %s to %s", original_size, img.size)

        w, h = img.size
        pad_w = (target_multiple - w % target_multiple) % target_multiple
        pad_h = (target_multiple - h % target_multiple) % target_multiple

        if pad_w > 0 or pad_h > 0:
            padded = Image.new("RGB", (w + pad_w, h + pad_h), (0, 0, 0))
            padded.paste(img, (0, 0))
            img = padded
            logger.info("Padded to %s (pad_w=%d, pad_h=%d)", img.size, pad_w, pad_h)

        metadata["padded_size"] = img.size
        metadata["pad_w"] = pad_w
        metadata["pad_h"] = pad_h

        return img, metadata

    def postprocess_image(
        self,
        image: Image.Image,
        metadata: Dict,
        color_correct: bool = True,
        reference: Optional[Image.Image] = None,
    ) -> Image.Image:
        """Restore original dimensions and optionally correct color.

        Removes any padding applied during preprocessing and resizes back
        to the original image dimensions. Optionally performs histogram-based
        color correction to match the original image's color distribution.

        Args:
            image: Edited image to postprocess.
            metadata: Metadata dict produced by preprocess_image.
            color_correct: Whether to apply color matching.
            reference: Reference image for color correction (typically the
                original input).

        Returns:
            The postprocessed PIL Image.
        """
        pad_w = metadata.get("pad_w", 0)
        pad_h = metadata.get("pad_h", 0)
        padded_size = metadata.get("padded_size", image.size)

        crop_w = padded_size[0] - pad_w
        crop_h = padded_size[1] - pad_h
        if pad_w > 0 or pad_h > 0:
            image = image.crop((0, 0, crop_w, crop_h))

        original_size = metadata.get("original_size", image.size)
        if image.size != original_size:
            image = image.resize(original_size, Image.LANCZOS)
            logger.info("Restored to original size %s", original_size)

        if color_correct and reference is not None:
            image = self._match_color(image, reference)

        return image

    @staticmethod
    def _match_color(target: Image.Image, reference: Image.Image) -> Image.Image:
        """Match the color statistics of target to reference.

        Uses per-channel mean and standard deviation matching in LAB-like
        space approximated via RGB to avoid heavy dependencies.
        """
        target_arr = np.array(target, dtype=np.float32)
        ref_arr = np.array(reference.resize(target.size, Image.LANCZOS), dtype=np.float32)

        for c in range(3):
            t_mean, t_std = target_arr[:, :, c].mean(), target_arr[:, :, c].std() + 1e-6
            r_mean, r_std = ref_arr[:, :, c].mean(), ref_arr[:, :, c].std() + 1e-6
            target_arr[:, :, c] = (target_arr[:, :, c] - t_mean) * (r_std / t_std) + r_mean

        target_arr = np.clip(target_arr, 0, 255).astype(np.uint8)
        return Image.fromarray(target_arr)

    def apply_mask(
        self,
        original: Image.Image,
        edited: Image.Image,
        mask: Image.Image,
    ) -> Image.Image:
        """Composite edited content into the original using a binary mask.

        White pixels in the mask select the edited image; black pixels
        retain the original.

        Args:
            original: Original unedited image.
            edited: Fully edited image.
            mask: Grayscale mask (white = use edited, black = use original).

        Returns:
            Composited PIL Image.
        """
        mask_resized = mask.resize(original.size, Image.LANCZOS)
        return Image.composite(edited.resize(original.size, Image.LANCZOS), original, mask_resized)

    def blend_results(
        self,
        original: Image.Image,
        edited: Image.Image,
        mask: Image.Image,
        feather_radius: int = 15,
    ) -> Image.Image:
        """Blend edited region into original using a feathered mask.

        Applies a Gaussian blur to the mask edges to produce smooth
        transitions between edited and original regions.

        Args:
            original: Original image.
            edited: Edited image.
            mask: Binary mask.
            feather_radius: Gaussian blur radius for feathering.

        Returns:
            Blended PIL Image with smooth transitions.
        """
        mask_resized = mask.resize(original.size, Image.LANCZOS)
        feathered = mask_resized.filter(ImageFilter.GaussianBlur(radius=feather_radius))
        edited_resized = edited.resize(original.size, Image.LANCZOS)
        return Image.composite(edited_resized, original, feathered)

    def process_single(
        self,
        image: Union[str, Path, Image.Image],
        instruction: str,
        mask: Optional[Union[str, Path, Image.Image]] = None,
        feather_radius: int = 15,
        color_correct: bool = True,
        **edit_kwargs,
    ) -> Image.Image:
        """Full pipeline for a single image: preprocess, edit, postprocess.

        Args:
            image: Input image.
            instruction: Text editing instruction.
            mask: Optional mask for region-specific editing.
            feather_radius: Blur radius for mask feathering.
            color_correct: Whether to apply color correction.
            **edit_kwargs: Forwarded to editor.edit_image.

        Returns:
            The final edited and postprocessed PIL Image.
        """
        if isinstance(image, (str, Path)):
            original = Image.open(image).convert("RGB")
        else:
            original = image.convert("RGB")

        preprocessed, metadata = self.preprocess_image(original)
        edited = self.editor.edit_image(preprocessed, instruction, **edit_kwargs)
        postprocessed = self.postprocess_image(
            edited, metadata, color_correct=color_correct, reference=original,
        )

        if mask is not None:
            if isinstance(mask, (str, Path)):
                mask_img = Image.open(mask).convert("L")
            else:
                mask_img = mask.convert("L")
            postprocessed = self.blend_results(original, postprocessed, mask_img, feather_radius)

        return postprocessed

    def batch_process_directory(
        self,
        input_dir: str,
        output_dir: str,
        instruction: str,
        mask_dir: Optional[str] = None,
        color_correct: bool = True,
        **edit_kwargs,
    ) -> List[Dict]:
        """Process every image in a directory.

        Args:
            input_dir: Directory containing input images.
            output_dir: Directory for output images.
            instruction: Text editing instruction.
            mask_dir: Optional directory containing masks with matching
                filenames.
            color_correct: Whether to apply color correction.
            **edit_kwargs: Forwarded to editor.edit_image.

        Returns:
            List of dicts with input_path, output_path, and status for
            each processed image.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            f for f in input_path.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not image_files:
            logger.warning("No supported images found in %s", input_dir)
            return []

        results: List[Dict] = []
        total = len(image_files)

        for idx, img_file in enumerate(image_files):
            logger.info("Processing %d/%d: %s", idx + 1, total, img_file.name)
            out_file = output_path / f"{img_file.stem}.{self.output_format}"

            mask = None
            if mask_dir:
                mask_file = Path(mask_dir) / img_file.name
                if mask_file.exists():
                    mask = str(mask_file)

            try:
                result = self.process_single(
                    image=str(img_file),
                    instruction=instruction,
                    mask=mask,
                    color_correct=color_correct,
                    **edit_kwargs,
                )
                result.save(str(out_file))
                results.append({
                    "input_path": str(img_file),
                    "output_path": str(out_file),
                    "status": "success",
                })
            except Exception as exc:
                logger.error("Failed to process %s: %s", img_file.name, exc)
                results.append({
                    "input_path": str(img_file),
                    "output_path": None,
                    "status": f"error: {exc}",
                })

        success_count = sum(1 for r in results if r["status"] == "success")
        logger.info("Batch complete: %d/%d succeeded", success_count, total)
        return results


class PipelineConfig:
    """Immutable configuration for a reusable editing pipeline."""

    __slots__ = (
        "instruction",
        "mask",
        "num_inference_steps",
        "image_guidance_scale",
        "guidance_scale",
        "color_correct",
        "seed",
    )

    def __init__(
        self,
        instruction: str,
        mask: Optional[Image.Image] = None,
        num_inference_steps: Optional[int] = None,
        image_guidance_scale: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        color_correct: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.instruction = instruction
        self.mask = mask
        self.num_inference_steps = num_inference_steps
        self.image_guidance_scale = image_guidance_scale
        self.guidance_scale = guidance_scale
        self.color_correct = color_correct
        self.seed = seed

    def to_dict(self) -> Dict:
        return {
            "instruction": self.instruction,
            "has_mask": self.mask is not None,
            "num_inference_steps": self.num_inference_steps,
            "image_guidance_scale": self.image_guidance_scale,
            "guidance_scale": self.guidance_scale,
            "color_correct": self.color_correct,
            "seed": self.seed,
        }
