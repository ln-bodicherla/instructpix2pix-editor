#!/usr/bin/env python3
"""Demonstration script for the InstructPix2Pix Image Editor.

Shows typical usage patterns including single edits, iterative refinement,
batch processing, style transfer, parameter sweeps, and comparison
generation.

Usage:
    python -m examples.demo --image path/to/image.jpg --output-dir ./demo_output
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.editor import InstructPix2PixEditor
from src.pipeline import EditingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_sample_image(size: tuple = (512, 512)) -> Image.Image:
    """Create a simple gradient test image when no real image is available."""
    import numpy as np

    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for y in range(size[1]):
        for x in range(size[0]):
            arr[y, x, 0] = int(255 * x / size[0])
            arr[y, x, 1] = int(255 * y / size[1])
            arr[y, x, 2] = int(255 * (1 - x / size[0]))
    return Image.fromarray(arr)


def demo_single_edit(editor: InstructPix2PixEditor, image: Image.Image, output_dir: str) -> None:
    """Demonstrate a single text-guided edit."""
    logger.info("--- Single Edit Demo ---")

    instructions = [
        "make it look like a snowy winter scene",
        "turn it into a sunset",
        "add dramatic lighting",
        "make it look like it was taken at night",
    ]

    for instruction in instructions:
        start = time.time()
        result = editor.edit_image(
            image=image,
            instruction=instruction,
            num_inference_steps=20,
            image_guidance_scale=1.5,
            guidance_scale=7.5,
            seed=42,
        )
        elapsed = time.time() - start

        safe_name = instruction.replace(" ", "_")[:50]
        output_path = os.path.join(output_dir, f"single_{safe_name}.png")
        result.save(output_path)
        logger.info("Saved: %s (%.2fs)", output_path, elapsed)

        comparison_path = os.path.join(output_dir, f"compare_{safe_name}.png")
        editor.save_comparison(image, result, comparison_path, instruction=instruction)


def demo_iterative_edit(editor: InstructPix2PixEditor, image: Image.Image, output_dir: str) -> None:
    """Demonstrate chained iterative edits."""
    logger.info("--- Iterative Edit Demo ---")

    instructions = [
        "make it sunset with warm orange tones",
        "add dramatic clouds in the sky",
        "increase the contrast and saturation",
    ]

    intermediates_dir = os.path.join(output_dir, "iterative_steps")
    start = time.time()
    result = editor.iterative_edit(
        image=image,
        instructions=instructions,
        num_inference_steps=20,
        image_guidance_scale=1.5,
        guidance_scale=7.5,
        seed=42,
        save_intermediates=True,
        output_dir=intermediates_dir,
    )
    elapsed = time.time() - start

    output_path = os.path.join(output_dir, "iterative_final.png")
    result.save(output_path)
    logger.info("Iterative result saved: %s (%.2fs total)", output_path, elapsed)


def demo_style_transfer(editor: InstructPix2PixEditor, image: Image.Image, output_dir: str) -> None:
    """Demonstrate style transfer with various presets."""
    logger.info("--- Style Transfer Demo ---")

    styles = ["watercolor", "oil_painting", "pencil_sketch", "anime", "cinematic", "vintage"]

    for style in styles:
        start = time.time()
        result = editor.style_transfer(
            image=image,
            style=style,
            strength=1.0,
            seed=42,
        )
        elapsed = time.time() - start

        output_path = os.path.join(output_dir, f"style_{style}.png")
        result.save(output_path)
        logger.info("Style '%s' saved: %s (%.2fs)", style, output_path, elapsed)


def demo_parameter_sweep(editor: InstructPix2PixEditor, image: Image.Image, output_dir: str) -> None:
    """Demonstrate grid search over guidance parameters."""
    logger.info("--- Parameter Sweep Demo ---")

    sweep_dir = os.path.join(output_dir, "parameter_sweep")
    start = time.time()
    results = editor.adjust_parameters(
        image=image,
        instruction="make it look like a watercolor painting",
        image_guidance_range=(1.0, 1.5, 2.0),
        guidance_range=(5.0, 7.5, 10.0),
        seed=42,
        output_dir=sweep_dir,
    )
    elapsed = time.time() - start

    logger.info("Parameter sweep complete: %d combinations (%.2fs)", len(results), elapsed)
    for entry in results:
        logger.info(
            "  img_gs=%.1f  txt_gs=%.1f  -> %s",
            entry["image_guidance_scale"],
            entry["guidance_scale"],
            entry.get("path", "in-memory"),
        )


def demo_batch_processing(pipeline: EditingPipeline, input_dir: str, output_dir: str) -> None:
    """Demonstrate directory-level batch processing."""
    logger.info("--- Batch Processing Demo ---")

    batch_output = os.path.join(output_dir, "batch_results")
    start = time.time()
    results = pipeline.batch_process_directory(
        input_dir=input_dir,
        output_dir=batch_output,
        instruction="apply a warm vintage filter",
        num_inference_steps=20,
        image_guidance_scale=1.5,
        guidance_scale=7.5,
        seed=42,
    )
    elapsed = time.time() - start

    success = sum(1 for r in results if r["status"] == "success")
    logger.info("Batch processing: %d/%d succeeded (%.2fs)", success, len(results), elapsed)


def demo_pipeline_preprocessing(pipeline: EditingPipeline, image: Image.Image, output_dir: str) -> None:
    """Demonstrate the preprocessing and postprocessing pipeline."""
    logger.info("--- Pipeline Preprocessing Demo ---")

    preprocessed, metadata = pipeline.preprocess_image(image)
    logger.info("Original size: %s", metadata["original_size"])
    logger.info("Padded size:   %s", metadata["padded_size"])
    logger.info("Padding:       w=%d h=%d", metadata["pad_w"], metadata["pad_h"])

    preprocessed_path = os.path.join(output_dir, "preprocessed.png")
    preprocessed.save(preprocessed_path)
    logger.info("Preprocessed image saved: %s", preprocessed_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="InstructPix2Pix Editor Demo")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="./demo_output", help="Output directory")
    parser.add_argument("--batch-dir", type=str, default=None, help="Directory of images for batch demo")
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip loading the model (preprocessing demo only)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.image and os.path.exists(args.image):
        image = Image.open(args.image).convert("RGB")
        logger.info("Loaded input image: %s (%s)", args.image, image.size)
    else:
        logger.info("No input image provided; generating a test gradient image")
        image = create_sample_image((512, 512))
        image.save(os.path.join(args.output_dir, "test_input.png"))

    editor = InstructPix2PixEditor()
    pipeline = EditingPipeline(editor=editor)

    demo_pipeline_preprocessing(pipeline, image, args.output_dir)

    if args.skip_model:
        logger.info("--skip-model flag set; skipping model-dependent demos")
        return

    logger.info("Loading InstructPix2Pix model ...")
    editor.load_model()

    demo_single_edit(editor, image, args.output_dir)
    demo_iterative_edit(editor, image, args.output_dir)
    demo_style_transfer(editor, image, args.output_dir)
    demo_parameter_sweep(editor, image, args.output_dir)

    if args.batch_dir and os.path.isdir(args.batch_dir):
        demo_batch_processing(pipeline, args.batch_dir, args.output_dir)
    else:
        logger.info("Skipping batch demo (no --batch-dir provided)")

    logger.info("All demos complete. Results in: %s", args.output_dir)


if __name__ == "__main__":
    main()
