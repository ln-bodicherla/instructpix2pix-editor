"""FastAPI application for InstructPix2Pix image editing.

Exposes REST endpoints for single-image editing, batch editing, iterative
editing, and style transfer. Images are accepted as file uploads and
returned as PNG bytes or base64-encoded strings.
"""

import base64
import io
import logging
import os
from typing import List, Optional

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel

from src.editor import InstructPix2PixEditor
from src.pipeline import EditingPipeline

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")


def _load_api_config() -> dict:
    path = os.path.abspath(CONFIG_PATH)
    if os.path.exists(path):
        with open(path, "r") as fh:
            return yaml.safe_load(fh) or {}
    return {}


config = _load_api_config()
api_cfg = config.get("api", {})

app = FastAPI(
    title="InstructPix2Pix Image Editor",
    description="Text-guided image editing via the InstructPix2Pix diffusion model.",
    version="1.0.0",
)

editor: Optional[InstructPix2PixEditor] = None
pipeline: Optional[EditingPipeline] = None


class EditRequest(BaseModel):
    instruction: str
    num_inference_steps: int = 20
    image_guidance_scale: float = 1.5
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    return_base64: bool = False


class IterativeEditRequest(BaseModel):
    instructions: List[str]
    num_inference_steps: int = 20
    image_guidance_scale: float = 1.5
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    return_base64: bool = False


class StyleTransferRequest(BaseModel):
    style: str
    strength: float = 1.0
    seed: Optional[int] = None
    return_base64: bool = False


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_id: str
    device: str


def _image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()


def _image_response(image: Image.Image, as_base64: bool = False) -> Response:
    img_bytes = _image_to_bytes(image)
    if as_base64:
        encoded = base64.b64encode(img_bytes).decode("utf-8")
        return Response(content=encoded, media_type="text/plain")
    return Response(content=img_bytes, media_type="image/png")


async def _read_upload(upload: UploadFile) -> Image.Image:
    contents = await upload.read()
    max_size = api_cfg.get("max_upload_size", 10_485_760)
    if len(contents) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds maximum size of {max_size} bytes",
        )
    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")


@app.on_event("startup")
async def startup_event() -> None:
    global editor, pipeline
    logger.info("Initializing model on startup ...")
    editor = InstructPix2PixEditor(config_path=os.path.abspath(CONFIG_PATH))
    editor.load_model()
    pipeline = EditingPipeline(config_path=os.path.abspath(CONFIG_PATH), editor=editor)
    logger.info("Model ready")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service health and model status."""
    loaded = editor is not None and editor.pipe is not None
    return HealthResponse(
        status="healthy" if loaded else "initializing",
        model_loaded=loaded,
        model_id=editor.model_id if editor else "N/A",
        device=editor.device if editor else "N/A",
    )


@app.post("/edit")
async def edit_image(
    image: UploadFile = File(...),
    instruction: str = Form(...),
    num_inference_steps: int = Form(20),
    image_guidance_scale: float = Form(1.5),
    guidance_scale: float = Form(7.5),
    seed: Optional[int] = Form(None),
    return_base64: bool = Form(False),
) -> Response:
    """Edit a single image with a text instruction.

    Upload an image and provide a natural language instruction. The edited
    image is returned as PNG bytes or as a base64-encoded string.
    """
    if editor is None or editor.pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    pil_image = await _read_upload(image)
    result = pipeline.process_single(
        image=pil_image,
        instruction=instruction,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    return _image_response(result, as_base64=return_base64)


@app.post("/edit/batch")
async def batch_edit(
    images: List[UploadFile] = File(...),
    instruction: str = Form(...),
    num_inference_steps: int = Form(20),
    image_guidance_scale: float = Form(1.5),
    guidance_scale: float = Form(7.5),
    seed: Optional[int] = Form(None),
) -> Response:
    """Edit multiple images with the same instruction.

    Returns a multipart response with each edited image as a separate
    base64-encoded entry in a JSON array.
    """
    if editor is None or editor.pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    pil_images = [await _read_upload(img) for img in images]
    results = editor.batch_edit(
        images=pil_images,
        instruction=instruction,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    encoded_results = []
    for result in results:
        img_bytes = _image_to_bytes(result)
        encoded_results.append(base64.b64encode(img_bytes).decode("utf-8"))

    import json
    return Response(
        content=json.dumps({"images": encoded_results, "count": len(encoded_results)}),
        media_type="application/json",
    )


@app.post("/edit/iterative")
async def iterative_edit(
    image: UploadFile = File(...),
    instructions: str = Form(...),
    num_inference_steps: int = Form(20),
    image_guidance_scale: float = Form(1.5),
    guidance_scale: float = Form(7.5),
    seed: Optional[int] = Form(None),
    return_base64: bool = Form(False),
) -> Response:
    """Apply multiple instructions sequentially to an image.

    The ``instructions`` parameter is a JSON-encoded list of strings, e.g.:
    ``["make it sunset", "add dramatic clouds"]``
    """
    if editor is None or editor.pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    import json
    try:
        instruction_list = json.loads(instructions)
        if not isinstance(instruction_list, list):
            raise ValueError("Expected a list")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"instructions must be a JSON list of strings: {exc}",
        )

    pil_image = await _read_upload(image)
    result = editor.iterative_edit(
        image=pil_image,
        instructions=instruction_list,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    return _image_response(result, as_base64=return_base64)


@app.post("/style-transfer")
async def style_transfer(
    image: UploadFile = File(...),
    style: str = Form(...),
    strength: float = Form(1.0),
    seed: Optional[int] = Form(None),
    return_base64: bool = Form(False),
) -> Response:
    """Apply a style transfer to an uploaded image.

    The ``style`` parameter can be a preset name (watercolor, oil_painting,
    pencil_sketch, anime, photorealistic, cinematic, vintage, pop_art) or a
    freeform style description.
    """
    if editor is None or editor.pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    pil_image = await _read_upload(image)
    result = editor.style_transfer(
        image=pil_image,
        style=style,
        strength=strength,
        seed=seed,
    )
    return _image_response(result, as_base64=return_base64)


if __name__ == "__main__":
    import uvicorn

    host = api_cfg.get("host", "0.0.0.0")
    port = api_cfg.get("port", 8000)
    uvicorn.run(app, host=host, port=port)
