# InstructPix2Pix Image Editor

Text-guided image editing using [InstructPix2Pix](https://arxiv.org/abs/2211.09800). Enables natural language based image modifications without manual editing tools. Supports batch editing, style transfer, and iterative refinement through an intuitive API and web interface.

## Features

- **Text-Guided Editing** -- Modify images using plain English instructions (e.g., "make it sunset", "add snow to the mountains")
- **Batch Processing** -- Apply the same instruction across an entire directory of images
- **Iterative Refinement** -- Chain multiple editing instructions sequentially for complex transformations
- **Style Transfer** -- Apply artistic styles via natural language ("make it look like a watercolor painting")
- **Parameter Optimization** -- Grid search utility to find optimal guidance scales for a given editing task
- **Side-by-Side Comparisons** -- Automatically generate before/after comparison images
- **REST API** -- FastAPI-based service for programmatic access
- **Gradio UI** -- Interactive web interface for real-time editing

## Architecture

```
instructpix2pix-editor/
├── src/
│   ├── editor.py          # Core InstructPix2PixEditor class
│   ├── pipeline.py         # End-to-end editing pipeline with pre/post processing
│   └── api.py              # FastAPI REST endpoints
├── examples/
│   └── demo.py             # Usage demonstrations
├── configs/
│   └── config.yaml         # Model and processing configuration
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/yourusername/instructpix2pix-editor.git
cd instructpix2pix-editor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### Python API

```python
from src.editor import InstructPix2PixEditor

editor = InstructPix2PixEditor()
editor.load_model()

result = editor.edit_image(
    image_path="input.jpg",
    instruction="make it look like a snowy winter scene",
    num_inference_steps=20,
    image_guidance_scale=1.5,
    guidance_scale=7.5,
)
result.save("output.png")
```

### Iterative Editing

```python
instructions = [
    "make it sunset",
    "add dramatic clouds",
    "increase the saturation",
]
result = editor.iterative_edit("input.jpg", instructions)
result.save("final_output.png")
```

### REST API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000

curl -X POST "http://localhost:8000/edit" \
  -F "image=@input.jpg" \
  -F "instruction=make it look like an oil painting" \
  -o output.png
```

### Batch Processing

```python
from src.pipeline import EditingPipeline

pipeline = EditingPipeline()
pipeline.batch_process_directory(
    input_dir="./images",
    output_dir="./edited",
    instruction="convert to pencil sketch style",
)
```

## Configuration

Edit `configs/config.yaml` to adjust model parameters, inference settings, and API configuration.

| Parameter | Description | Default |
|---|---|---|
| `model.model_id` | HuggingFace model identifier | `timbrooks/instruct-pix2pix` |
| `model.device` | Compute device | `cuda` |
| `editing.num_inference_steps` | Diffusion steps | `20` |
| `editing.image_guidance_scale` | How closely to follow the input image | `1.5` |
| `editing.guidance_scale` | How closely to follow the text instruction | `7.5` |
| `processing.max_image_size` | Maximum dimension before downscaling | `1024` |

## Guidance Scale Tips

- **image_guidance_scale** controls fidelity to the original image. Higher values preserve more of the original.
- **guidance_scale** controls adherence to the text instruction. Higher values produce stronger edits.
- For subtle edits, use `image_guidance_scale=1.8` and `guidance_scale=5.0`.
- For dramatic transformations, use `image_guidance_scale=1.0` and `guidance_scale=10.0`.

## Requirements

- Python 3.9+
- CUDA-capable GPU with at least 8GB VRAM (16GB recommended for high-resolution editing)
- PyTorch 2.0+

## License

MIT License
