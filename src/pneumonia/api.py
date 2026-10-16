
"""FastAPI application for pneumonia classification."""

import io
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.h5"
MODEL_PATH = Path(os.getenv("PNEUMONIA_MODEL_PATH", str(DEFAULT_MODEL_PATH)))

IMG_SIZE = (224, 224)
CLASS_NAMES = ("NORMAL", "PNEUMONIA")

app = FastAPI(
    title="Pneumonia Detection API",
    description="Classifies chest X-ray images using a trained ResNet50 model.",
    version="1.0.0",
)

model: Any = None


@app.on_event("startup")
def load_prediction_model() -> None:
    """Load the trained model when the API starts."""
    global model

    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}. "
            "Train the model or set PNEUMONIA_MODEL_PATH."
        )

    model = load_model(str(MODEL_PATH), compile=False)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert an uploaded image into the model's expected input format."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            image = image.resize(IMG_SIZE)
            image_array = img_to_array(image)

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from exc

    # Match the original model's ImageDataGenerator rescaling.
    image_array = image_array / 255.0

    return np.expand_dims(image_array, axis=0).astype(np.float32)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Report whether the model is loaded."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The prediction model is not loaded.",
        )

    return {"status": "ok", "model": "loaded"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Predict whether a chest X-ray is classified as pneumonia."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The prediction model is not loaded.",
        )

    allowed_types = {"image/jpeg", "image/png", "image/webp"}

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        # Avoid processing unnecessarily large uploads.
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="The image must be smaller than 10 MB.",
            )

        processed_image = preprocess_image(image_bytes)
        prediction = np.asarray(
            model.predict(processed_image, verbose=0)
        ).reshape(-1)

        if prediction.size != 1 or not np.isfinite(prediction[0]):
            raise HTTPException(
                status_code=500,
                detail="The model returned an invalid prediction.",
            )

        # The original model uses a sigmoid output:
        # 0 = NORMAL and 1 = PNEUMONIA.
        pneumonia_probability = float(
            np.clip(prediction[0], 0.0, 1.0)
        )

        predicted_class = (
            "PNEUMONIA"
            if pneumonia_probability >= 0.5
            else "NORMAL"
        )

        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "pneumonia_probability": round(
                pneumonia_probability, 4
            ),
            "normal_probability": round(
                1.0 - pneumonia_probability, 4
            ),
            "threshold": 0.5,
            "message": (
                "Research model output only. "
                "This prediction is not a medical diagnosis."
            ),
        }

    finally:
        await file.close()