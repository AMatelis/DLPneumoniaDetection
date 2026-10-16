
"""Streamlit dashboard for the pneumonia detection research project."""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.h5"
MODEL_PATH = Path(
    os.getenv("PNEUMONIA_MODEL_PATH", str(DEFAULT_MODEL_PATH))
)

IMG_SIZE = (224, 224)
MAX_UPLOAD_SIZE = 10 * 1024 * 1024


st.set_page_config(
    page_title="Pneumonia Detection Research",
    page_icon="🩻",
    layout="wide",
)


@st.cache_resource
def get_model(model_path: str, modified_time: float) -> Any:
    """Load and cache the trained Keras model."""
    del modified_time  # Included in the cache key to detect model updates.
    return load_model(model_path, compile=False)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Prepare an image using the original model's input preprocessing."""
    image = image.convert("RGB").resize(IMG_SIZE)
    image_array = img_to_array(image).astype(np.float32)
    image_array /= 255.0

    return np.expand_dims(image_array, axis=0)


def predict_image(model: Any, image: Image.Image) -> dict[str, Any]:
    """Return the predicted class and sigmoid probability."""
    batch = preprocess_image(image)
    predictions = np.asarray(
        model.predict(batch, verbose=0)
    ).reshape(-1)

    if predictions.size != 1 or not np.isfinite(predictions[0]):
        raise ValueError("The model returned an invalid prediction.")

    probability = float(np.clip(predictions[0], 0.0, 1.0))
    predicted_class = (
        "PNEUMONIA" if probability >= 0.5 else "NORMAL"
    )

    return {
        "predicted_class": predicted_class,
        "pneumonia_probability": probability,
        "normal_probability": 1.0 - probability,
    }


def display_results(result: dict[str, Any]) -> None:
    """Display the model's output."""
    st.subheader("Prediction results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted class", result["predicted_class"])

    with col2:
        st.metric(
            "Pneumonia model score",
            f"{result['pneumonia_probability']:.1%}",
        )

    st.write("Model output scores")
    st.progress(result["pneumonia_probability"])
    st.caption(
        "Pneumonia score: "
        f"{result['pneumonia_probability']:.1%}"
    )
    st.caption(
        "Normal score: "
        f"{result['normal_probability']:.1%}"
    )

    st.warning(
        "Research use only. This model has not been established as a "
        "clinical diagnostic tool. Its predictions may be incorrect "
        "and should not guide medical decisions."
    )


def display_evaluation_results() -> None:
    """Show available evaluation JSON files from the outputs directory."""
    st.subheader("Saved evaluation results")

    output_dir = PROJECT_ROOT / "outputs"

    if not output_dir.is_dir():
        st.info("No outputs directory was found yet.")
        return

    json_files = sorted(output_dir.rglob("*.json"))

    if not json_files:
        st.info(
            "No JSON result files found. Run an evaluation first."
        )
        return

    # Limit the list so the dashboard remains easy to navigate.
    selected_file = st.selectbox(
        "Choose a saved JSON file",
        options=json_files[:100],
        format_func=lambda path: str(path.relative_to(PROJECT_ROOT)),
    )

    try:
        with selected_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        st.caption(str(selected_file.relative_to(PROJECT_ROOT)))
        st.json(data)

    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not read this JSON file: {exc}")


def main() -> None:
    """Run the Streamlit application."""
    st.title("Pneumonia Detection Research Dashboard")
    st.write(
        "Explore model predictions from a trained chest X-ray "
        "classification model and inspect saved experiment results."
    )

    with st.sidebar:
        st.header("Model information")
        st.write("Architecture: ResNet50")
        st.write(f"Input size: {IMG_SIZE[0]} × {IMG_SIZE[1]}")
        st.write(f"Model path: `{MODEL_PATH}`")

        st.divider()
        st.caption(
            "The displayed scores are model outputs, not confirmed "
            "clinical probabilities."
        )

    prediction_tab, evaluation_tab = st.tabs(
        ["X-ray prediction", "Evaluation results"]
    )

    with prediction_tab:
        st.subheader("Upload a chest X-ray")

        uploaded_file = st.file_uploader(
            "Select a JPEG, PNG, or WebP image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Maximum file size: 10 MB.",
        )

        if uploaded_file is not None:
            if uploaded_file.size > MAX_UPLOAD_SIZE:
                st.error("The uploaded image must be smaller than 10 MB.")
                return

            try:
                with Image.open(uploaded_file) as source_image:
                    image = source_image.convert("RGB")

            except (UnidentifiedImageError, OSError, ValueError):
                st.error("The uploaded file could not be opened as an image.")
                return

            left, right = st.columns([1, 1])

            with left:
                st.image(
                    image,
                    caption="Uploaded chest X-ray",
                    use_container_width=True,
                )

            with right:
                if not MODEL_PATH.is_file():
                    st.error(
                        "Trained model not found. Place the model at "
                        f"`{MODEL_PATH}` or set the "
                        "`PNEUMONIA_MODEL_PATH` environment variable."
                    )
                else:
                    try:
                        model = get_model(
                            str(MODEL_PATH),
                            MODEL_PATH.stat().st_mtime,
                        )

                        with st.spinner("Running model prediction..."):
                            result = predict_image(model, image)

                        display_results(result)

                    except Exception as exc:
                        st.error(
                            "Prediction failed. Check the model file "
                            "and preprocessing configuration."
                        )
                        st.exception(exc)

    with evaluation_tab:
        st.write(
            "Browse JSON files saved by your training and evaluation "
            "pipeline. The files shown depend on what your scripts "
            "actually save."
        )
        display_evaluation_results()


if __name__ == "__main__":
    main()