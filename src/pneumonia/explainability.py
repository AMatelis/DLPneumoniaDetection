
"""Grad-CAM explainability and quantitative analysis for pneumonia detection."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from pneumonia.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_SIZE = (224, 224)


def _resolve_output_dir(exp_dir: str | Path | None = None) -> Path:
    """Resolve and create the directory used for explainability outputs."""
    if exp_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "explainability"
    else:
        output_dir = Path(exp_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _find_last_conv_layer(model: tf.keras.Model) -> tuple[tf.keras.layers.Layer, tf.keras.Model]:
    """
    Find the last convolutional layer.

    Supports a model containing a nested ResNet50 base model, as used by
    the project's Sequential architecture, and many ordinary Keras models.
    """
    # First, search nested models in reverse order. This handles the
    # project's Sequential([ResNet50, pooling, dense, dropout, sigmoid])
    # architecture.
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            try:
                nested_layer, nested_model = _find_last_conv_layer(layer)
                return nested_layer, nested_model
            except ValueError:
                continue

    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv2D,
    )

    for layer in reversed(model.layers):
        if isinstance(layer, conv_types):
            return layer, model

    raise ValueError(
        "No convolutional layer was found. Grad-CAM requires a model "
        "containing convolutional feature maps."
    )


def _prepare_image(
    image: str | Path | Image.Image | np.ndarray,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an image and prepare it using the project's 1/255 rescaling.

    Returns:
        batch: Float32 model input with shape (1, height, width, 3).
        display_image: RGB image scaled to [0, 1] for plotting.
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            pil_image = source.convert("RGB").resize(
                (image_size[1], image_size[0])
            )
            display_image = np.asarray(pil_image, dtype=np.float32) / 255.0

    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB").resize(
            (image_size[1], image_size[0])
        )
        display_image = np.asarray(pil_image, dtype=np.float32) / 255.0

    else:
        array = np.asarray(image)

        if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
            raise ValueError(
                "Image arrays must have shape (height, width, channels), "
                "with 1, 3, or 4 channels."
            )

        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        elif array.shape[-1] == 4:
            array = array[..., :3]

        if np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.float32) / 255.0
        else:
            array = array.astype(np.float32)
            if array.size and (array.min() < 0 or array.max() > 1):
                raise ValueError(
                    "Floating-point image arrays must already be in [0, 1]."
                )

        array = np.clip(array, 0.0, 1.0)
        pil_image = Image.fromarray(
            np.round(array * 255).astype(np.uint8)
        ).resize((image_size[1], image_size[0]))

        display_image = np.asarray(pil_image, dtype=np.float32) / 255.0

    batch = np.expand_dims(display_image, axis=0).astype(np.float32)
    return batch, display_image


def _build_gradcam_model(
    model: tf.keras.Model,
) -> tuple[tf.keras.Model, Any]:
    """
    Build a model returning convolutional activations and final predictions.

    For a nested ResNet50, apply the original model's top layers to the
    feature extractor's output so the gradient path remains connected.
    """
    conv_layer, containing_model = _find_last_conv_layer(model)

    if containing_model is model:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output],
        )
        return grad_model, conv_layer.name

    # The convolutional layer belongs to a nested model, typically ResNet50.
    feature_extractor = tf.keras.Model(
        inputs=containing_model.inputs,
        outputs=[conv_layer.output, containing_model.output],
    )

    try:
        nested_index = next(
            index
            for index, layer in enumerate(model.layers)
            if layer is containing_model
        )
    except StopIteration as exc:
        raise ValueError(
            "Could not locate the nested feature extractor in the model."
        ) from exc

    top_layers = model.layers[nested_index + 1 :]

    def forward_from_features(features: tf.Tensor) -> tf.Tensor:
        output = features
        for layer in top_layers:
            output = layer(output, training=False)
        return output

    inputs = tf.keras.Input(shape=model.input_shape[1:])
    conv_activations, base_features = feature_extractor(inputs)
    predictions = forward_from_features(base_features)

    grad_model = tf.keras.Model(
        inputs=inputs,
        outputs=[conv_activations, predictions],
    )
    return grad_model, conv_layer.name


def make_gradcam_heatmap(
    model: tf.keras.Model,
    image: str | Path | Image.Image | np.ndarray,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    class_index: int | None = None,
) -> tuple[np.ndarray, float]:
    """
    Generate a Grad-CAM heatmap for a binary sigmoid classifier.

    Args:
        model: Trained Keras model with a sigmoid output.
        image: Image path, PIL image, or RGB NumPy array.
        image_size: Expected (height, width).
        class_index: 1 for the positive class, 0 for the negative class.
            Defaults to the predicted class.

    Returns:
        A normalized heatmap in [0, 1] and the model's sigmoid score.
    """
    batch, _ = _prepare_image(image, image_size)
    grad_model, _ = _build_gradcam_model(model)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(batch, training=False)
        score = tf.reshape(predictions, [-1])[0]

        if class_index not in (None, 0, 1):
            raise ValueError("class_index must be None, 0, or 1.")

        if class_index == 0:
            target_score = 1.0 - score
        elif class_index == 1:
            target_score = score
        else:
            target_score = tf.where(score >= 0.5, score, 1.0 - score)

    gradients = tape.gradient(target_score, conv_outputs)

    if gradients is None:
        raise RuntimeError(
            "Gradients could not be calculated for the selected "
            "convolutional layer. Check the model architecture."
        )

    # Average gradients over spatial positions to obtain channel weights.
    channel_weights = tf.reduce_mean(gradients, axis=(1, 2))
    feature_maps = conv_outputs[0]
    weights = channel_weights[0]

    heatmap = tf.reduce_sum(
        feature_maps * weights[tf.newaxis, tf.newaxis, :],
        axis=-1,
    )
    heatmap = tf.nn.relu(heatmap)

    max_value = tf.reduce_max(heatmap)
    heatmap = tf.where(
        max_value > 0,
        heatmap / (max_value + tf.keras.backend.epsilon()),
        tf.zeros_like(heatmap),
    )

    return heatmap.numpy(), float(score.numpy())


def save_gradcam_visualization(
    model: tf.keras.Model,
    image: str | Path | Image.Image | np.ndarray,
    output_path: str | Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    class_index: int | None = None,
    alpha: float = 0.4,
) -> dict[str, Any]:
    """Save the original image, Grad-CAM heatmap, and overlay."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    batch, display_image = _prepare_image(image, image_size)
    heatmap, score = make_gradcam_heatmap(
        model,
        batch[0],
        image_size=image_size,
        class_index=class_index,
    )

    resized_heatmap = tf.image.resize(
        heatmap[..., np.newaxis],
        image_size,
        method="bilinear",
    ).numpy().squeeze()

    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predicted_class = "PNEUMONIA" if score >= 0.5 else "NORMAL"

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(display_image)
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    axes[1].imshow(resized_heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM heatmap")
    axes[1].axis("off")

    axes[2].imshow(display_image)
    axes[2].imshow(resized_heatmap, cmap="jet", alpha=alpha, vmin=0, vmax=1)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(
        f"Predicted class: {predicted_class} | "
        f"Sigmoid score: {score:.3f}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "output_path": str(output_path),
        "predicted_class": predicted_class,
        "pneumonia_score": score,
        "heatmap_min": float(resized_heatmap.min()),
        "heatmap_max": float(resized_heatmap.max()),
    }


def analyze_gradcam_quantitative(
    model: tf.keras.Model,
    data_generator: Any,
    num_samples: int = 50,
    exp_dir: str | Path | None = None,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> dict[str, Any]:
    """
    Generate Grad-CAM maps for a sample of a Keras data generator.

    Saves individual overlays and a JSON summary. The summary reports
    prediction counts and heatmap statistics, not clinical validity.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")

    output_dir = _resolve_output_dir(exp_dir)
    gradcam_dir = output_dir / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    if len(data_generator) == 0:
        raise ValueError("The data generator contains no batches.")

    if hasattr(data_generator, "reset"):
        data_generator.reset()

    total_samples = min(
        num_samples,
        int(getattr(data_generator, "samples", num_samples)),
    )

    rows: list[dict[str, Any]] = []
    saved = 0

    for batch_index in range(len(data_generator)):
        batch = data_generator[batch_index]

        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError(
                "Expected each generator batch to contain images and labels."
            )

        images, labels = batch[0], batch[1]

        for image_index in range(len(images)):
            if saved >= total_samples:
                break

            image_array = np.asarray(images[image_index], dtype=np.float32)
            label_value = int(np.asarray(labels[image_index]).reshape(-1)[0])

            output_path = gradcam_dir / f"gradcam_{saved + 1:04d}.png"

            try:
                result = save_gradcam_visualization(
                    model,
                    image_array,
                    output_path,
                    image_size=image_size,
                )

                rows.append(
                    {
                        "sample_index": saved,
                        "true_label_index": label_value,
                        "true_label": (
                            "PNEUMONIA" if label_value == 1 else "NORMAL"
                        ),
                        **result,
                    }
                )
                saved += 1

            except Exception:
                logger.exception(
                    "Grad-CAM failed for sample %d.", saved
                )
                rows.append(
                    {
                        "sample_index": saved,
                        "true_label_index": label_value,
                        "error": "Grad-CAM generation failed; see logs.",
                    }
                )
                saved += 1

        if saved >= total_samples:
            break

    successful = [
        row for row in rows if "pneumonia_score" in row
    ]

    summary = {
        "requested_samples": num_samples,
        "processed_samples": saved,
        "successful_visualizations": len(successful),
        "failed_visualizations": saved - len(successful),
        "predicted_normal": sum(
            row["predicted_class"] == "NORMAL" for row in successful
        ),
        "predicted_pneumonia": sum(
            row["predicted_class"] == "PNEUMONIA" for row in successful
        ),
        "mean_pneumonia_score": (
            float(np.mean([row["pneumonia_score"] for row in successful]))
            if successful
            else None
        ),
        "samples": rows,
    }

    summary_path = output_dir / "gradcam_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)

    logger.info(
        "Grad-CAM complete: %d/%d visualizations succeeded. Results: %s",
        len(successful),
        saved,
        gradcam_dir,
    )

    return summary