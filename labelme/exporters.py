from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable
from typing import Sequence

import numpy as np
import PIL.Image

from labelme import utils

_VALID_EXTENSIONS: tuple[str, ...] = (".onnx", ".pt", ".pth", ".safetensors")


def _sanitize_label(label: str) -> str:
    sanitized: str = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return sanitized or "label"


def _shape_to_mask(
    image_shape: tuple[int, int], shape: dict
) -> np.ndarray:
    mask: np.ndarray
    if shape.get("shape_type") == "mask" and shape.get("mask") is not None:
        mask = np.zeros(image_shape, dtype=bool)
        points = np.asarray(shape.get("points"), dtype=int)
        if points.size != 4:
            return mask
        (x1, y1), (x2, y2) = points
        y1 = max(0, min(image_shape[0], y1))
        y2 = max(0, min(image_shape[0], y2))
        x1 = max(0, min(image_shape[1], x1))
        x2 = max(0, min(image_shape[1], x2))
        mask[y1 : y2 + 1, x1 : x2 + 1] = np.asarray(shape.get("mask"), dtype=bool)
        return mask

    mask = utils.shape_to_mask(
        image_shape,
        points=shape.get("points", []),
        shape_type=shape.get("shape_type"),
    )
    return mask.astype(bool)


def _shape_to_bbox(
    image_shape: tuple[int, int], shape: dict
) -> tuple[float, float, float, float] | None:
    if shape.get("shape_type") == "point":
        return None

    if shape.get("shape_type") == "mask" and shape.get("mask") is not None:
        mask = _shape_to_mask(image_shape, shape)
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            return None
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
    else:
        points = np.asarray(shape.get("points", []), dtype=float)
        if points.size == 0:
            return None
        xs = points[:, 0]
        ys = points[:, 1]
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

    height, width = image_shape
    x1 = float(np.clip(x1, 0, width))
    x2 = float(np.clip(x2, 0, width))
    y1 = float(np.clip(y1, 0, height))
    y2 = float(np.clip(y2, 0, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def export_formats(
    base_path: str | os.PathLike[str],
    image: np.ndarray,
    shapes: Sequence[dict],
    formats: Iterable[str],
    label_names: Sequence[str],
) -> None:
    normalized_formats = {fmt.lower() for fmt in formats if fmt}
    normalized_formats.discard("json")
    if not normalized_formats:
        return

    path = Path(base_path)
    image_shape = image.shape[:2]

    class_names = [
        name for name in label_names if name not in (None, "_background_")
    ]
    class_map = {name: idx for idx, name in enumerate(class_names)}

    if "yolo" in normalized_formats:
        _export_yolo(path, image_shape, shapes, class_map)
    if "unet" in normalized_formats:
        _export_unet(path, image_shape, shapes)


def _export_yolo(
    base_path: Path,
    image_shape: tuple[int, int],
    shapes: Sequence[dict],
    class_map: dict[str, int],
) -> None:
    if not class_map:
        return

    yolo_dir = base_path.parent / "yolo"
    _ensure_directory(yolo_dir)
    label_file = yolo_dir / f"{base_path.stem}.txt"

    height, width = image_shape
    lines: list[str] = []
    for shape in shapes:
        label = shape.get("label")
        if label not in class_map:
            continue
        bbox = _shape_to_bbox(image_shape, shape)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        x_center = ((x1 + x2) / 2) / width
        y_center = ((y1 + y2) / 2) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        lines.append(
            f"{class_map[label]} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )

    label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    classes_file = yolo_dir / "classes.txt"
    ordered_classes = [
        name for name, _ in sorted(class_map.items(), key=lambda item: item[1])
    ]
    classes_file.write_text("\n".join(ordered_classes) + "\n", encoding="utf-8")


def _export_unet(
    base_path: Path,
    image_shape: tuple[int, int],
    shapes: Sequence[dict],
) -> None:
    mask_dir = base_path.parent / "unet"
    _ensure_directory(mask_dir)

    label_masks: dict[str, np.ndarray] = {}
    for shape in shapes:
        label = shape.get("label")
        if not label:
            continue
        mask = _shape_to_mask(image_shape, shape)
        if not mask.any():
            continue
        if label not in label_masks:
            label_masks[label] = np.zeros(image_shape, dtype=np.uint8)
        label_masks[label] = np.maximum(label_masks[label], mask.astype(np.uint8))

    for label, mask in label_masks.items():
        if not mask.any():
            continue
        filename = mask_dir / f"{base_path.stem}_{_sanitize_label(label)}.png"
        PIL.Image.fromarray(mask * 255).save(filename)


def contains_model_files(directory: str | os.PathLike[str], keyword: str | None = None) -> bool:
    path = Path(directory).expanduser()
    if not path.exists():
        return False

    keyword_lower = keyword.lower() if keyword else None
    for extension in _VALID_EXTENSIONS:
        for file in path.rglob(f"*{extension}"):
            if keyword_lower is None or keyword_lower in file.stem.lower():
                return True
    return False
