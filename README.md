python add_text_zones_to_yolo_dataset.py \
  --images-dir master_dataset/images \
  --labels-dir master_dataset/labels \
  --model runs/detect/train/weights/best.pt \
  --output-labels-dir master_dataset/labels_with_text \
  --container-class 0 \
  --text-class 1 \
  --pad-x-ratio 0.10 \
  --pad-y-ratio 0.10 \
  --conf 0.30 \
  --device 0 \
  --dry-run




python add_text_zones_to_yolo_dataset.py \
  --images-dir master_dataset/images \
  --labels-dir master_dataset/labels \
  --model runs/detect/train/weights/best.pt \
  --output-labels-dir master_dataset/labels_with_text \
  --container-class 0 \
  --text-class 1 \
  --pad-x-ratio 0.10 \
  --pad-y-ratio 0.10 \
  --conf 0.30 \
  --device 0 \
  --save-debug \
  --debug-dir debug_text_zones






#!/usr/bin/env python3
"""
Добавляет в существующий YOLO-dataset зоны текста, найденные второй YOLO-моделью.

Исходная разметка:
    <container_class> x_center y_center width height

Выходная разметка:
    исходные bbox + <text_class> x_center y_center width height

Модель зон текста запускается не на полном изображении, а на расширенном кропе
каждого bbox номера контейнера.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class Annotation:
    class_id: int
    xc: float
    yc: float
    width: float
    height: float


@dataclass(frozen=True)
class TextPrediction:
    xyxy: tuple[float, float, float, float]
    confidence: float
    detector_class: int


@dataclass
class Statistics:
    images_found: int = 0
    images_processed: int = 0
    images_without_labels: int = 0
    images_without_parent_boxes: int = 0
    parent_boxes_processed: int = 0
    raw_detections: int = 0
    filtered_by_class: int = 0
    filtered_by_geometry: int = 0
    duplicates_removed: int = 0
    text_boxes_added: int = 0
    errors: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Добавить bbox зон текста в YOLO-разметку, запуская модель "
            "на расширенных кропах bbox номера контейнера."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Обязательные пути.
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Корень папки с исходными изображениями.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
        help="Корень папки с существующими YOLO txt-аннотациями.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Путь к weights второй YOLO-модели, например best.pt.",
    )
    parser.add_argument(
        "--output-labels-dir",
        type=Path,
        required=True,
        help="Новая папка для объединённой разметки. Исходные labels не меняются.",
    )

    # Классы итогового датасета.
    parser.add_argument(
        "--container-class",
        type=int,
        default=0,
        help="ID существующего класса bbox номера контейнера.",
    )
    parser.add_argument(
        "--text-class",
        type=int,
        default=1,
        help="ID добавляемого класса зоны текста в итоговом датасете.",
    )
    parser.add_argument(
        "--detector-classes",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Классы второй модели, которые считать зонами текста. "
            "Если не указано, принимаются все классы модели."
        ),
    )

    # Расширение исходного bbox перед созданием кропа.
    parser.add_argument(
        "--pad-x-ratio",
        type=float,
        default=0.10,
        help="Расширение bbox слева и справа как доля ширины исходного bbox.",
    )
    parser.add_argument(
        "--pad-y-ratio",
        type=float,
        default=0.10,
        help="Расширение bbox сверху и снизу как доля высоты исходного bbox.",
    )
    parser.add_argument(
        "--pad-x-pixels",
        type=int,
        default=0,
        help="Дополнительное расширение слева и справа в пикселях.",
    )
    parser.add_argument(
        "--pad-y-pixels",
        type=int,
        default=0,
        help="Дополнительное расширение сверху и снизу в пикселях.",
    )

    # Необязательный ручной resize кропа.
    parser.add_argument(
        "--resize-crop-width",
        type=int,
        default=None,
        help=(
            "Принудительная ширина кропа перед моделью. Использовать только если "
            "обучающие кропы предварительно растягивались вручную."
        ),
    )
    parser.add_argument(
        "--resize-crop-height",
        type=int,
        default=None,
        help="Принудительная высота кропа перед моделью.",
    )

    # Параметры инференса Ultralytics.
    parser.add_argument("--conf", type=float, default=0.25, help="Порог confidence.")
    parser.add_argument("--iou", type=float, default=0.70, help="IoU для NMS модели.")
    parser.add_argument("--imgsz", type=int, default=640, help="Размер инференса YOLO.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Устройство: "cpu", "0", "0,1" и т. п. Если не задано — выбор Ultralytics.',
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=100,
        help="Максимум предсказаний второй модели на один кроп.",
    )
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Использовать FP16, если устройство и модель это поддерживают.",
    )
    parser.add_argument(
        "--agnostic-nms",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Class-agnostic NMS внутри второй модели.",
    )

    # Геометрические фильтры.
    parser.add_argument(
        "--min-inside-ratio",
        type=float,
        default=0.50,
        help=(
            "Минимальная доля площади найденной зоны, находящаяся внутри "
            "исходного bbox номера. 0 отключает проверку."
        ),
    )
    parser.add_argument(
        "--clip-to-container",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Обрезать добавляемые зоны по границам исходного bbox номера.",
    )
    parser.add_argument(
        "--min-width-px",
        type=float,
        default=2.0,
        help="Минимальная ширина итоговой зоны в пикселях.",
    )
    parser.add_argument(
        "--min-height-px",
        type=float,
        default=2.0,
        help="Минимальная высота итоговой зоны в пикселях.",
    )
    parser.add_argument(
        "--dedup-iou",
        type=float,
        default=0.70,
        help=(
            "Удалять повторные добавляемые bbox с IoU не ниже этого значения. "
            "Значение 1 отключает практическое удаление дублей."
        ),
    )

    # Поведение при повторном запуске.
    parser.add_argument(
        "--replace-existing-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Удалять из исходной разметки старые объекты text-class перед "
            "добавлением новых. Это предотвращает накопление дублей."
        ),
    )

    # Поиск файлов и отладка.
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Искать изображения во вложенных папках.",
    )
    parser.add_argument(
        "--image-exts",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".webp"],
        help="Расширения изображений.",
    )
    parser.add_argument(
        "--save-debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Сохранять изображения с нарисованными bbox.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("debug_text_zones"),
        help="Папка для отладочных изображений.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Остановить весь процесс при первой ошибке.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Выполнить обработку без записи labels и debug-изображений.",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.images_dir.is_dir():
        raise ValueError(f"Папка images не найдена: {args.images_dir}")
    if not args.labels_dir.is_dir():
        raise ValueError(f"Папка labels не найдена: {args.labels_dir}")
    if not args.model.is_file():
        raise ValueError(f"Файл модели не найден: {args.model}")

    if args.images_dir.resolve() == args.output_labels_dir.resolve():
        raise ValueError("output-labels-dir не должна совпадать с images-dir.")

    for name in ("pad_x_ratio", "pad_y_ratio"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} не может быть отрицательным.")

    for name in ("pad_x_pixels", "pad_y_pixels"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} не может быть отрицательным.")

    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf должен быть в диапазоне 0..1.")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou должен быть в диапазоне 0..1.")
    if not 0.0 <= args.min_inside_ratio <= 1.0:
        raise ValueError("--min-inside-ratio должен быть в диапазоне 0..1.")
    if not 0.0 <= args.dedup_iou <= 1.0:
        raise ValueError("--dedup-iou должен быть в диапазоне 0..1.")
    if args.imgsz <= 0 or args.max_det <= 0:
        raise ValueError("--imgsz и --max-det должны быть больше нуля.")

    resize_values = (args.resize_crop_width, args.resize_crop_height)
    if (resize_values[0] is None) != (resize_values[1] is None):
        raise ValueError(
            "--resize-crop-width и --resize-crop-height должны задаваться вместе."
        )
    if resize_values[0] is not None and (
        resize_values[0] <= 0 or resize_values[1] <= 0
    ):
        raise ValueError("Размер ручного resize должен быть больше нуля.")

    if args.container_class < 0 or args.text_class < 0:
        raise ValueError("ID классов не могут быть отрицательными.")
    if args.container_class == args.text_class:
        raise ValueError("--container-class и --text-class должны отличаться.")


def discover_images(
    images_dir: Path, extensions: Iterable[str], recursive: bool
) -> list[Path]:
    normalized_exts = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions
    }
    iterator = images_dir.rglob("*") if recursive else images_dir.glob("*")
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in normalized_exts
    )


def read_yolo_labels(label_path: Path) -> list[Annotation]:
    if not label_path.exists():
        raise FileNotFoundError(str(label_path))

    annotations: list[Annotation] = []
    text = label_path.read_text(encoding="utf-8").strip()

    if not text:
        return annotations

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        parts = raw_line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{label_path}, строка {line_number}: ожидалось 5 значений "
                f"detect-разметки, получено {len(parts)}."
            )

        try:
            class_value = float(parts[0])
            class_id = int(class_value)
            values = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(
                f"{label_path}, строка {line_number}: нечисловое значение."
            ) from exc

        if not math.isclose(class_value, class_id):
            raise ValueError(
                f"{label_path}, строка {line_number}: ID класса должен быть целым."
            )

        xc, yc, width, height = values
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"{label_path}, строка {line_number}: обнаружен NaN или inf."
            )
        if width <= 0 or height <= 0:
            raise ValueError(
                f"{label_path}, строка {line_number}: ширина и высота должны быть > 0."
            )
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(
                f"{label_path}, строка {line_number}: координаты должны быть в 0..1."
            )

        annotations.append(Annotation(class_id, xc, yc, width, height))

    return annotations


def yolo_to_xyxy(
    annotation: Annotation, image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    box_width = annotation.width * image_width
    box_height = annotation.height * image_height
    center_x = annotation.xc * image_width
    center_y = annotation.yc * image_height

    return (
        center_x - box_width / 2,
        center_y - box_height / 2,
        center_x + box_width / 2,
        center_y + box_height / 2,
    )


def xyxy_to_yolo(
    class_id: int,
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> Annotation:
    x1, y1, x2, y2 = box
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    xc = x1 + width / 2
    yc = y1 + height / 2

    return Annotation(
        class_id=class_id,
        xc=min(1.0, max(0.0, xc / image_width)),
        yc=min(1.0, max(0.0, yc / image_height)),
        width=min(1.0, max(0.0, width / image_width)),
        height=min(1.0, max(0.0, height / image_height)),
    )


def clip_box(
    box: tuple[float, float, float, float],
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        min(x_max, max(x_min, x1)),
        min(y_max, max(y_min, y1)),
        min(x_max, max(x_min, x2)),
        min(y_max, max(y_min, y2)),
    )


def intersection_box(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> Optional[tuple[float, float, float, float]]:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection = intersection_box(first, second)
    if intersection is None:
        return 0.0

    intersection_area = box_area(intersection)
    union = box_area(first) + box_area(second) - intersection_area
    return intersection_area / union if union > 0 else 0.0


def inside_ratio(
    child: tuple[float, float, float, float],
    parent: tuple[float, float, float, float],
) -> float:
    child_area = box_area(child)
    if child_area <= 0:
        return 0.0

    intersection = intersection_box(child, parent)
    return box_area(intersection) / child_area if intersection else 0.0


def expand_parent_box(
    parent: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    pad_x_ratio: float,
    pad_y_ratio: float,
    pad_x_pixels: int,
    pad_y_pixels: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = parent
    width = x2 - x1
    height = y2 - y1

    pad_x = width * pad_x_ratio + pad_x_pixels
    pad_y = height * pad_y_ratio + pad_y_pixels

    crop_x1 = max(0, math.floor(x1 - pad_x))
    crop_y1 = max(0, math.floor(y1 - pad_y))
    crop_x2 = min(image_width, math.ceil(x2 + pad_x))
    crop_y2 = min(image_height, math.ceil(y2 + pad_y))

    return crop_x1, crop_y1, crop_x2, crop_y2


def deduplicate_predictions(
    predictions: list[TextPrediction], iou_threshold: float
) -> tuple[list[TextPrediction], int]:
    if not predictions:
        return [], 0

    ordered = sorted(predictions, key=lambda item: item.confidence, reverse=True)
    kept: list[TextPrediction] = []
    removed = 0

    for candidate in ordered:
        if any(box_iou(candidate.xyxy, accepted.xyxy) >= iou_threshold for accepted in kept):
            removed += 1
            continue
        kept.append(candidate)

    return kept, removed


def format_annotation(annotation: Annotation) -> str:
    return (
        f"{annotation.class_id} "
        f"{annotation.xc:.6f} "
        f"{annotation.yc:.6f} "
        f"{annotation.width:.6f} "
        f"{annotation.height:.6f}"
    )


def atomic_write_labels(path: Path, annotations: list[Annotation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    content = "\n".join(format_annotation(item) for item in annotations)
    if content:
        content += "\n"

    temp_path.write_text(content, encoding="utf-8")
    os.replace(temp_path, path)


def draw_debug(
    image: np.ndarray,
    parent_boxes: list[tuple[float, float, float, float]],
    crop_boxes: list[tuple[int, int, int, int]],
    predictions: list[TextPrediction],
) -> np.ndarray:
    output = image.copy()

    # Синий: исходный bbox номера контейнера.
    for x1, y1, x2, y2 in parent_boxes:
        cv2.rectangle(
            output,
            (round(x1), round(y1)),
            (round(x2), round(y2)),
            (255, 0, 0),
            2,
        )

    # Жёлтый: расширенный кроп, поданный во вторую модель.
    for x1, y1, x2, y2 in crop_boxes:
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 1)

    # Красный: добавленная зона текста.
    for prediction in predictions:
        x1, y1, x2, y2 = prediction.xyxy
        cv2.rectangle(
            output,
            (round(x1), round(y1)),
            (round(x2), round(y2)),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            output,
            f"text {prediction.confidence:.2f}",
            (round(x1), max(15, round(y1) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def predict_text_zones(
    model: YOLO,
    image: np.ndarray,
    parent_box: tuple[float, float, float, float],
    args: argparse.Namespace,
    stats: Statistics,
) -> tuple[list[TextPrediction], tuple[int, int, int, int]]:
    image_height, image_width = image.shape[:2]

    crop_box = expand_parent_box(
        parent=parent_box,
        image_width=image_width,
        image_height=image_height,
        pad_x_ratio=args.pad_x_ratio,
        pad_y_ratio=args.pad_y_ratio,
        pad_x_pixels=args.pad_x_pixels,
        pad_y_pixels=args.pad_y_pixels,
    )
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box

    if crop_x2 - crop_x1 < 2 or crop_y2 - crop_y1 < 2:
        return [], crop_box

    original_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    model_input = original_crop
    scale_x = 1.0
    scale_y = 1.0

    if args.resize_crop_width is not None:
        model_input = cv2.resize(
            original_crop,
            (args.resize_crop_width, args.resize_crop_height),
            interpolation=cv2.INTER_LINEAR,
        )
        scale_x = original_crop.shape[1] / args.resize_crop_width
        scale_y = original_crop.shape[0] / args.resize_crop_height

    predict_kwargs = {
        "source": model_input,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "max_det": args.max_det,
        "half": args.half,
        "agnostic_nms": args.agnostic_nms,
        "verbose": False,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    results = model.predict(**predict_kwargs)
    if not results or results[0].boxes is None:
        return [], crop_box

    boxes = results[0].boxes
    if len(boxes) == 0:
        return [], crop_box

    xyxy_array = boxes.xyxy.detach().cpu().numpy()
    confidence_array = boxes.conf.detach().cpu().numpy()
    class_array = boxes.cls.detach().cpu().numpy().astype(int)

    accepted_classes = (
        set(args.detector_classes) if args.detector_classes is not None else None
    )

    predictions: list[TextPrediction] = []

    for local_box, confidence, detector_class in zip(
        xyxy_array, confidence_array, class_array
    ):
        stats.raw_detections += 1

        if accepted_classes is not None and detector_class not in accepted_classes:
            stats.filtered_by_class += 1
            continue

        local_x1, local_y1, local_x2, local_y2 = map(float, local_box)

        # Возврат из ручного resize к размеру расширенного кропа.
        local_x1 *= scale_x
        local_x2 *= scale_x
        local_y1 *= scale_y
        local_y2 *= scale_y

        global_box = (
            crop_x1 + local_x1,
            crop_y1 + local_y1,
            crop_x1 + local_x2,
            crop_y1 + local_y2,
        )
        global_box = clip_box(global_box, 0.0, 0.0, image_width, image_height)

        if inside_ratio(global_box, parent_box) < args.min_inside_ratio:
            stats.filtered_by_geometry += 1
            continue

        if args.clip_to_container:
            clipped = intersection_box(global_box, parent_box)
            if clipped is None:
                stats.filtered_by_geometry += 1
                continue
            global_box = clipped

        width = global_box[2] - global_box[0]
        height = global_box[3] - global_box[1]
        if width < args.min_width_px or height < args.min_height_px:
            stats.filtered_by_geometry += 1
            continue

        predictions.append(
            TextPrediction(
                xyxy=global_box,
                confidence=float(confidence),
                detector_class=int(detector_class),
            )
        )

    return predictions, crop_box


def process_image(
    image_path: Path,
    model: YOLO,
    args: argparse.Namespace,
    stats: Statistics,
) -> None:
    relative_path = image_path.relative_to(args.images_dir)
    label_path = args.labels_dir / relative_path.with_suffix(".txt")
    output_label_path = args.output_labels_dir / relative_path.with_suffix(".txt")

    if not label_path.exists():
        stats.images_without_labels += 1
        print(f"[SKIP] Нет label: {label_path}")
        return

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV не смог прочитать изображение: {image_path}")

    image_height, image_width = image.shape[:2]
    annotations = read_yolo_labels(label_path)

    parent_annotations = [
        item for item in annotations if item.class_id == args.container_class
    ]

    if not parent_annotations:
        stats.images_without_parent_boxes += 1

    if args.replace_existing_text:
        base_annotations = [
            item for item in annotations if item.class_id != args.text_class
        ]
        old_text_boxes: list[tuple[float, float, float, float]] = []
    else:
        base_annotations = list(annotations)
        old_text_boxes = [
            yolo_to_xyxy(item, image_width, image_height)
            for item in annotations
            if item.class_id == args.text_class
        ]

    parent_boxes: list[tuple[float, float, float, float]] = []
    crop_boxes: list[tuple[int, int, int, int]] = []
    new_predictions: list[TextPrediction] = []

    for parent_annotation in parent_annotations:
        raw_parent_box = yolo_to_xyxy(
            parent_annotation, image_width, image_height
        )
        parent_box = clip_box(
            raw_parent_box, 0.0, 0.0, image_width, image_height
        )

        if box_area(parent_box) <= 0:
            stats.filtered_by_geometry += 1
            continue

        stats.parent_boxes_processed += 1
        parent_boxes.append(parent_box)

        predictions, crop_box = predict_text_zones(
            model=model,
            image=image,
            parent_box=parent_box,
            args=args,
            stats=stats,
        )
        crop_boxes.append(crop_box)
        new_predictions.extend(predictions)

    new_predictions, removed = deduplicate_predictions(
        new_predictions, args.dedup_iou
    )
    stats.duplicates_removed += removed

    # Если старые text-class оставляются, не добавляем к ним почти идентичные новые bbox.
    if old_text_boxes:
        filtered_predictions: list[TextPrediction] = []
        for prediction in new_predictions:
            if any(
                box_iou(prediction.xyxy, old_box) >= args.dedup_iou
                for old_box in old_text_boxes
            ):
                stats.duplicates_removed += 1
            else:
                filtered_predictions.append(prediction)
        new_predictions = filtered_predictions

    added_annotations = [
        xyxy_to_yolo(
            class_id=args.text_class,
            box=prediction.xyxy,
            image_width=image_width,
            image_height=image_height,
        )
        for prediction in new_predictions
    ]

    output_annotations = base_annotations + added_annotations

    if not args.dry_run:
        atomic_write_labels(output_label_path, output_annotations)

        if args.save_debug:
            debug_path = args.debug_dir / relative_path
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_image = draw_debug(
                image=image,
                parent_boxes=parent_boxes,
                crop_boxes=crop_boxes,
                predictions=new_predictions,
            )
            if not cv2.imwrite(str(debug_path), debug_image):
                raise OSError(f"Не удалось записать debug: {debug_path}")

    stats.text_boxes_added += len(added_annotations)
    stats.images_processed += 1

    print(
        f"[OK] {relative_path} | parents={len(parent_boxes)} "
        f"| text_added={len(added_annotations)}"
    )


def print_summary(stats: Statistics, args: argparse.Namespace) -> None:
    mode = "DRY RUN" if args.dry_run else "ЗАПИСЬ ВЫПОЛНЕНА"
    print("\n" + "=" * 64)
    print(f"РЕЗУЛЬТАТ: {mode}")
    print(f"Найдено изображений:              {stats.images_found}")
    print(f"Успешно обработано:               {stats.images_processed}")
    print(f"Без соответствующего label:       {stats.images_without_labels}")
    print(f"Без bbox container-class:         {stats.images_without_parent_boxes}")
    print(f"Обработано bbox контейнеров:       {stats.parent_boxes_processed}")
    print(f"Сырых detections второй модели:   {stats.raw_detections}")
    print(f"Отфильтровано по классу:          {stats.filtered_by_class}")
    print(f"Отфильтровано по геометрии:       {stats.filtered_by_geometry}")
    print(f"Удалено дублей:                   {stats.duplicates_removed}")
    print(f"Добавлено bbox зон текста:        {stats.text_boxes_added}")
    print(f"Ошибок:                           {stats.errors}")
    if not args.dry_run:
        print(f"Выходные labels:                  {args.output_labels_dir}")
        if args.save_debug:
            print(f"Debug-изображения:                {args.debug_dir}")
    print("=" * 64)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    if not args.dry_run:
        args.output_labels_dir.mkdir(parents=True, exist_ok=True)
        if args.save_debug:
            args.debug_dir.mkdir(parents=True, exist_ok=True)

    images = discover_images(
        images_dir=args.images_dir,
        extensions=args.image_exts,
        recursive=args.recursive,
    )

    stats = Statistics(images_found=len(images))

    if not images:
        print("Изображения не найдены.", file=sys.stderr)
        return 2

    print(f"Загрузка модели: {args.model}")
    model = YOLO(str(args.model))

    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {image_path}")
        try:
            process_image(
                image_path=image_path,
                model=model,
                args=args,
                stats=stats,
            )
        except Exception as exc:
            stats.errors += 1
            print(f"[ERROR] {image_path}: {exc}", file=sys.stderr)
            if args.strict:
                raise

    print_summary(stats, args)
    return 0 if stats.errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
