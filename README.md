#!/usr/bin/env python3
"""
Анализ кадров контейнеров обученной YOLO-моделью.

Скрипт рекурсивно ищет изображения внутри связок device_* / cam_*,
копирует их в результат с группировкой по качеству детекции, сохраняет
label-файлы и формирует CSV-отчеты.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - проверяется при запуске у пользователя
    YOLO = None  # type: ignore[assignment]


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RESULT_GROUPS = ("zero_objects", "low_conf", "single_good", "multiple_objects")
DEFAULT_CLASS_NAMES = {
    0: "end_full_readable",
    1: "end_partial_visible",
    2: "end_unreadable",
    3: "side_full_readable",
    4: "side_partial_visible",
    5: "side_unreadable",
}


@dataclass(frozen=True)
class ImageInfo:
    """Метаданные изображения, извлеченные из пути."""

    source_path: Path
    device_id: str
    camera_id: str
    frame_number: int | None
    image_name: str


@dataclass(frozen=True)
class Detection:
    """Одна детекция YOLO в нормализованном формате."""

    class_id: int
    class_name: str
    confidence: float
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class ProcessedImage:
    """Итог обработки одного изображения."""

    info: ImageInfo
    detections: list[Detection]
    result_group: str
    saved_image_path: Path
    saved_label_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Анализ изображений контейнеров через ultralytics.YOLO."
    )
    parser.add_argument("--weights", required=True, type=Path, help="Путь к весам YOLO")
    parser.add_argument("--src", required=True, type=Path, help="Корневая папка с кадрами")
    parser.add_argument("--out", required=True, type=Path, help="Папка результата")
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.25,
        help="Минимальная confidence для учета детекции",
    )
    parser.add_argument(
        "--good-conf",
        type=float,
        default=0.50,
        help="Порог хорошей confidence",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Устройство инференса: cpu, cuda:0 и т.д.",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Необязательный список классов через запятую: 0,1 или имена классов",
    )
    parser.add_argument(
        "--frame-regex",
        default=None,
        help=(
            "Необязательное regex-правило для номера фрейма. "
            "Если есть группа, берется первая группа. Пример: 'img_(\\d+)'"
        ),
    )
    return parser.parse_args()


def validate_common_args(args: argparse.Namespace) -> None:
    if YOLO is None:
        raise RuntimeError(
            "Не установлен пакет ultralytics. Установите его: pip install ultralytics"
        )
    if not args.weights.is_file():
        raise FileNotFoundError(f"Файл весов не найден: {args.weights}")
    if not args.src.is_dir():
        raise NotADirectoryError(f"Исходная папка не найдена: {args.src}")
    if not 0.0 <= args.min_conf <= 1.0:
        raise ValueError("--min-conf должен быть в диапазоне от 0 до 1")
    if not 0.0 <= args.good_conf <= 1.0:
        raise ValueError("--good-conf должен быть в диапазоне от 0 до 1")
    if args.min_conf > args.good_conf:
        raise ValueError("--min-conf не должен быть больше --good-conf")
    if args.frame_regex:
        try:
            re.compile(args.frame_regex)
        except re.error as exc:
            raise ValueError(f"Некорректный --frame-regex: {exc}") from exc


def sorted_paths(paths: Iterable[Path]) -> list[Path]:
    return sorted(paths, key=lambda p: str(p).lower())


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def is_device_dir_name(name: str) -> bool:
    return name.lower().startswith("device_")


def is_camera_dir_name(name: str) -> bool:
    return name.lower().startswith("cam_")


def int_from_regex_match(match: re.Match[str]) -> int | None:
    value = match.group(1) if match.groups() else match.group(0)
    number_match = re.search(r"\d+", value)
    return int(number_match.group(0)) if number_match else None


def extract_frame_number(path: Path, frame_regex: str | None = None) -> int | None:
    path_parts = [path.stem, *reversed(path.parent.parts)]

    if frame_regex:
        for part in path_parts:
            custom_match = re.search(frame_regex, part, flags=re.IGNORECASE)
            if custom_match:
                return int_from_regex_match(custom_match)

    for part in path_parts:
        frame_match = re.search(r"frame[_-]?(\d+)", part, flags=re.IGNORECASE)
        if frame_match:
            return int(frame_match.group(1))

    number_matches = re.findall(r"\d+", path.stem)
    if number_matches:
        return int(number_matches[-1])

    return None


def extract_image_info(path: Path, frame_regex: str | None = None) -> ImageInfo | None:
    parts = path.parts
    device_index = next(
        (idx for idx, part in enumerate(parts) if is_device_dir_name(part)), None
    )
    if device_index is None:
        return None

    camera_index = next(
        (
            idx
            for idx in range(device_index + 1, len(parts))
            if is_camera_dir_name(parts[idx])
        ),
        None,
    )
    if camera_index is None:
        return None

    return ImageInfo(
        source_path=path,
        device_id=parts[device_index],
        camera_id=parts[camera_index],
        frame_number=extract_frame_number(path, frame_regex),
        image_name=path.name,
    )


def discover_images(
    src: Path, frame_regex: str | None = None
) -> tuple[list[ImageInfo], set[str], set[tuple[str, str]]]:
    devices = {path.name for path in src.rglob("*") if path.is_dir() and is_device_dir_name(path.name)}
    cameras: set[tuple[str, str]] = set()
    images: list[ImageInfo] = []

    for image_path in sorted_paths(path for path in src.rglob("*") if is_image(path)):
        info = extract_image_info(image_path, frame_regex)
        if info is None:
            continue
        images.append(info)
        cameras.add((info.device_id, info.camera_id))

    return images, devices, cameras


def normalize_model_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, (list, tuple)):
        return {idx: str(value) for idx, value in enumerate(raw_names)}
    return DEFAULT_CLASS_NAMES.copy()


def class_name(class_id: int, model_names: dict[int, str]) -> str:
    return model_names.get(class_id, DEFAULT_CLASS_NAMES.get(class_id, str(class_id)))


def parse_classes(raw_classes: str | None, model_names: dict[int, str]) -> list[int] | None:
    if not raw_classes:
        return None

    reverse_names = {name: class_id for class_id, name in model_names.items()}
    parsed: list[int] = []
    for raw_item in raw_classes.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if item.isdigit():
            parsed.append(int(item))
            continue
        if item in reverse_names:
            parsed.append(reverse_names[item])
            continue
        raise ValueError(f"Класс из --classes не найден в модели: {item}")

    return parsed or None


def classify_result(detections: list[Detection], good_conf: float) -> str:
    if not detections:
        return "zero_objects"
    if len(detections) >= 2:
        return "multiple_objects"
    return "single_good" if detections[0].confidence >= good_conf else "low_conf"


def single_detection_class_folder(detections: list[Detection]) -> str | None:
    return detections[0].class_name if len(detections) == 1 else None


def short_path_hash(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]


def destination_paths(
    info: ImageInfo, detections: list[Detection], result_group: str, out_dir: Path
) -> tuple[Path, Path]:
    class_folder = single_detection_class_folder(detections)
    relative_parts = [info.device_id, info.camera_id, result_group]
    if class_folder is not None:
        relative_parts.append(class_folder)

    image_dir = out_dir / "images" / Path(*relative_parts)
    label_dir = out_dir / "labels" / Path(*relative_parts)
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / info.image_name
    label_path = label_dir / f"{info.source_path.stem}.txt"

    if image_path.exists() or label_path.exists():
        suffix = short_path_hash(info.source_path)
        image_path = image_dir / f"{info.source_path.stem}_{suffix}{info.source_path.suffix}"
        label_path = label_dir / f"{info.source_path.stem}_{suffix}.txt"

    return image_path, label_path


def write_label_file(path: Path, detections: list[Detection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as label_file:
        for detection in detections:
            # Формат: class_id x_center y_center width height confidence class_name
            label_file.write(
                f"{detection.class_id} "
                f"{detection.x_center:.6f} "
                f"{detection.y_center:.6f} "
                f"{detection.width:.6f} "
                f"{detection.height:.6f} "
                f"{detection.confidence:.6f} "
                f"{detection.class_name}\n"
            )


def run_yolo_on_image(
    model: Any,
    image_path: Path,
    min_conf: float,
    device: str | None,
    classes: list[int] | None,
    model_names: dict[int, str],
) -> list[Detection]:
    predict_kwargs: dict[str, Any] = {
        "source": str(image_path),
        "conf": min_conf,
        "verbose": False,
    }
    if device:
        predict_kwargs["device"] = device
    if classes is not None:
        predict_kwargs["classes"] = classes

    results = model.predict(**predict_kwargs)
    if not results:
        return []

    result = results[0]
    if result.boxes is None:
        return []

    detections: list[Detection] = []
    for box in result.boxes:
        confidence = float(box.conf[0])
        if confidence < min_conf:
            continue

        class_id = int(box.cls[0])
        x_center, y_center, width, height = (float(value) for value in box.xywhn[0].tolist())
        detections.append(
            Detection(
                class_id=class_id,
                class_name=class_name(class_id, model_names),
                confidence=confidence,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )

    return sorted(detections, key=lambda item: item.confidence, reverse=True)


def ensure_result_directories(out_dir: Path) -> Path:
    reports_dir = out_dir / "reports"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def process_images(
    model: Any,
    images: list[ImageInfo],
    out_dir: Path,
    min_conf: float,
    good_conf: float,
    device: str | None,
    classes: list[int] | None,
    model_names: dict[int, str],
    reports_dir: Path,
) -> tuple[list[ProcessedImage], list[dict[str, str]]]:
    processed: list[ProcessedImage] = []
    errors: list[dict[str, str]] = []
    total = len(images)
    started_at = time.time()

    for index, info in enumerate(images, start=1):
        try:
            detections = run_yolo_on_image(
                model=model,
                image_path=info.source_path,
                min_conf=min_conf,
                device=device,
                classes=classes,
                model_names=model_names,
            )
            result_group = classify_result(detections, good_conf)
            saved_image_path, saved_label_path = destination_paths(
                info=info,
                detections=detections,
                result_group=result_group,
                out_dir=out_dir,
            )
            shutil.copy2(info.source_path, saved_image_path)
            write_label_file(saved_label_path, detections)
            processed.append(
                ProcessedImage(
                    info=info,
                    detections=detections,
                    result_group=result_group,
                    saved_image_path=saved_image_path,
                    saved_label_path=saved_label_path,
                )
            )
        except Exception as exc:  # noqa: BLE001 - важно продолжать обработку
            errors.append(
                {
                    "source_path": str(info.source_path),
                    "device_id": info.device_id,
                    "camera_id": info.camera_id,
                    "frame_number": "" if info.frame_number is None else str(info.frame_number),
                    "image_name": info.image_name,
                    "error": repr(exc),
                }
            )

        show_progress(index=index, total=total, started_at=started_at)

    print()
    write_errors_csv(reports_dir / "errors.csv", errors)
    return processed, errors


def show_progress(index: int, total: int, started_at: float) -> None:
    elapsed = max(time.time() - started_at, 0.001)
    speed = index / elapsed
    percent = index * 100 / total
    message = f"\rОбработка: {index}/{total} ({percent:6.2f}%), {speed:.2f} img/s"
    print(message, end="", flush=True)


def csv_list(values: Iterable[Any]) -> str:
    return ";".join(str(value) for value in values)


def confidences(detections: list[Detection]) -> list[float]:
    return [detection.confidence for detection in detections]


def max_conf(detections: list[Detection]) -> float | None:
    values = confidences(detections)
    return max(values) if values else None


def mean_conf(detections: list[Detection]) -> float | None:
    values = confidences(detections)
    return mean(values) if values else None


def format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_errors_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "source_path",
        "device_id",
        "camera_id",
        "frame_number",
        "image_name",
        "error",
    ]
    write_csv(path, fieldnames, rows)


def detection_row(item: ProcessedImage) -> dict[str, Any]:
    detections = item.detections
    return {
        "source_path": str(item.info.source_path),
        "device_id": item.info.device_id,
        "camera_id": item.info.camera_id,
        "frame_number": "" if item.info.frame_number is None else item.info.frame_number,
        "image_name": item.info.image_name,
        "objects_count": len(detections),
        "result_group": item.result_group,
        "class_ids": csv_list(detection.class_id for detection in detections),
        "class_names": csv_list(detection.class_name for detection in detections),
        "confidences": csv_list(f"{detection.confidence:.6f}" for detection in detections),
        "max_conf": format_float(max_conf(detections)),
        "mean_conf": format_float(mean_conf(detections)),
        "bbox_count": len(detections),
        "saved_image_path": str(item.saved_image_path),
        "saved_label_path": str(item.saved_label_path),
    }


def write_detections_csv(path: Path, processed: list[ProcessedImage]) -> None:
    fieldnames = [
        "source_path",
        "device_id",
        "camera_id",
        "frame_number",
        "image_name",
        "objects_count",
        "result_group",
        "class_ids",
        "class_names",
        "confidences",
        "max_conf",
        "mean_conf",
        "bbox_count",
        "saved_image_path",
        "saved_label_path",
    ]
    write_csv(path, fieldnames, [detection_row(item) for item in processed])


def readable_count(detections: Iterable[Detection]) -> int:
    return sum("full_readable" in detection.class_name for detection in detections)


def partial_visible_count(detections: Iterable[Detection]) -> int:
    return sum("partial_visible" in detection.class_name for detection in detections)


def unreadable_count(detections: Iterable[Detection]) -> int:
    return sum("unreadable" in detection.class_name for detection in detections)


def end_count(detections: Iterable[Detection]) -> int:
    return sum(detection.class_name.startswith("end_") for detection in detections)


def side_count(detections: Iterable[Detection]) -> int:
    return sum(detection.class_name.startswith("side_") for detection in detections)


def camera_sort_key(row: dict[str, Any]) -> tuple[int, float, float, int, int]:
    return (
        int(row["single_good_count"]),
        float(row["good_detection_rate"]),
        float(row["average_confidence"] or 0.0),
        -int(row["zero_objects_count"]),
        -int(row["multiple_objects_count"]),
    )


def write_camera_summary_csv(
    path: Path, processed: list[ProcessedImage]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[ProcessedImage]] = defaultdict(list)
    for item in processed:
        grouped[(item.info.device_id, item.info.camera_id)].append(item)

    rows: list[dict[str, Any]] = []
    for (device_id, camera_id), items in grouped.items():
        group_counter = Counter(item.result_group for item in items)
        all_detections = [
            detection for item in items for detection in item.detections
        ]
        conf_values = [detection.confidence for detection in all_detections]
        total_images = len(items)
        row = {
            "device_id": device_id,
            "camera_id": camera_id,
            "total_images": total_images,
            "zero_objects_count": group_counter["zero_objects"],
            "low_conf_count": group_counter["low_conf"],
            "single_good_count": group_counter["single_good"],
            "multiple_objects_count": group_counter["multiple_objects"],
            "good_detection_rate": f"{group_counter['single_good'] / total_images:.6f}",
            "average_confidence": format_float(mean(conf_values) if conf_values else None),
            "median_confidence": format_float(median(conf_values) if conf_values else None),
            "max_confidence": format_float(max(conf_values) if conf_values else None),
            "readable_count": readable_count(all_detections),
            "partial_visible_count": partial_visible_count(all_detections),
            "unreadable_count": unreadable_count(all_detections),
            "end_count": end_count(all_detections),
            "side_count": side_count(all_detections),
            "camera_rank": 0,
        }
        rows.append(row)

    rows.sort(key=camera_sort_key, reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["camera_rank"] = rank

    fieldnames = [
        "camera_rank",
        "device_id",
        "camera_id",
        "total_images",
        "zero_objects_count",
        "low_conf_count",
        "single_good_count",
        "multiple_objects_count",
        "good_detection_rate",
        "average_confidence",
        "median_confidence",
        "max_confidence",
        "readable_count",
        "partial_visible_count",
        "unreadable_count",
        "end_count",
        "side_count",
    ]
    write_csv(path, fieldnames, rows)
    return rows


def best_single_detection(item: ProcessedImage) -> Detection | None:
    return item.detections[0] if len(item.detections) == 1 else None


def write_frame_summary_csv(path: Path, processed: list[ProcessedImage]) -> None:
    rows: list[dict[str, Any]] = []
    for item in processed:
        detection = best_single_detection(item)
        rows.append(
            {
                "device_id": item.info.device_id,
                "camera_id": item.info.camera_id,
                "frame_number": "" if item.info.frame_number is None else item.info.frame_number,
                "result_group": item.result_group,
                "class_name": "" if detection is None else detection.class_name,
                "confidence": "" if detection is None else f"{detection.confidence:.6f}",
                "source_path": str(item.info.source_path),
            }
        )

    fieldnames = [
        "device_id",
        "camera_id",
        "frame_number",
        "result_group",
        "class_name",
        "confidence",
        "source_path",
    ]
    write_csv(path, fieldnames, rows)


def is_full_readable(detection: Detection) -> bool:
    return "full_readable" in detection.class_name


def orientation(detection: Detection) -> str:
    if detection.class_name.startswith("end_"):
        return "end"
    if detection.class_name.startswith("side_"):
        return "side"
    return "unknown"


def best_frames_rows(processed: list[ProcessedImage]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[ProcessedImage]] = defaultdict(list)
    for item in processed:
        if len(item.detections) == 1:
            grouped[(item.info.device_id, item.info.camera_id)].append(item)

    rows: list[dict[str, Any]] = []
    for (device_id, camera_id), items in grouped.items():
        items_by_conf = sorted(
            items,
            key=lambda item: (
                item.detections[0].confidence,
                -1 if item.info.frame_number is None else item.info.frame_number,
            ),
            reverse=True,
        )

        for rank, item in enumerate(items_by_conf[:10], start=1):
            rows.append(best_frame_row("top_confidence", rank, item))

        readable_items = [
            item for item in items_by_conf if is_full_readable(item.detections[0])
        ]
        end_items = [
            item for item in items_by_conf if orientation(item.detections[0]) == "end"
        ]
        side_items = [
            item for item in items_by_conf if orientation(item.detections[0]) == "side"
        ]

        if readable_items:
            rows.append(best_frame_row("best_readable", 1, readable_items[0]))
        if end_items:
            rows.append(best_frame_row("best_end", 1, end_items[0]))
        if side_items:
            rows.append(best_frame_row("best_side", 1, side_items[0]))

    return rows


def best_frame_row(kind: str, rank: int, item: ProcessedImage) -> dict[str, Any]:
    detection = item.detections[0]
    return {
        "selection_type": kind,
        "rank": rank,
        "device_id": item.info.device_id,
        "camera_id": item.info.camera_id,
        "frame_number": "" if item.info.frame_number is None else item.info.frame_number,
        "class_id": detection.class_id,
        "class_name": detection.class_name,
        "confidence": f"{detection.confidence:.6f}",
        "source_path": str(item.info.source_path),
        "saved_image_path": str(item.saved_image_path),
        "saved_label_path": str(item.saved_label_path),
    }


def write_best_frames_csv(path: Path, processed: list[ProcessedImage]) -> list[dict[str, Any]]:
    rows = best_frames_rows(processed)
    fieldnames = [
        "selection_type",
        "rank",
        "device_id",
        "camera_id",
        "frame_number",
        "class_id",
        "class_name",
        "confidence",
        "source_path",
        "saved_image_path",
        "saved_label_path",
    ]
    write_csv(path, fieldnames, rows)
    return rows


def consecutive_key(item: ProcessedImage) -> tuple[int, str]:
    if item.info.frame_number is None:
        return (sys.maxsize, item.info.image_name)
    return (item.info.frame_number, item.info.image_name)


def candidate_for_best_range(item: ProcessedImage, good_conf: float) -> bool:
    if len(item.detections) != 1:
        return False
    detection = item.detections[0]
    return detection.confidence >= good_conf and is_full_readable(detection)


def best_consecutive_ranges(processed: list[ProcessedImage], good_conf: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[ProcessedImage]] = defaultdict(list)
    for item in processed:
        if item.info.frame_number is not None and candidate_for_best_range(item, good_conf):
            grouped[(item.info.device_id, item.info.camera_id)].append(item)

    rows: list[dict[str, Any]] = []
    for (device_id, camera_id), items in grouped.items():
        ordered = sorted(items, key=consecutive_key)
        if not ordered:
            continue

        ranges: list[list[ProcessedImage]] = []
        current_range = [ordered[0]]
        for item in ordered[1:]:
            previous = current_range[-1]
            if item.info.frame_number == previous.info.frame_number + 1:
                current_range.append(item)
            else:
                ranges.append(current_range)
                current_range = [item]
        ranges.append(current_range)

        best_range = max(
            ranges,
            key=lambda group: (
                len(group),
                mean(item.detections[0].confidence for item in group),
                max(item.detections[0].confidence for item in group),
            ),
        )
        conf_values = [item.detections[0].confidence for item in best_range]
        rows.append(
            {
                "device_id": device_id,
                "camera_id": camera_id,
                "start_frame": best_range[0].info.frame_number,
                "end_frame": best_range[-1].info.frame_number,
                "frames_count": len(best_range),
                "average_confidence": f"{mean(conf_values):.6f}",
                "max_confidence": f"{max(conf_values):.6f}",
            }
        )

    rows.sort(key=lambda row: (row["device_id"], row["camera_id"]))
    return rows


def write_best_ranges_csv(
    path: Path, processed: list[ProcessedImage], good_conf: float
) -> list[dict[str, Any]]:
    rows = best_consecutive_ranges(processed, good_conf)
    fieldnames = [
        "device_id",
        "camera_id",
        "start_frame",
        "end_frame",
        "frames_count",
        "average_confidence",
        "max_confidence",
    ]
    write_csv(path, fieldnames, rows)
    return rows


def relative_uri(target_path: str | Path, base_dir: Path) -> str:
    """Возвращает относительный URI для ссылок из HTML-отчета."""

    try:
        relative = Path(target_path).resolve().relative_to(base_dir.resolve())
    except ValueError:
        try:
            relative = Path(target_path).resolve().relative_to(base_dir.parent.resolve())
            return "../" + relative.as_posix()
        except ValueError:
            return Path(target_path).as_posix()
    return relative.as_posix()


def css_bar(value: float, maximum: float) -> str:
    if maximum <= 0:
        return "0%"
    return f"{min(max(value / maximum, 0.0), 1.0) * 100:.2f}%"


def group_counts(processed: list[ProcessedImage]) -> Counter[str]:
    return Counter(item.result_group for item in processed)


def render_camera_table(camera_rows: list[dict[str, Any]]) -> str:
    max_single_good = max(
        (int(row["single_good_count"]) for row in camera_rows), default=0
    )
    rows_html: list[str] = []
    for row in camera_rows:
        rate = float(row["good_detection_rate"])
        single_good = int(row["single_good_count"])
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(row['camera_rank']))}</td>"
            f"<td>{html.escape(str(row['device_id']))}</td>"
            f"<td>{html.escape(str(row['camera_id']))}</td>"
            f"<td>{html.escape(str(row['total_images']))}</td>"
            f"<td><div class=\"bar\"><span style=\"width:{css_bar(single_good, max_single_good)}\"></span></div>{single_good}</td>"
            f"<td><div class=\"bar good\"><span style=\"width:{rate * 100:.2f}%\"></span></div>{rate:.3f}</td>"
            f"<td>{html.escape(str(row['average_confidence']))}</td>"
            f"<td>{html.escape(str(row['zero_objects_count']))}</td>"
            f"<td>{html.escape(str(row['multiple_objects_count']))}</td>"
            "</tr>"
        )

    return "\n".join(rows_html)


def render_best_frames(best_frame_rows: list[dict[str, Any]], reports_dir: Path) -> str:
    top_rows = [row for row in best_frame_rows if row["selection_type"] == "top_confidence"]
    cards: list[str] = []
    for row in top_rows[:60]:
        image_uri = relative_uri(row["saved_image_path"], reports_dir)
        cards.append(
            "<article class=\"frame-card\">"
            f"<a href=\"{html.escape(image_uri)}\"><img src=\"{html.escape(image_uri)}\" loading=\"lazy\" alt=\"\"></a>"
            "<div class=\"frame-meta\">"
            f"<strong>{html.escape(str(row['device_id']))} / {html.escape(str(row['camera_id']))}</strong>"
            f"<span>rank {html.escape(str(row['rank']))}, frame {html.escape(str(row['frame_number']))}</span>"
            f"<span>{html.escape(str(row['class_name']))}</span>"
            f"<span>conf {html.escape(str(row['confidence']))}</span>"
            "</div>"
            "</article>"
        )
    if not cards:
        return "<p class=\"muted\">Нет кадров с ровно одной детекцией.</p>"
    return "\n".join(cards)


def render_ranges_table(best_range_rows: list[dict[str, Any]]) -> str:
    rows_html: list[str] = []
    for row in best_range_rows:
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(row['device_id']))}</td>"
            f"<td>{html.escape(str(row['camera_id']))}</td>"
            f"<td>{html.escape(str(row['start_frame']))}</td>"
            f"<td>{html.escape(str(row['end_frame']))}</td>"
            f"<td>{html.escape(str(row['frames_count']))}</td>"
            f"<td>{html.escape(str(row['average_confidence']))}</td>"
            f"<td>{html.escape(str(row['max_confidence']))}</td>"
            "</tr>"
        )
    return "\n".join(rows_html)


def write_html_report(
    path: Path,
    processed: list[ProcessedImage],
    camera_rows: list[dict[str, Any]],
    best_frame_rows: list[dict[str, Any]],
    best_range_rows: list[dict[str, Any]],
) -> None:
    counts = group_counts(processed)
    total = len(processed)
    best_camera = camera_rows[0] if camera_rows else None
    best_camera_text = (
        f"{best_camera['device_id']} / {best_camera['camera_id']}"
        if best_camera
        else "нет данных"
    )

    cards = "".join(
        f"<div class=\"metric {group}\"><span>{group}</span><strong>{counts[group]}</strong></div>"
        for group in RESULT_GROUPS
    )
    html_text = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Отчет анализа контейнеров</title>
  <style>
    body {{ margin: 0; font: 14px/1.45 Arial, sans-serif; color: #17202a; background: #f6f7f9; }}
    header {{ padding: 28px 32px; color: white; background: #1f2937; }}
    h1, h2 {{ margin: 0; }}
    h1 {{ font-size: 28px; }}
    h2 {{ margin: 28px 0 12px; font-size: 20px; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 24px 32px 48px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 18px; }}
    .metric {{ border-left: 5px solid #607d8b; padding: 12px 14px; background: white; box-shadow: 0 1px 2px rgba(0,0,0,.08); }}
    .metric span {{ display: block; color: #5f6b7a; }}
    .metric strong {{ font-size: 26px; }}
    .metric.single_good {{ border-color: #2e7d32; }}
    .metric.low_conf {{ border-color: #f9a825; }}
    .metric.zero_objects {{ border-color: #c62828; }}
    .metric.multiple_objects {{ border-color: #6a1b9a; }}
    table {{ width: 100%; border-collapse: collapse; background: white; box-shadow: 0 1px 2px rgba(0,0,0,.08); }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #e4e7eb; text-align: left; vertical-align: middle; }}
    th {{ position: sticky; top: 0; background: #eef1f5; z-index: 1; }}
    .bar {{ display: inline-block; width: 120px; height: 8px; margin-right: 8px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }}
    .bar span {{ display: block; height: 100%; background: #2563eb; }}
    .bar.good span {{ background: #16a34a; }}
    .frames {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 14px; }}
    .frame-card {{ overflow: hidden; background: white; box-shadow: 0 1px 2px rgba(0,0,0,.08); }}
    .frame-card img {{ display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: cover; background: #d1d5db; }}
    .frame-meta {{ display: grid; gap: 3px; padding: 10px; }}
    .frame-meta span {{ color: #526071; }}
    .muted {{ color: #687385; }}
    .links a {{ color: #1d4ed8; margin-right: 14px; }}
  </style>
</head>
<body>
  <header>
    <h1>Отчет анализа контейнеров</h1>
    <p>Всего обработано: {total}. Лучшая пара device/camera: {html.escape(best_camera_text)}.</p>
  </header>
  <main>
    <section class="summary">
      <div class="metric"><span>total_images</span><strong>{total}</strong></div>
      {cards}
    </section>

    <h2>Рейтинг камер</h2>
    <table>
      <thead>
        <tr>
          <th>rank</th><th>device</th><th>camera</th><th>images</th>
          <th>single_good</th><th>good rate</th><th>avg conf</th>
          <th>zero</th><th>multiple</th>
        </tr>
      </thead>
      <tbody>
        {render_camera_table(camera_rows)}
      </tbody>
    </table>

    <h2>Лучшие фреймы</h2>
    <div class="frames">
      {render_best_frames(best_frame_rows, path.parent)}
    </div>

    <h2>Лучшие диапазоны фреймов</h2>
    <table>
      <thead>
        <tr>
          <th>device</th><th>camera</th><th>start_frame</th><th>end_frame</th>
          <th>frames_count</th><th>average_confidence</th><th>max_confidence</th>
        </tr>
      </thead>
      <tbody>
        {render_ranges_table(best_range_rows)}
      </tbody>
    </table>

    <h2>CSV-файлы</h2>
    <p class="links">
      <a href="detections.csv">detections.csv</a>
      <a href="camera_summary.csv">camera_summary.csv</a>
      <a href="frame_summary.csv">frame_summary.csv</a>
      <a href="best_frames.csv">best_frames.csv</a>
      <a href="best_frame_ranges.csv">best_frame_ranges.csv</a>
      <a href="errors.csv">errors.csv</a>
    </p>
  </main>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def write_reports(
    reports_dir: Path, processed: list[ProcessedImage], good_conf: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    write_detections_csv(reports_dir / "detections.csv", processed)
    camera_rows = write_camera_summary_csv(reports_dir / "camera_summary.csv", processed)
    write_frame_summary_csv(reports_dir / "frame_summary.csv", processed)
    best_frame_rows = write_best_frames_csv(reports_dir / "best_frames.csv", processed)
    best_range_rows = write_best_ranges_csv(
        reports_dir / "best_frame_ranges.csv", processed, good_conf
    )
    write_html_report(
        reports_dir / "report.html",
        processed,
        camera_rows,
        best_frame_rows,
        best_range_rows,
    )
    return camera_rows, best_frame_rows, best_range_rows


def validate_discovery(
    images: list[ImageInfo], devices: set[str], cameras: set[tuple[str, str]]
) -> None:
    if not devices:
        raise RuntimeError("Не найдены папки device_*")
    if not cameras:
        raise RuntimeError("Не найдены пары device_* / cam_* с изображениями")
    if not images:
        raise RuntimeError(
            "Не найдены изображения с расширениями "
            f"{', '.join(sorted(IMAGE_EXTENSIONS))} внутри device_* / cam_*"
        )


def print_summary(
    processed: list[ProcessedImage],
    errors: list[dict[str, str]],
    camera_rows: list[dict[str, Any]],
    best_frame_rows: list[dict[str, Any]],
    reports_dir: Path,
) -> None:
    group_counter = Counter(item.result_group for item in processed)
    print("\nИтоговая статистика")
    print(f"Всего успешно обработано изображений: {len(processed)}")
    print(f"Ошибок обработки изображений: {len(errors)}")
    for group in RESULT_GROUPS:
        print(f"{group}: {group_counter[group]}")

    if camera_rows:
        best_camera = camera_rows[0]
        print(
            "Лучшая камера: "
            f"{best_camera['camera_id']} "
            f"(single_good={best_camera['single_good_count']}, "
            f"good_detection_rate={best_camera['good_detection_rate']})"
        )
        print(
            "Лучшая пара device + camera: "
            f"{best_camera['device_id']} + {best_camera['camera_id']}"
        )
    else:
        print("Лучшая камера: нет данных")
        print("Лучшая пара device + camera: нет данных")

    top_frames = [row for row in best_frame_rows if row["selection_type"] == "top_confidence"][:10]
    print("Лучшие фреймы:")
    if not top_frames:
        print("  нет кадров с ровно одной детекцией")
    for row in top_frames:
        print(
            "  "
            f"{row['device_id']} / {row['camera_id']} / frame={row['frame_number']} "
            f"/ {row['class_name']} / conf={row['confidence']}"
        )

    print(f"Путь к отчетам: {reports_dir}")


def main() -> int:
    args = parse_args()
    try:
        validate_common_args(args)
        reports_dir = ensure_result_directories(args.out)

        print("Поиск изображений...")
        images, devices, cameras = discover_images(args.src, args.frame_regex)
        validate_discovery(images, devices, cameras)
        print(f"Найдено device_*: {len(devices)}")
        print(f"Найдено пар device_* / cam_* с изображениями: {len(cameras)}")
        print(f"Найдено изображений: {len(images)}")

        print("Загрузка YOLO-модели...")
        model = YOLO(str(args.weights))
        model_names = normalize_model_names(getattr(model, "names", None))
        classes = parse_classes(args.classes, model_names)

        processed, errors = process_images(
            model=model,
            images=images,
            out_dir=args.out,
            min_conf=args.min_conf,
            good_conf=args.good_conf,
            device=args.device,
            classes=classes,
            model_names=model_names,
            reports_dir=reports_dir,
        )

        camera_rows, best_frame_rows, _best_range_rows = write_reports(
            reports_dir=reports_dir,
            processed=processed,
            good_conf=args.good_conf,
        )
        print_summary(
            processed=processed,
            errors=errors,
            camera_rows=camera_rows,
            best_frame_rows=best_frame_rows,
            reports_dir=reports_dir,
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI должен вывести понятную ошибку
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
