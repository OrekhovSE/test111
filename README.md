from ultralytics import YOLO
from pathlib import Path
import argparse
import csv
import hashlib
import shutil
import sys


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Сортировка изображений по результатам YOLO: "
            "0 объектов, низкая уверенность, хороший результат, несколько объектов."
        )
    )

    parser.add_argument(
        "--weights",
        required=True,
        help="Путь к весам YOLO, например /home/user/models/best.pt",
    )

    parser.add_argument(
        "--src",
        required=True,
        help="Корневая папка с изображениями и вложенными папками",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Папка для сохранения результата",
    )

    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.25,
        help=(
            "Минимальная уверенность, начиная с которой YOLO учитывает объект. "
            "По умолчанию: 0.25"
        ),
    )

    parser.add_argument(
        "--good-conf",
        type=float,
        default=0.50,
        help=(
            "Минимальная уверенность для хорошего изображения. "
            "По умолчанию: 0.50"
        ),
    )

    parser.add_argument(
        "--classes",
        default=None,
        help=(
            "Идентификаторы учитываемых классов через запятую, например 0 или 0,1,2. "
            "Если не указано, учитываются все классы."
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Устройство для YOLO. Например: 0 для первой видеокарты или cpu. "
            "Если не указано, Ultralytics выберет устройство автоматически."
        ),
    )

    parser.add_argument(
        "--no-save-zero",
        action="store_true",
        help="Не сохранять изображения, где найдено 0 объектов",
    )

    parser.add_argument(
        "--no-save-low-conf",
        action="store_true",
        help="Не сохранять изображения с 1 объектом и низкой уверенностью",
    )

    parser.add_argument(
        "--no-save-good",
        action="store_true",
        help="Не сохранять изображения с 1 объектом и хорошей уверенностью",
    )

    parser.add_argument(
        "--no-save-multiple",
        action="store_true",
        help="Не сохранять изображения, где найдено 2 и более объектов",
    )

    return parser.parse_args()


def validate_args(args):
    weights = Path(args.weights).expanduser().resolve()
    src_root = Path(args.src).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()

    if not weights.is_file():
        print(f"Ошибка: файл весов не найден: {weights}")
        sys.exit(1)

    if not src_root.is_dir():
        print(f"Ошибка: исходная папка не найдена: {src_root}")
        sys.exit(1)

    if not 0 <= args.min_conf <= 1:
        print("Ошибка: --min-conf должен находиться в диапазоне от 0 до 1")
        sys.exit(1)

    if not 0 <= args.good_conf <= 1:
        print("Ошибка: --good-conf должен находиться в диапазоне от 0 до 1")
        sys.exit(1)

    if args.good_conf <= args.min_conf:
        print("Ошибка: --good-conf должен быть больше --min-conf")
        sys.exit(1)

    return weights, src_root, out_root


def parse_target_classes(classes_argument):
    if classes_argument is None:
        return None

    try:
        return {
            int(value.strip())
            for value in classes_argument.split(",")
            if value.strip()
        }
    except ValueError:
        print("Ошибка: --classes должен содержать номера классов, например 0 или 0,1,2")
        sys.exit(1)


def make_unique_name(image_path: Path, source_root: Path) -> str:
    """
    Формирует уникальное имя на основе полного относительного пути.

    Например:
    source/folder1/subfolder/image.jpg

    станет примерно:
    folder1__subfolder__image__a81d390e.jpg
    """
    relative_path = image_path.relative_to(source_root)

    safe_parts = [
        part.replace(" ", "_")
        for part in relative_path.with_suffix("").parts
    ]

    safe_name = "__".join(safe_parts)

    path_hash = hashlib.md5(
        str(relative_path).encode("utf-8")
    ).hexdigest()[:8]

    return f"{safe_name}__{path_hash}{image_path.suffix.lower()}"


def get_output_directories(out_root: Path, category: str):
    """
    Возвращает папки images и labels для соответствующего случая.
    """

    if category == "good":
        images_dir = out_root / "good_images"
        labels_dir = out_root / "good_labels"

    else:
        images_dir = out_root / "bad_images" / category
        labels_dir = out_root / "bad_labels" / category

    return images_dir, labels_dir


def save_detection_result(
    image_path: Path,
    source_root: Path,
    out_root: Path,
    category: str,
    detections: list,
):
    images_dir, labels_dir = get_output_directories(
        out_root=out_root,
        category=category,
    )

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    new_image_name = make_unique_name(
        image_path=image_path,
        source_root=source_root,
    )

    new_label_name = Path(new_image_name).with_suffix(".txt").name

    destination_image = images_dir / new_image_name
    destination_label = labels_dir / new_label_name

    shutil.copy2(image_path, destination_image)

    with destination_label.open("w", encoding="utf-8") as label_file:
        for detection in detections:
            label_file.write(detection["label_line"] + "\n")

    return destination_image, destination_label


def main():
    args = parse_args()

    weights, src_root, out_root = validate_args(args)
    target_classes = parse_target_classes(args.classes)

    save_settings = {
        "zero_objects": not args.no_save_zero,
        "low_conf": not args.no_save_low_conf,
        "good": not args.no_save_good,
        "multiple_objects": not args.no_save_multiple,
    }

    print("Настройки сохранения:")
    print(f"  zero_objects:      {save_settings['zero_objects']}")
    print(f"  low_conf:          {save_settings['low_conf']}")
    print(f"  good:              {save_settings['good']}")
    print(f"  multiple_objects:  {save_settings['multiple_objects']}")
    print()

    print(f"Веса: {weights}")
    print(f"Исходная папка: {src_root}")
    print(f"Результат: {out_root}")
    print(f"Минимальная уверенность: {args.min_conf}")
    print(f"Порог хорошего результата: {args.good_conf}")

    if target_classes is None:
        print("Классы: все")
    else:
        print(f"Классы: {sorted(target_classes)}")

    print()

    out_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))

    images = sorted(
        path
        for path in src_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and out_root not in path.parents
    )

    print(f"Найдено изображений: {len(images)}")

    report_path = out_root / "report.csv"

    counters = {
        "zero_objects": 0,
        "low_conf": 0,
        "good": 0,
        "multiple_objects": 0,
        "saved": 0,
        "not_saved": 0,
        "errors": 0,
    }

    predict_arguments = {
        "conf": args.min_conf,
        "verbose": False,
    }

    if args.device is not None:
        predict_arguments["device"] = args.device

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as report_file:

        writer = csv.writer(report_file)

        writer.writerow([
            "source_image",
            "category",
            "reason",
            "objects_count",
            "classes",
            "confidences",
            "saved",
            "saved_image",
            "saved_label",
        ])

        for index, image_path in enumerate(images, start=1):
            print(
                f"\rОбработка: {index}/{len(images)}",
                end="",
                flush=True,
            )

            try:
                results = model.predict(
                    source=str(image_path),
                    **predict_arguments,
                )

                result = results[0]
                detections = []

                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        confidence = float(box.conf.item())

                        if (
                            target_classes is not None
                            and class_id not in target_classes
                        ):
                            continue

                        x_center, y_center, width, height = (
                            box.xywhn[0].tolist()
                        )

                        detections.append({
                            "class_id": class_id,
                            "confidence": confidence,
                            "label_line": (
                                f"{class_id} "
                                f"{x_center:.6f} "
                                f"{y_center:.6f} "
                                f"{width:.6f} "
                                f"{height:.6f}"
                            ),
                        })

                object_count = len(detections)

                if object_count == 0:
                    category = "zero_objects"
                    reason = "Не найдено объектов с заданной минимальной уверенностью"

                elif object_count == 1:
                    confidence = detections[0]["confidence"]

                    if confidence < args.good_conf:
                        category = "low_conf"
                        reason = (
                            f"Найден 1 объект с низкой уверенностью: "
                            f"{confidence:.4f}"
                        )
                    else:
                        category = "good"
                        reason = (
                            f"Найден 1 объект с хорошей уверенностью: "
                            f"{confidence:.4f}"
                        )

                else:
                    category = "multiple_objects"
                    reason = f"Найдено объектов: {object_count}"

                counters[category] += 1

                saved_image = ""
                saved_label = ""
                was_saved = False

                if save_settings[category]:
                    saved_image_path, saved_label_path = save_detection_result(
                        image_path=image_path,
                        source_root=src_root,
                        out_root=out_root,
                        category=category,
                        detections=detections,
                    )

                    saved_image = str(saved_image_path)
                    saved_label = str(saved_label_path)
                    was_saved = True
                    counters["saved"] += 1

                else:
                    counters["not_saved"] += 1

                classes_text = ",".join(
                    str(detection["class_id"])
                    for detection in detections
                )

                confidences_text = ",".join(
                    f"{detection['confidence']:.4f}"
                    for detection in detections
                )

                writer.writerow([
                    str(image_path),
                    category,
                    reason,
                    object_count,
                    classes_text,
                    confidences_text,
                    "yes" if was_saved else "no",
                    saved_image,
                    saved_label,
                ])

            except Exception as error:
                counters["errors"] += 1

                writer.writerow([
                    str(image_path),
                    "error",
                    str(error),
                    "",
                    "",
                    "",
                    "no",
                    "",
                    "",
                ])

                print()
                print(f"Ошибка при обработке: {image_path}")
                print(error)

    print()
    print()
    print("Обработка завершена")
    print(f"Всего изображений: {len(images)}")
    print(f"0 объектов: {counters['zero_objects']}")
    print(f"1 объект с низкой уверенностью: {counters['low_conf']}")
    print(f"1 объект с хорошей уверенностью: {counters['good']}")
    print(f"2 и более объектов: {counters['multiple_objects']}")
    print(f"Сохранено файлов: {counters['saved']}")
    print(f"Не сохранено из-за отключенных случаев: {counters['not_saved']}")
    print(f"Ошибок: {counters['errors']}")
    print(f"Отчет: {report_path}")


if __name__ == "__main__":
    main()
