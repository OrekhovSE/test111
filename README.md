from ultralytics import YOLO
from pathlib import Path
import shutil
import hashlib
import csv
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Поиск плохих изображений по результатам YOLO"
    )

    parser.add_argument("--weights", required=True, help="Путь к весам YOLO")
    parser.add_argument("--src", required=True, help="Корневая папка с изображениями")
    parser.add_argument("--out", required=True, help="Папка для сохранения результата")

    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.25,
        help="Минимальная уверенность для учета детекции"
    )

    parser.add_argument(
        "--low-conf",
        type=float,
        default=0.50,
        help="Порог низкой уверенности"
    )

    parser.add_argument(
        "--classes",
        default=None,
        help="Классы через запятую, например 0 или 0,1,2. Если не указано - берутся все"
    )

    return parser.parse_args()


def make_unique_name(img_path: Path, root: Path) -> str:
    rel = img_path.relative_to(root)
    safe_name = "__".join(rel.with_suffix("").parts)
    h = hashlib.md5(str(rel).encode("utf-8")).hexdigest()[:8]
    return f"{safe_name}__{h}{img_path.suffix.lower()}"


def main():
    args = parse_args()

    weights = Path(args.weights)
    src_root = Path(args.src)
    out_root = Path(args.out)

    min_conf = args.min_conf
    low_conf = args.low_conf

    target_classes = (
        [int(x.strip()) for x in args.classes.split(",")]
        if args.classes
        else None
    )

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    bad_images_root = out_root / "bad_images"
    bad_labels_root = out_root / "bad_labels"

    reason_folders = [
        "zero_objects",
        "low_conf",
        "multiple_objects"
    ]

    for folder in reason_folders:
        (bad_images_root / folder).mkdir(parents=True, exist_ok=True)
        (bad_labels_root / folder).mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))

    images = [
        p for p in src_root.rglob("*")
        if p.is_file() and p.suffix.lower() in image_exts
    ]

    print(f"Найдено изображений: {len(images)}")

    report_path = out_root / "bad_report.csv"

    total_ok = 0
    total_bad = 0
    total_zero = 0
    total_low_conf = 0
    total_multiple = 0
    total_errors = 0

    with open(report_path, "w", newline="", encoding="utf-8") as report_file:
        writer = csv.writer(report_file)
        writer.writerow([
            "source_image",
            "saved_image",
            "case_folder",
            "reason",
            "objects_count",
            "classes",
            "confidences"
        ])

        for img_path in images:
            try:
                results = model.predict(
                    source=str(img_path),
                    conf=min_conf,
                    verbose=False
                )

                result = results[0]
                detections = []

                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls.item())
                        conf = float(box.conf.item())

                        if target_classes is not None and cls_id not in target_classes:
                            continue

                        x_center, y_center, width, height = box.xywhn[0].tolist()

                        detections.append({
                            "cls_id": cls_id,
                            "conf": conf,
                            "label_line": f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        })

                det_count = len(detections)

                is_bad = False
                case_folder = None
                reason = ""

                if det_count == 0:
                    is_bad = True
                    case_folder = "zero_objects"
                    reason = "objects_count_0"
                    total_zero += 1

                elif det_count > 1:
                    is_bad = True
                    case_folder = "multiple_objects"
                    reason = f"objects_count_{det_count}"
                    total_multiple += 1

                else:
                    only_conf = detections[0]["conf"]

                    if only_conf < low_conf:
                        is_bad = True
                        case_folder = "low_conf"
                        reason = f"low_conf_{only_conf:.4f}"
                        total_low_conf += 1

                if not is_bad:
                    total_ok += 1
                    continue

                new_image_name = make_unique_name(img_path, src_root)
                new_label_name = Path(new_image_name).with_suffix(".txt").name

                dst_image = bad_images_root / case_folder / new_image_name
                dst_label = bad_labels_root / case_folder / new_label_name

                shutil.copy2(img_path, dst_image)

                with open(dst_label, "w", encoding="utf-8") as f:
                    for det in detections:
                        f.write(det["label_line"] + "\n")

                writer.writerow([
                    str(img_path),
                    str(dst_image),
                    case_folder,
                    reason,
                    det_count,
                    ",".join(str(det["cls_id"]) for det in detections),
                    ",".join(f"{det['conf']:.4f}" for det in detections)
                ])

                total_bad += 1

            except Exception as e:
                print(f"Ошибка с файлом: {img_path}")
                print(e)
                total_errors += 1

    print("Готово")
    print(f"Хороших изображений: {total_ok}")
    print(f"Плохих изображений всего: {total_bad}")
    print(f"  zero_objects: {total_zero}")
    print(f"  low_conf: {total_low_conf}")
    print(f"  multiple_objects: {total_multiple}")
    print(f"Ошибок обработки: {total_errors}")
    print(f"Результат: {out_root}")
    print(f"Отчет: {report_path}")


if __name__ == "__main__":
    main()
