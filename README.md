python3 collect_yolo_dataset.py \
  --weights /home/stepan/yolo/best.pt \
  --src /home/stepan/photos \
  --out /home/stepan/result_dataset \
  --conf 0.25


  from ultralytics import YOLO
from pathlib import Path
import shutil
import hashlib
import csv
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Сбор images и labels по предсказаниям YOLO"
    )

    parser.add_argument(
        "--weights",
        required=True,
        help="Путь к весам YOLO, например /home/user/best.pt"
    )

    parser.add_argument(
        "--src",
        required=True,
        help="Корневая папка с фотографиями"
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Папка для результата"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Минимальная уверенность, по умолчанию 0.25"
    )

    parser.add_argument(
        "--classes",
        default=None,
        help="Классы через запятую, например 0 или 0,1,2. Если не указано — берутся все классы"
    )

    return parser.parse_args()


args = parse_args()

WEIGHTS = Path(args.weights)
SRC_ROOT = Path(args.src)
OUT_ROOT = Path(args.out)
CONF_THRES = args.conf

TARGET_CLASSES = (
    [int(x.strip()) for x in args.classes.split(",")]
    if args.classes
    else None
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

OUT_IMAGES = OUT_ROOT / "images"
OUT_LABELS = OUT_ROOT / "labels"

OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_LABELS.mkdir(parents=True, exist_ok=True)

model = YOLO(str(WEIGHTS))


def make_unique_name(img_path: Path, root: Path) -> str:
    rel = img_path.relative_to(root)
    safe_name = "__".join(rel.with_suffix("").parts)
    h = hashlib.md5(str(rel).encode("utf-8")).hexdigest()[:8]
    return f"{safe_name}__{h}{img_path.suffix.lower()}"


images = [
    p for p in SRC_ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
]

print(f"Найдено изображений: {len(images)}")

kept = 0
skipped = 0

report_path = OUT_ROOT / "report.csv"

with open(report_path, "w", newline="", encoding="utf-8") as report_file:
    writer = csv.writer(report_file)
    writer.writerow([
        "source_image",
        "saved_image",
        "objects_count",
        "classes",
        "max_conf"
    ])

    for img_path in images:
        try:
            results = model.predict(
                source=str(img_path),
                conf=CONF_THRES,
                verbose=False
            )

            result = results[0]

            label_lines = []
            found_classes = []
            confidences = []

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())

                    if TARGET_CLASSES is not None and cls_id not in TARGET_CLASSES:
                        continue

                    x_center, y_center, width, height = box.xywhn[0].tolist()

                    label_lines.append(
                        f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    )

                    found_classes.append(cls_id)
                    confidences.append(conf)

            if not label_lines:
                skipped += 1
                continue

            new_image_name = make_unique_name(img_path, SRC_ROOT)
            new_label_name = Path(new_image_name).with_suffix(".txt").name

            dst_image = OUT_IMAGES / new_image_name
            dst_label = OUT_LABELS / new_label_name

            shutil.copy2(img_path, dst_image)

            with open(dst_label, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))

            kept += 1

            writer.writerow([
                str(img_path),
                str(dst_image),
                len(label_lines),
                ",".join(map(str, sorted(set(found_classes)))),
                round(max(confidences), 4)
            ])

        except Exception as e:
            print(f"Ошибка с файлом: {img_path}")
            print(e)
            skipped += 1

print("Готово")
print(f"Сохранено изображений: {kept}")
print(f"Пропущено изображений: {skipped}")
print(f"Результат: {OUT_ROOT}")
print(f"Отчет: {report_path}")
