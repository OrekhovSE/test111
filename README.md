from pathlib import Path
import random
import shutil
import yaml

# =========================
# НАСТРОЙКИ
# =========================

SOURCE_DIR = Path("master_dataset")
OUTPUT_DIR = Path("yolo_dataset")

# Доля val для каждой подпапки
# Например: 0.2 = 20% val, 80% train
VAL_RATIO_BY_GROUP = {
    "vertical": 0.2,
    "oneline": 0.2,
    "twolines": 0.2,
}

# Если появится папка, которой нет выше, будет использовано это значение
DEFAULT_VAL_RATIO = 0.2

SEED = 42

# Если True — копирует файлы
# Если False — создает symlink, чтобы не дублировать датасет
COPY_FILES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Названия классов YOLO
# Замени на свои классы
CLASS_NAMES = [
    "number"
]


# =========================
# ЛОГИКА
# =========================

def copy_or_link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if COPY_FILES:
        shutil.copy2(src, dst)
    else:
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())


def get_label_path(image_path: Path, images_root: Path, labels_root: Path) -> Path:
    relative_path = image_path.relative_to(images_root)
    return labels_root / relative_path.with_suffix(".txt")


def split_group(group_name: str, images_root: Path, labels_root: Path):
    group_images_dir = images_root / group_name

    images = [
        p for p in group_images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        print(f"[SKIP] {group_name}: изображений не найдено")
        return

    random.shuffle(images)

    val_ratio = VAL_RATIO_BY_GROUP.get(group_name, DEFAULT_VAL_RATIO)
    val_count = round(len(images) * val_ratio)

    if len(images) > 1 and val_ratio > 0:
        val_count = max(1, val_count)

    val_images = set(images[:val_count])
    train_images = set(images[val_count:])

    print(
        f"[OK] {group_name}: всего={len(images)}, "
        f"train={len(train_images)}, val={len(val_images)}"
    )

    for image_path in images:
        split_name = "val" if image_path in val_images else "train"

        relative_path = image_path.relative_to(images_root)

        dst_image = OUTPUT_DIR / "images" / split_name / relative_path
        dst_label = OUTPUT_DIR / "labels" / split_name / relative_path.with_suffix(".txt")

        src_label = get_label_path(image_path, images_root, labels_root)

        copy_or_link(image_path, dst_image)

        if src_label.exists():
            copy_or_link(src_label, dst_label)
        else:
            # Если у изображения нет разметки, создаем пустой label.
            # Для YOLO это означает изображение без объектов.
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            dst_label.write_text("", encoding="utf-8")
            print(f"[WARN] Нет label для: {image_path}")


def create_data_yaml():
    data = {
        "path": str(OUTPUT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": CLASS_NAMES,
    }

    yaml_path = OUTPUT_DIR / "data.yaml"
    yaml_path.write_text(
        yaml.dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8"
    )

    print(f"[OK] Создан файл: {yaml_path}")


def main():
    random.seed(SEED)

    images_root = SOURCE_DIR / "images"
    labels_root = SOURCE_DIR / "labels"

    if not images_root.exists():
        raise FileNotFoundError(f"Не найдена папка: {images_root}")

    if not labels_root.exists():
        raise FileNotFoundError(f"Не найдена папка: {labels_root}")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    groups = [
        p.name for p in images_root.iterdir()
        if p.is_dir()
    ]

    if not groups:
        raise RuntimeError("В папке images нет подпапок с датасетами")

    for group_name in groups:
        split_group(group_name, images_root, labels_root)

    create_data_yaml()

    print("\nГотово. Теперь можно запускать обучение YOLO.")


if __name__ == "__main__":
    main()
