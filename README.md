from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
LAYOUTS = ("vertical", "twolines", "oneline", "unknown")


def parse_yolo_rows(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    with label_path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"{label_path}:{line_no}: expected YOLO row: class x y w h")

            try:
                cls = int(float(parts[0]))
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError as exc:
                raise ValueError(f"{label_path}:{line_no}: invalid numeric values") from exc

            if width <= 0 or height <= 0:
                raise ValueError(f"{label_path}:{line_no}: width and height must be positive")

            rows.append((cls, x_center, y_center, width, height))

    return rows


def classify_by_ratio(
    width: float,
    height: float,
    *,
    vertical_max_ratio: float,
    twolines_max_ratio: float,
    oneline_min_ratio: float,
) -> str:
    ratio = width / height
    if ratio <= vertical_max_ratio:
        return "vertical"
    if ratio <= twolines_max_ratio:
        return "twolines"
    if ratio >= oneline_min_ratio:
        return "oneline"
    return "unknown"


def find_image_for_label(label_path: Path, labels_dir: Path, images_dir: Path) -> Path | None:
    rel = label_path.relative_to(labels_dir).with_suffix("")
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / rel.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def copy_or_move(src: Path, dst: Path, *, move: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def iter_label_files(labels_dir: Path, *, recursive: bool) -> list[Path]:
    pattern = "**/*.txt" if recursive else "*.txt"
    out: list[Path] = []
    for path in labels_dir.glob(pattern):
        if not path.is_file():
            continue
        rel_parts = set(path.relative_to(labels_dir).parts)
        if rel_parts & set(LAYOUTS):
            continue
        out.append(path)
    return sorted(out)


def sort_dataset(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset).resolve() if args.dataset else None
    images_dir = Path(args.images).resolve() if args.images else (dataset_dir / "images" if dataset_dir else None)
    labels_dir = Path(args.labels).resolve() if args.labels else (dataset_dir / "labels" if dataset_dir else None)

    if images_dir is None or labels_dir is None:
        raise SystemExit("Use --dataset or provide both --images and --labels")
    if not images_dir.exists():
        raise SystemExit(f"Images folder not found: {images_dir}")
    if not labels_dir.exists():
        raise SystemExit(f"Labels folder not found: {labels_dir}")

    apply_changes = bool(args.apply)
    move = args.mode == "move"
    stats = {
        "total": 0,
        "vertical": 0,
        "twolines": 0,
        "oneline": 0,
        "unknown": 0,
        "missing_image": 0,
        "invalid_label": 0,
        "multiple_boxes": 0,
    }

    for label_path in iter_label_files(labels_dir, recursive=args.recursive):
        stats["total"] += 1
        image_path = find_image_for_label(label_path, labels_dir, images_dir)
        layout = "unknown"

        if image_path is None:
            stats["missing_image"] += 1
        else:
            try:
                rows = parse_yolo_rows(label_path)
                if len(rows) != 1:
                    stats["multiple_boxes"] += 1
                if len(rows) == 1:
                    _cls, _x, _y, width, height = rows[0]
                    layout = classify_by_ratio(
                        width,
                        height,
                        vertical_max_ratio=args.vertical_max_ratio,
                        twolines_max_ratio=args.twolines_max_ratio,
                        oneline_min_ratio=args.oneline_min_ratio,
                    )
            except Exception as exc:
                stats["invalid_label"] += 1
                if args.verbose:
                    print(f"invalid_label: {label_path} ({exc})")

        stats[layout] += 1
        if args.skip_unknown and layout == "unknown":
            continue

        label_rel = label_path.relative_to(labels_dir)
        dst_label = labels_dir / layout / label_rel

        dst_image = None
        if image_path is not None:
            image_rel = image_path.relative_to(images_dir)
            dst_image = images_dir / layout / image_rel

        if args.verbose or not apply_changes:
            action = args.mode if apply_changes else "dry-run"
            image_text = str(image_path) if image_path else "MISSING_IMAGE"
            print(f"{action}: {layout}: {image_text} -> {dst_image}; {label_path} -> {dst_label}")

        if apply_changes:
            if image_path is not None and dst_image is not None:
                copy_or_move(image_path, dst_image, move=move)
            copy_or_move(label_path, dst_label, move=move)

    print(
        "done:",
        f"mode={args.mode}",
        f"applied={apply_changes}",
        f"total={stats['total']}",
        f"vertical={stats['vertical']}",
        f"twolines={stats['twolines']}",
        f"oneline={stats['oneline']}",
        f"unknown={stats['unknown']}",
        f"missing_image={stats['missing_image']}",
        f"invalid_label={stats['invalid_label']}",
        f"multiple_boxes={stats['multiple_boxes']}",
    )
    if not apply_changes:
        print("No files were changed. Add --apply to copy or move files.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sort YOLO container images/labels into vertical, twolines, oneline folders by bbox aspect ratio."
    )
    parser.add_argument("-d", "--dataset", help="Dataset folder containing images/ and labels/.")
    parser.add_argument("-i", "--images", help="Folder with images. Use instead of --dataset.")
    parser.add_argument("-l", "--labels", help="Folder with YOLO .txt labels. Use instead of --dataset.")
    parser.add_argument("--mode", choices=["copy", "move"], default="copy", help="Copy or move files.")
    parser.add_argument("--apply", action="store_true", help="Actually copy or move files. Without it, only prints a dry run.")
    parser.add_argument("--recursive", action="store_true", help="Scan labels folder recursively.")
    parser.add_argument("--skip-unknown", action="store_true", help="Do not save unknown files.")
    parser.add_argument("--vertical-max-ratio", type=float, default=0.22, help="w/h <= value means vertical.")
    parser.add_argument("--twolines-max-ratio", type=float, default=3.0, help="w/h <= value means twolines after vertical check.")
    parser.add_argument("--oneline-min-ratio", type=float, default=3.0, help="w/h >= value means oneline.")
    parser.add_argument("--verbose", action="store_true", help="Print each processed file.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return sort_dataset(args)


if __name__ == "__main__":
    raise SystemExit(main())
