from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.container_layout import classify_container_layout  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LAYOUT_TYPES = {"oneline", "twolines", "vertical"}


def _iter_images(folder: Path, *, recursive: bool = False) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for path in folder.glob(pattern):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _copy_or_move(src: Path, dst: Path, *, move: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def _relative_output_path(path: Path, base: Path, output: Path, layout: str) -> Path:
    try:
        rel = path.relative_to(base)
    except ValueError:
        rel = Path(path.name)
    return output / layout / rel


def split_by_layout(args: argparse.Namespace) -> int:
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    config = _load_config(Path(args.config).resolve() if args.config else None)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    stats = {"oneline": 0, "twolines": 0, "vertical": 0, "unknown": 0, "failed": 0}

    for image_path in _iter_images(input_dir, recursive=args.recursive):
        img = cv2.imread(str(image_path))
        if img is None:
            stats["failed"] += 1
            continue

        layout = classify_container_layout(img, config=config)
        if layout not in LAYOUT_TYPES:
            layout = "unknown"

        if layout == "unknown" and args.skip_unknown:
            stats["unknown"] += 1
            continue

        dst = _relative_output_path(image_path, input_dir, output_dir, layout)
        _copy_or_move(image_path, dst, move=args.move)

        label_path = image_path.with_suffix(".txt")
        if label_path.exists():
            _copy_or_move(label_path, dst.with_suffix(".txt"), move=args.move)

        stats[layout] += 1

    print(
        "split_by_layout:",
        f"oneline={stats['oneline']}",
        f"twolines={stats['twolines']}",
        f"vertical={stats['vertical']}",
        f"unknown={stats['unknown']}",
        f"failed={stats['failed']}",
    )
    return 0


def _label_path_for_image(image_path: Path, image_root: Path, labels_root: Path) -> Path:
    try:
        rel = image_path.relative_to(image_root)
    except ValueError:
        rel = Path(image_path.name)
    return (labels_root / rel).with_suffix(".txt")


def _bbox_count(label_path: Path) -> int | None:
    if not label_path.exists():
        return None
    with label_path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def check_label_count(args: argparse.Namespace) -> int:
    images_dir = Path(args.images).resolve()
    labels_dir = Path(args.labels).resolve() if args.labels else images_dir
    output_dir = Path(args.output).resolve()
    expected = int(args.expected)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    checked = 0
    wrong = 0
    missing = 0

    for image_path in _iter_images(images_dir, recursive=args.recursive):
        checked += 1
        label_path = _label_path_for_image(image_path, images_dir, labels_dir)
        count = _bbox_count(label_path)

        if count == expected:
            continue

        wrong += 1
        if count is None:
            missing += 1

        try:
            rel = image_path.relative_to(images_dir)
        except ValueError:
            rel = Path(image_path.name)

        dst_image = output_dir / rel
        _copy_or_move(image_path, dst_image, move=args.move)
        if label_path.exists():
            _copy_or_move(label_path, dst_image.with_suffix(".txt"), move=args.move)

    print(
        "check_label_count:",
        f"expected={expected}",
        f"checked={checked}",
        f"wrong={wrong}",
        f"missing_labels={missing}",
        f"wrong_dir={output_dir}",
    )
    return 0


def split_layout_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    recursive: bool = False,
    skip_unknown: bool = False,
    move: bool = False,
) -> int:
    """Notebook-friendly wrapper for splitting crops by layout."""
    return split_by_layout(
        argparse.Namespace(
            input=str(input_dir),
            output=str(output_dir),
            config=str(config_path) if config_path else None,
            recursive=recursive,
            skip_unknown=skip_unknown,
            move=move,
        )
    )


def check_yolo_box_count(
    images_dir: str | Path,
    expected: int,
    *,
    labels_dir: str | Path | None = None,
    output_dir: str | Path = "wrong",
    recursive: bool = False,
    move: bool = False,
) -> int:
    """Notebook-friendly wrapper for copying wrong YOLO labels to a folder."""
    return check_label_count(
        argparse.Namespace(
            images=str(images_dir),
            labels=str(labels_dir) if labels_dir else None,
            expected=int(expected),
            output=str(output_dir),
            recursive=recursive,
            move=move,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Container crop dataset helper scripts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    split = subparsers.add_parser("split-layout", help="Split crop images into oneline/twolines/vertical folders.")
    split.add_argument("-i", "--input", required=True, help="Folder with crop images.")
    split.add_argument("-o", "--output", required=True, help="Output folder for split images.")
    split.add_argument("--config", help="Optional JSON config with layout thresholds.")
    split.add_argument("--recursive", action="store_true", help="Scan input folder recursively.")
    split.add_argument("--skip-unknown", action="store_true", help="Do not save unknown layout images.")
    split.add_argument("--move", action="store_true", help="Move files instead of copying them.")
    split.set_defaults(func=split_by_layout)

    check = subparsers.add_parser("check-label-count", help="Copy images with wrong YOLO bbox count to wrong folder.")
    check.add_argument("-i", "--images", required=True, help="Folder with images.")
    check.add_argument("-l", "--labels", help="Folder with YOLO .txt labels. Defaults to images folder.")
    check.add_argument("-e", "--expected", required=True, type=int, help="Expected number of bbox rows in label.")
    check.add_argument("-o", "--output", default="wrong", help="Folder for wrong images and labels.")
    check.add_argument("--recursive", action="store_true", help="Scan images folder recursively.")
    check.add_argument("--move", action="store_true", help="Move files instead of copying them.")
    check.set_defaults(func=check_label_count)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
