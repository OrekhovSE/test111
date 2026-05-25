from __future__ import annotations

import logging
import os
from itertools import product
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

Box = tuple[int, int, int, int]

STATUS_FULL_CONFIRMED = "FULL_11_CONFIRMED"
STATUS_FULL_GUESSED = "FULL_11_GUESSED_CHECK"
STATUS_PARTIAL_10 = "PARTIAL_10"
STATUS_RAW_ONLY = "RAW_ONLY"
STATUS_NOT_FOUND = "NOT_FOUND"
LETTER_CONFUSION_GROUPS = (set("TIFL"),)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
logger = logging.getLogger("app")

VERTICAL_CFG_DEFAULTS: dict[str, Any] = {
    "vertical_yolo_model_path": "",
    "vertical_yolo_conf": 0.25,
    "vertical_yolo_iou": 0.45,
    "vertical_yolo_imgsz": 1280,
    "vertical_yolo_max_det": 32,
    "vertical_yolo_device": "",

    "vertical_symbol_min_conf": 0.0,
    "vertical_symbol_pad_x_ratio": 0.18,
    "vertical_symbol_pad_y_ratio": 0.14,
    "vertical_max_symbols": 11,

    "vertical_use_iso_guess_for_check_digit": True,

    "vertical_read_mode": "symbols",  # symbols / strip

    "vertical_char_canvas_size": 128,
    "vertical_char_fg_pad": 3,
    "vertical_char_min_v": 135,
    "vertical_char_max_s": 125,
    "vertical_char_scale": 0.72,
    "vertical_char_bg": "black",

    "vertical_strip_target_h": 96,
    "vertical_strip_gap": 10,
    "vertical_strip_outer_pad": 14,
    "vertical_strip_bg": "black",
    "vertical_strip_scale": 2.10,

    "vertical_debug_save_strip": True,
}

_MODEL: Any | None = None
_MODEL_PATH: str = ""
_CORE: Any | None = None


class _CoreProxy:
    def __getattr__(self, name: str) -> Any:
        global _CORE
        if _CORE is None:
            from . import container_ocr_core
            _CORE = container_ocr_core
        return getattr(_CORE, name)


core = _CoreProxy()


def _cfg(config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    merged = dict(VERTICAL_CFG_DEFAULTS)
    if config:
        for key in merged:
            if key in config:
                merged[key] = config[key]
    return merged


def _empty_result(status: str = STATUS_NOT_FOUND, *, reason: str = "") -> dict[str, Any]:
    return {
        "raw_text": "",
        "base10": "",
        "check_digit": "",
        "full_code": "",
        "score": -1.0,
        "is_valid_iso": False,
        "status": status,
        "check_digit_source": "",
        "guessed_check_digit": False,
        "segmentation_score": 0.0,
        "segmentation_method": "yolov11_boxes",
        "debug_dir": "",
        "failure_reason": reason,
    }


def _candidate_model_paths(config: dict[str, Any]) -> list[Path]:
    paths: list[str] = []
    configured = str(config.get("vertical_yolo_model_path") or "").strip()
    if configured:
        paths.append(configured)

    env_path = os.getenv("VERTICAL_YOLO_MODEL_PATH", "").strip()
    if env_path:
        paths.append(env_path)

    paths.append("models/YOLO/best_vertical.pt")

    out: list[Path] = []
    seen: set[str] = set()
    for item in paths:
        path = Path(item)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _resolve_model_path(config: dict[str, Any]) -> Path:
    for path in _candidate_model_paths(config):
        if path.exists():
            return path
    tried = ", ".join(str(path) for path in _candidate_model_paths(config))
    raise FileNotFoundError(
        "Vertical YOLO model not found. Set vertical_yolo_model_path or "
        f"VERTICAL_YOLO_MODEL_PATH. Tried: {tried}"
    )


def _load_model(config: Optional[dict[str, Any]] = None) -> Any:
    global _MODEL, _MODEL_PATH
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ultralytics is required for vertical YOLO OCR. Install requirements.txt first."
        ) from exc

    cfg = _cfg(config)
    path = str(_resolve_model_path(cfg))
    if _MODEL is None or _MODEL_PATH != path:
        _MODEL = YOLO(path)
        _MODEL_PATH = path
    return _MODEL


def reload_model() -> dict[str, Any]:
    global _MODEL, _MODEL_PATH
    _MODEL = None
    _MODEL_PATH = ""
    return {"reloaded": True}


def _clip_box(box: Box, width: int, height: int) -> Optional[Box]:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _pad_box(
    box: Box,
    width: int,
    height: int,
    *,
    pad_x_ratio: float,
    pad_y_ratio: float,
) -> Optional[Box]:
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = max(1, int(round(bw * pad_x_ratio)))
    pad_y = max(1, int(round(bh * pad_y_ratio)))
    return _clip_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)


def _serialize_box(box: Box) -> list[int]:
    return [int(box[0]), int(box[1]), int(box[2]), int(box[3])]


def _bg_value(name: str) -> int:
    return 255 if str(name).strip().lower() in {"white", "light"} else 0


def _white_symbol_mask(roi: np.ndarray, config: Optional[dict[str, Any]] = None) -> np.ndarray:
    if roi is None or getattr(roi, "size", 0) == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    cfg = _cfg(config)
    bgr = core._ensure_bgr(roi)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    min_v = int(cfg["vertical_char_min_v"])
    max_s = int(cfg["vertical_char_max_s"])

    mask_hsv = cv2.inRange(
        hsv,
        np.array([0, 0, min_v], dtype=np.uint8),
        np.array([180, max_s, 255], dtype=np.uint8),
    )

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _thr, mask_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(mask_otsu) > (mask_otsu.size * 0.65):
        mask_otsu = 255 - mask_otsu

    mask = cv2.bitwise_or(mask_hsv, mask_otsu)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _mask_bbox(mask: np.ndarray, pad: int = 2) -> Optional[Box]:
    if mask is None or getattr(mask, "size", 0) == 0:
        return None

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    h, w = mask.shape[:2]
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(w, int(xs.max()) + pad + 1)
    y2 = min(h, int(ys.max()) + pad + 1)

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _tight_crop_white_symbol(
    roi: np.ndarray,
    config: Optional[dict[str, Any]] = None,
    *,
    check_digit: bool = False,
) -> np.ndarray:
    if roi is None or getattr(roi, "size", 0) == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    cfg = _cfg(config)
    bgr = core._ensure_bgr(roi)

    if check_digit:
        inner_reader = getattr(core, "_safe_inner_crop", None)
        if callable(inner_reader):
            inner = inner_reader(bgr, margin_ratio=0.10)
            if inner is not None and getattr(inner, "size", 0):
                bgr = inner

    mask = _white_symbol_mask(bgr, cfg)
    bbox = _mask_bbox(mask, pad=int(cfg["vertical_char_fg_pad"]))
    if bbox is None:
        return bgr

    x1, y1, x2, y2 = bbox
    cropped = bgr[y1:y2, x1:x2]
    return cropped if cropped is not None and cropped.size else bgr


def _isolate_symbol_on_bg(
    roi: np.ndarray,
    config: Optional[dict[str, Any]] = None,
    *,
    bg_key: str = "vertical_char_bg",
) -> np.ndarray:
    if roi is None or getattr(roi, "size", 0) == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    cfg = _cfg(config)
    bgr = core._ensure_bgr(roi)
    bg = _bg_value(cfg.get(bg_key, "black"))

    mask = _white_symbol_mask(bgr, cfg)
    out = np.full_like(bgr, bg)

    if np.count_nonzero(mask) >= 6:
        out[mask > 0] = bgr[mask > 0]
        return out

    return bgr


def _prepare_char_roi(
    roi: np.ndarray,
    config: Optional[dict[str, Any]] = None,
    *,
    check_digit: bool = False,
) -> np.ndarray:
    cfg = _cfg(config)
    out_size = max(64, int(cfg["vertical_char_canvas_size"]))
    scale_ratio = float(cfg["vertical_char_scale"])
    bg = _bg_value(cfg.get("vertical_char_bg", "black"))

    tight = _tight_crop_white_symbol(roi, cfg, check_digit=check_digit)
    isolated = _isolate_symbol_on_bg(tight, cfg, bg_key="vertical_char_bg")

    h, w = isolated.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((out_size, out_size, 3), bg, dtype=np.uint8)

    scale = (out_size * scale_ratio) / float(max(h, w))
    new_w = max(8, int(round(w * scale)))
    new_h = max(8, int(round(h * scale)))

    resized = cv2.resize(isolated, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((out_size, out_size, 3), bg, dtype=np.uint8)

    ox = (out_size - new_w) // 2
    oy = (out_size - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    return canvas


def _prepare_strip_symbol(
    roi: np.ndarray,
    config: Optional[dict[str, Any]] = None,
    *,
    check_digit: bool = False,
) -> np.ndarray:
    # Temporary strip experiment: read YOLO crops as-is, without symbol mask isolation.
    if roi is None or getattr(roi, "size", 0) == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return core._ensure_bgr(roi)


def _best_char_from_text_pairs(
    text_pairs: Sequence[tuple[str, float]],
    *,
    kind: str,
) -> tuple[str, float, str]:
    best_char = ""
    best_score = -1.0
    best_raw = ""

    for raw_text, raw_score in text_pairs:
        text = core.clean(str(raw_text or ""))
        if not text:
            continue

        if kind == "letter":
            mapped = "".join(core._TO_LETTER.get(ch, ch) for ch in text)
            valid = [(idx, ch) for idx, ch in enumerate(mapped) if ch.isalpha()]
        else:
            mapped = "".join(core._TO_DIGIT.get(ch, ch) for ch in text)
            valid = [(idx, ch) for idx, ch in enumerate(mapped) if ch.isdigit()]

        if not valid:
            continue

        for idx, ch in valid:
            score = float(raw_score)
            if len(valid) == 1:
                score += 0.12
            if len(mapped) <= 2:
                score += 0.05
            if idx == 0 or idx == len(mapped) - 1:
                score += 0.02
            if len(valid) > 1:
                score -= 0.03 * float(len(valid) - 1)

            if score > best_score:
                best_char = ch
                best_score = score
                best_raw = str(raw_text or "")

    return best_char, best_score, best_raw


def _detect_symbol_boxes(img: np.ndarray, config: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    if img is None or getattr(img, "size", 0) == 0:
        return []

    cfg = _cfg(config)
    model = _load_model(cfg)

    kwargs: dict[str, Any] = {
        "conf": float(cfg["vertical_yolo_conf"]),
        "iou": float(cfg["vertical_yolo_iou"]),
        "imgsz": int(cfg["vertical_yolo_imgsz"]),
        "max_det": int(cfg["vertical_yolo_max_det"]),
        "verbose": False,
    }
    device = str(cfg.get("vertical_yolo_device") or "").strip()
    if device:
        kwargs["device"] = device

    predictions = model.predict(img, **kwargs)
    if not predictions:
        return []

    result = predictions[0]
    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None or getattr(boxes_obj, "xyxy", None) is None:
        return []

    xyxy = boxes_obj.xyxy.detach().cpu().numpy()
    confs = (
        boxes_obj.conf.detach().cpu().numpy()
        if getattr(boxes_obj, "conf", None) is not None
        else np.ones((len(xyxy),), dtype=np.float32)
    )

    h, w = img.shape[:2]
    min_conf = float(cfg["vertical_symbol_min_conf"])

    detections: list[dict[str, Any]] = []
    for coords, conf in zip(xyxy, confs):
        score = float(conf)
        if score < min_conf:
            continue

        box = _clip_box(
            (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])),
            w,
            h,
        )
        if box is None:
            continue

        detections.append({"box": box, "score": score})

    detections.sort(key=lambda item: (int(item["box"][1]), int(item["box"][0])))

    max_symbols = int(cfg["vertical_max_symbols"])
    if max_symbols > 0:
        detections = detections[:max_symbols]

    return detections


def _crop_symbol(img: np.ndarray, box: Box, config: Optional[dict[str, Any]] = None) -> Optional[np.ndarray]:
    if img is None or getattr(img, "size", 0) == 0:
        return None

    cfg = _cfg(config)
    h, w = img.shape[:2]
    padded = _pad_box(
        box,
        w,
        h,
        pad_x_ratio=float(cfg["vertical_symbol_pad_x_ratio"]),
        pad_y_ratio=float(cfg["vertical_symbol_pad_y_ratio"]),
    )
    if padded is None:
        return None

    x1, y1, x2, y2 = padded
    roi = img[y1:y2, x1:x2]
    return roi if roi is not None and roi.size else None


def _symbol_rois_from_detections(
    img: np.ndarray,
    detections: Sequence[dict[str, Any]],
    config: Optional[dict[str, Any]] = None,
) -> list[np.ndarray]:
    rois: list[np.ndarray] = []
    max_symbols = int(_cfg(config)["vertical_max_symbols"])
    for det in detections[:max_symbols]:
        roi = _crop_symbol(img, tuple(det["box"]), config)
        if roi is not None and roi.size:
            rois.append(roi)
    return rois


def _compose_horizontal_strip(
    rois: Sequence[np.ndarray],
    config: Optional[dict[str, Any]] = None,
) -> Optional[np.ndarray]:
    if not rois:
        return None

    cfg = _cfg(config)
    target_h = max(24, int(cfg["vertical_strip_target_h"]))
    gap = max(0, int(cfg["vertical_strip_gap"]))
    outer_pad = max(0, int(cfg["vertical_strip_outer_pad"]))
    bg_value = _bg_value(cfg.get("vertical_strip_bg", "black"))

    prepared: list[np.ndarray] = []
    for idx, roi in enumerate(rois):
        if roi is None or getattr(roi, "size", 0) == 0:
            continue

        clean_roi = _prepare_strip_symbol(
            roi,
            cfg,
            check_digit=(idx == len(rois) - 1 and len(rois) >= 11),
        )
        h, w = clean_roi.shape[:2]
        if h <= 0 or w <= 0:
            continue

        scale = target_h / float(h)
        target_w = max(8, int(round(w * scale)))
        resized = cv2.resize(clean_roi, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        prepared.append(resized)

    if not prepared:
        return None

    total_w = outer_pad * 2 + sum(item.shape[1] for item in prepared) + gap * max(0, len(prepared) - 1)
    canvas = np.full((target_h + outer_pad * 2, total_w, 3), bg_value, dtype=np.uint8)

    x = outer_pad
    for item in prepared:
        y = outer_pad
        canvas[y:y + item.shape[0], x:x + item.shape[1]] = item
        x += item.shape[1] + gap

    return canvas


def _clean_letter(text: str) -> str:
    value = core.clean(text)
    value = "".join(core._TO_LETTER.get(ch, ch) for ch in value)
    letters = [ch for ch in value if ch.isalpha()]
    return letters[0] if letters else ""


def _clean_digit(text: str) -> str:
    value = core.clean(text)
    value = "".join(core._TO_DIGIT.get(ch, ch) for ch in value)
    digits = [ch for ch in value if ch.isdigit()]
    return digits[0] if digits else ""


def _vertical_strip_check_digit_repairs(raw: str) -> list[str]:
    text = core.clean(raw)
    if not text:
        return []

    out = [text]

    # Paddle can read the square frame around the check digit as an extra "1".
    # Example: CAIU83550816 should also be tried as CAIU8355086.
    for i in range(max(1, len(text) - 10)):
        part = text[i:i + 12]
        if len(part) != 12:
            continue
        if part[:4].isalpha() and part[4:].isdigit() and part[10] == "1":
            repaired = text[:i] + part[:10] + part[11] + text[i + 12:]
            if repaired not in out:
                out.append(repaired)

    return out


def _iso_correct_owner_confusions(owner: str, base6: str, check_digit: str) -> tuple[str, bool]:
    if len(owner) != 4 or len(base6) != 6 or len(check_digit) != 1:
        return owner, False
    if not base6.isdigit() or not check_digit.isdigit():
        return owner, False
    if core._is_valid_iso6346(f"{owner}{base6}{check_digit}"):
        return owner, False

    choices: list[list[str]] = []
    has_confusable = False
    for ch in owner:
        group = next((grp for grp in LETTER_CONFUSION_GROUPS if ch in grp), None)
        if group is None:
            choices.append([ch])
            continue
        has_confusable = True
        choices.append(sorted(group))

    if not has_confusable:
        return owner, False

    valid: list[str] = []
    for candidate_chars in product(*choices):
        candidate_owner = "".join(candidate_chars)
        if candidate_owner == owner:
            continue
        if core._is_valid_iso6346(f"{candidate_owner}{base6}{check_digit}"):
            valid.append(candidate_owner)

    if len(valid) == 1:
        return valid[0], True
    return owner, False


def _read_letter_roi(roi: np.ndarray, config: Optional[dict[str, Any]] = None) -> tuple[str, float, str]:
    if roi is None or getattr(roi, "size", 0) == 0:
        return "", -1.0, ""

    best_text = ""
    best_score = -1.0
    best_raw = ""

    direct_readers = [
        getattr(core, "_paddle_letter_from_roi_vertical", None),
        getattr(core, "_paddle_letter_from_roi", None),
    ]
    for reader in direct_readers:
        if not callable(reader):
            continue
        raw, score = reader(roi)
        letter = _clean_letter(str(raw or ""))
        if letter and float(score) > best_score:
            best_text = letter
            best_score = float(score)
            best_raw = str(raw or "")

    if best_text:
        return best_text, best_score, best_raw

    prepared = _prepare_char_roi(roi, config, check_digit=False)

    variants = [
        prepared,
        core._preprocess_fast(prepared),
        cv2.resize(prepared, None, fx=1.20, fy=1.20, interpolation=cv2.INTER_CUBIC),
    ]
    bonuses = [0.06, 0.08, 0.04]

    text_pairs: list[tuple[str, float]] = []
    batch_reader = getattr(core, "_run_text_recognition_batch", None)
    if callable(batch_reader):
        recognized = batch_reader(variants)
        for (text, score), bonus in zip(recognized, bonuses):
            if text:
                text_pairs.append((str(text), float(score) + bonus))

    best_text, best_score, best_raw = _best_char_from_text_pairs(text_pairs, kind="letter")
    if best_text:
        return best_text, best_score, best_raw

    fallback_readers = [
        getattr(core, "_paddle_letter_from_roi_vertical", None),
        getattr(core, "_paddle_letter_from_roi", None),
    ]
    for reader in fallback_readers:
        if not callable(reader):
            continue
        raw, score = reader(prepared)
        letter = _clean_letter(str(raw or ""))
        if letter and float(score) > best_score:
            best_text = letter
            best_score = float(score)
            best_raw = str(raw or "")

    return best_text, best_score, best_raw


def _read_digit_roi(
    roi: np.ndarray,
    *,
    check_digit: bool = False,
    config: Optional[dict[str, Any]] = None,
) -> tuple[str, float, str]:
    if roi is None or getattr(roi, "size", 0) == 0:
        return "", -1.0, ""

    best_text = ""
    best_score = -1.0
    best_raw = ""

    direct_readers: list[Any] = []
    if check_digit:
        direct_readers.append(lambda item: core._read_digit_from_roi(item, square_bias=True))
    direct_readers.extend(
        [
            getattr(core, "_paddle_digit_from_roi", None),
            getattr(core, "_read_digit_from_roi", None),
        ]
    )

    for reader in direct_readers:
        if not callable(reader):
            continue
        try:
            raw, score = reader(roi)
        except TypeError:
            raw, score = reader(roi, square_bias=check_digit)

        digit = _clean_digit(str(raw or ""))
        if digit and float(score) > best_score:
            best_text = digit
            best_score = float(score)
            best_raw = str(raw or "")

    if best_text:
        return best_text, best_score, best_raw

    prepared = _prepare_char_roi(roi, config, check_digit=check_digit)

    variants = [
        prepared,
        core._preprocess_fast(prepared),
        cv2.resize(prepared, None, fx=1.20, fy=1.20, interpolation=cv2.INTER_CUBIC),
    ]
    bonuses = [0.06, 0.08, 0.04]

    text_pairs: list[tuple[str, float]] = []
    batch_reader = getattr(core, "_run_text_recognition_batch", None)
    if callable(batch_reader):
        recognized = batch_reader(variants)
        for (text, score), bonus in zip(recognized, bonuses):
            if text:
                text_pairs.append((str(text), float(score) + bonus))

    best_text, best_score, best_raw = _best_char_from_text_pairs(text_pairs, kind="digit")
    if best_text:
        return best_text, best_score, best_raw

    fallback_readers: list[Any] = []
    if check_digit:
        fallback_readers.append(lambda item: core._read_digit_from_roi(item, square_bias=True))
    fallback_readers.extend(
        [
            getattr(core, "_paddle_digit_from_roi", None),
            getattr(core, "_read_digit_from_roi", None),
        ]
    )

    for reader in fallback_readers:
        if not callable(reader):
            continue
        try:
            raw, score = reader(prepared)
        except TypeError:
            raw, score = reader(prepared, square_bias=check_digit)

        digit = _clean_digit(str(raw or ""))
        if digit and float(score) > best_score:
            best_text = digit
            best_score = float(score)
            best_raw = str(raw or "")

    return best_text, best_score, best_raw


def _decode_full_code_text(raw: str, score: float, config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    result = _empty_result(STATUS_RAW_ONLY if raw else STATUS_NOT_FOUND, reason="strip_no_container_candidate")
    result["raw_text"] = core.clean(raw)
    result["score"] = float(score)

    best_code = ""
    best_score = -1.0
    best_valid = False

    original_raw = core.clean(raw)
    for raw_variant in _vertical_strip_check_digit_repairs(raw):
        for normalized in core._normalized_candidates(raw_variant):
            quality = core._container_text_quality(normalized)
            final = 0.75 * float(score) + 0.25 * quality
            is_valid = core._is_valid_iso6346(normalized)
            if is_valid:
                final += 0.10
            if raw_variant != original_raw:
                final += 0.08
            if final > best_score:
                best_code = normalized
                best_score = final
                best_valid = is_valid

    if not best_code:
        for base10 in core._normalized_base_candidates(raw):
            quality = core._container_base_quality(base10)
            final = 0.80 * float(score) + 0.20 * quality
            if final > best_score:
                best_code = base10
                best_score = final
                best_valid = False

    if not best_code:
        return result

    result["base10"] = best_code[:10]
    result["score"] = float(best_score)

    if len(best_code) == 11:
        result["check_digit"] = best_code[10]
        result["full_code"] = best_code
        result["is_valid_iso"] = bool(best_valid)
        result["status"] = STATUS_FULL_CONFIRMED if best_valid else "FULL_11"
        result["check_digit_source"] = "paddle_strip"
        return result

    if bool(_cfg(config)["vertical_use_iso_guess_for_check_digit"]):
        guessed = str(core._iso6346_check_digit(best_code[:10]))
        result["check_digit"] = guessed
        result["full_code"] = f"{best_code[:10]}{guessed}"
        result["is_valid_iso"] = True
        result["status"] = STATUS_FULL_GUESSED
        result["check_digit_source"] = "iso_guess"
        result["guessed_check_digit"] = True
    else:
        result["full_code"] = best_code[:10]
        result["status"] = STATUS_PARTIAL_10

    return result


def _rank_strip_candidate(result: dict[str, Any]) -> float:
    rank = float(result.get("score") or -1.0)
    status = str(result.get("status") or "")

    if result.get("raw_text"):
        rank += 0.5
    if result.get("base10"):
        rank += 3.0
    if result.get("full_code"):
        rank += 4.0

    if status == STATUS_FULL_GUESSED:
        rank += 4.5
    elif status == STATUS_FULL_CONFIRMED:
        rank += 7.0
    elif status == "FULL_11":
        rank += 3.5
    elif status == STATUS_PARTIAL_10:
        rank += 2.0

    if result.get("is_valid_iso"):
        rank += 4.0
    return rank


def _is_good_strip_result(result: dict[str, Any]) -> bool:
    if not result:
        return False
    status = str(result.get("status") or "")
    return bool(result.get("base10")) and status in {
        STATUS_FULL_CONFIRMED,
        STATUS_FULL_GUESSED,
        "FULL_11",
        STATUS_PARTIAL_10,
    }


def _decode_strip_rois(
    rois: Sequence[np.ndarray],
    detections: Sequence[dict[str, Any]],
    config: Optional[dict[str, Any]] = None,
    *,
    strip_order: str,
) -> dict[str, Any]:
    strip = _compose_horizontal_strip(rois[:11], config)
    if strip is None or strip.size == 0:
        result = _empty_result(STATUS_NOT_FOUND, reason="strip_compose_failed")
        result["detected_count"] = len(detections)
        result["strip_order"] = strip_order
        return result

    cfg = _cfg(config)

    if bool(cfg.get("vertical_debug_save_strip", True)):
        debug_dir = PROJECT_ROOT / "crops" / "debug_runtime" / "vertical_strip"
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / f"last_strip_{strip_order}.jpg"), strip)
        cv2.imwrite(str(debug_dir / "last_strip.jpg"), strip)

    scale = float(cfg.get("vertical_strip_scale", 2.10))
    strip_fast = core._preprocess_fast(strip)

    variants = [
        cv2.resize(strip, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC),
        cv2.resize(strip_fast, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC),
        cv2.resize(strip, None, fx=scale * 1.15, fy=scale * 1.15, interpolation=cv2.INTER_CUBIC),
    ]
    bonuses = [0.05, 0.08, 0.03]

    best_pairs: list[tuple[str, float]] = []
    batch_reader = getattr(core, "_run_text_recognition_batch", None)
    if callable(batch_reader):
        recognized = batch_reader(variants)
        for (text, score), bonus in zip(recognized, bonuses):
            if text:
                best_pairs.append((str(text), float(score) + bonus))

    if not best_pairs:
        for profile, bonus in (("minimal", 0.01),):
            for text, score in core._collect_strip_pairs(strip, phase_bonus=bonus, profile=profile):
                best_pairs.append((str(text), float(score)))

    best_result = _empty_result(STATUS_NOT_FOUND, reason="strip_no_text_pairs")
    best_rank = -999.0

    for raw, score in best_pairs:
        candidate = _decode_full_code_text(str(raw), float(score), config)
        rank = _rank_strip_candidate(candidate)
        if rank > best_rank:
            best_rank = rank
            best_result = candidate

    result = best_result
    result["read_mode"] = "strip"
    result["strip_order"] = strip_order
    result["detected_count"] = len(detections)
    result["segmentation_method"] = "yolov11_boxes_strip"
    result["segmentation_score"] = float(
        sum(float(item.get("score") or 0.0) for item in detections[:11]) / float(max(1, min(11, len(detections))))
    )
    result["strip_shape"] = [int(strip.shape[0]), int(strip.shape[1])]
    result["text_pairs"] = [{"text": text, "score": round(float(score), 4)} for text, score in best_pairs[:12]]
    return result


def _decode_strip_symbols(
    img: np.ndarray,
    detections: Sequence[dict[str, Any]],
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    rois = _symbol_rois_from_detections(img, detections, config)
    if len(rois) < 10:
        result = _empty_result(STATUS_NOT_FOUND if not rois else STATUS_RAW_ONLY, reason="not_enough_yolo_boxes")
        result["detected_count"] = len(detections)
        return result

    forward = _decode_strip_rois(rois, detections, config, strip_order="top_down")
    if _is_good_strip_result(forward):
        return forward

    backward = _decode_strip_rois(list(reversed(rois)), detections, config, strip_order="bottom_up")
    if _rank_strip_candidate(backward) > _rank_strip_candidate(forward):
        backward["fallback_used"] = "bottom_up_strip"
        backward["top_down_candidate"] = {
            "raw_text": forward.get("raw_text"),
            "base10": forward.get("base10"),
            "full_code": forward.get("full_code"),
            "status": forward.get("status"),
            "score": forward.get("score"),
        }
        return backward

    forward["fallback_tried"] = "bottom_up_strip"
    forward["bottom_up_candidate"] = {
        "raw_text": backward.get("raw_text"),
        "base10": backward.get("base10"),
        "full_code": backward.get("full_code"),
        "status": backward.get("status"),
        "score": backward.get("score"),
    }
    return forward


def _decode_detected_symbols(
    img: np.ndarray,
    detections: Sequence[dict[str, Any]],
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if len(detections) < 10:
        result = _empty_result(STATUS_NOT_FOUND if not detections else STATUS_RAW_ONLY, reason="not_enough_yolo_boxes")
        result["detected_count"] = len(detections)
        return result

    cfg = _cfg(config)

    chars: list[str] = []
    raw_parts: list[str] = []
    read_scores: list[float] = []
    rows: list[dict[str, Any]] = []

    for idx, det in enumerate(detections[:11]):
        box = tuple(det["box"])
        roi = _crop_symbol(img, box, cfg)

        if roi is None:
            char, read_score, raw = "", -1.0, ""
        elif idx < 4:
            char, read_score, raw = _read_letter_roi(roi, cfg)
        else:
            char, read_score, raw = _read_digit_roi(roi, check_digit=(idx == 10), config=cfg)

        chars.append(char)
        raw_parts.append(raw or char)
        read_scores.append(float(read_score))
        rows.append(
            {
                "index": idx,
                "box": _serialize_box(box),
                "det_conf": round(float(det.get("score") or 0.0), 4),
                "char": char,
                "ocr_score": round(float(read_score), 4),
                "raw": raw,
            }
        )

    owner = "".join(chars[:4])
    base6 = "".join(chars[4:10])
    check_digit = chars[10] if len(chars) >= 11 else ""

    raw_text = "".join(part for part in chars if part) or "".join(part for part in raw_parts if part)

    score_values = [score for score in read_scores if score >= 0.0]
    ocr_score = sum(score_values) / float(len(score_values)) if score_values else -1.0
    det_score = sum(float(item.get("score") or 0.0) for item in detections[:11]) / float(max(1, min(11, len(detections))))
    score = (0.55 * max(ocr_score, 0.0)) + (0.45 * max(det_score, 0.0))

    if len(owner) != 4 or len(base6) != 6:
        result = _empty_result(STATUS_RAW_ONLY if raw_text else STATUS_NOT_FOUND, reason="paddle_symbol_read_failed")
        result.update(
            {
                "raw_text": raw_text,
                "score": float(score),
                "detected_count": len(detections),
                "symbols": rows,
                "segmentation_score": float(det_score),
            }
        )
        return result

    owner_corrected = False
    if len(check_digit) == 1 and check_digit.isdigit():
        owner, owner_corrected = _iso_correct_owner_confusions(owner, base6, check_digit)
        if owner_corrected:
            for row_idx, ch in enumerate(owner):
                rows[row_idx]["char"] = ch
                rows[row_idx]["iso_corrected"] = True

    base10 = f"{owner}{base6}"
    result = {
        "raw_text": raw_text,
        "base10": base10,
        "check_digit": "",
        "full_code": base10,
        "score": float(score),
        "is_valid_iso": False,
        "status": STATUS_PARTIAL_10,
        "check_digit_source": "none",
        "guessed_check_digit": False,
        "segmentation_score": float(det_score),
        "segmentation_method": "yolov11_boxes",
        "debug_dir": "",
        "detected_count": len(detections),
        "read_mode": "symbols",
        "symbols": rows,
    }
    if owner_corrected:
        result["owner_iso_corrected"] = True

    if len(check_digit) == 1 and check_digit.isdigit() and core._is_valid_iso6346(f"{base10}{check_digit}"):
        result["check_digit"] = check_digit
        result["full_code"] = f"{base10}{check_digit}"
        result["is_valid_iso"] = True
        result["status"] = STATUS_FULL_CONFIRMED
        result["check_digit_source"] = "paddle"
        return result

    if bool(cfg["vertical_use_iso_guess_for_check_digit"]):
        guessed = str(core._iso6346_check_digit(base10))
        result["check_digit"] = guessed
        result["full_code"] = f"{base10}{guessed}"
        result["is_valid_iso"] = True
        result["status"] = STATUS_FULL_GUESSED
        result["check_digit_source"] = "iso_guess"
        result["guessed_check_digit"] = True
        if check_digit:
            result["ocr_check_digit"] = check_digit
        return result

    if check_digit:
        result["check_digit"] = check_digit
        result["full_code"] = f"{base10}{check_digit}"
        result["status"] = "FULL_11"

    return result


def _vertical_prepare_aligned_column(img, config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    mask = (
        np.zeros(img.shape[:2], dtype=np.uint8)
        if img is not None and getattr(img, "size", 0)
        else np.zeros((1, 1), dtype=np.uint8)
    )
    return {
        "aligned_image": img,
        "mask": mask,
        "angle": 0.0,
        "column_bbox": None,
        "debug": {"method": "yolov11_boxes", "normalization": "none"},
    }


def _run_vertical_pipeline(
    img,
    *,
    config: Optional[dict[str, Any]] = None,
    allow_grid_fallback: bool = True,
    allow_strip_rescue: bool = True,
    simple_ocr: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], int]:
    prepared = _vertical_prepare_aligned_column(img, config)

    try:
        detections = _detect_symbol_boxes(img, config)
    except Exception as exc:
        result = _empty_result(STATUS_NOT_FOUND, reason=str(exc))
        return result, [], prepared, -1

    boxes = [tuple(item["box"]) for item in detections]
    candidate = {
        "name": "yolov11_boxes",
        "mask_name": "yolo",
        "selection_score": float(sum(float(item.get("score") or 0.0) for item in detections)),
        "segmentation_score": float(sum(float(item.get("score") or 0.0) for item in detections) / float(max(1, len(detections)))),
        "box_count": len(boxes),
        "boxes": boxes,
        "raw_boxes": boxes,
        "repairs": [],
        "profile": {"expected_h": 0, "threshold": 0.0, "focus_x": (0, 0)},
        "minima": [],
    }

    mode = str(_cfg(config).get("vertical_read_mode") or "symbols").strip().lower()
    logger.info("vertical_yolo_pipeline mode=%s detections=%d", mode, len(detections))

    if mode in {"strip", "line", "string"}:
        result = _decode_strip_symbols(img, detections, config)
    else:
        result = _decode_detected_symbols(img, detections, config)

    result["segmentation_boxes"] = [
        {
            "index": idx,
            "box": _serialize_box(tuple(item["box"])),
            "score": round(float(item.get("score") or 0.0), 4),
        }
        for idx, item in enumerate(detections[:11])
    ]
    result["segmentation_overlay_type"] = "vertical_symbols"

    return result, [candidate] if detections else [], prepared, 0 if detections else -1


def ocr_vertical_narrow_boxed(img, config=None) -> dict[str, Any]:
    result, _candidates, _prepared, _best_index = _run_vertical_pipeline(img, config=config)
    return result


def ocr_vertical_container_regions_charwise(img, config=None) -> dict[str, Any]:
    result, _candidates, _prepared, _best_index = _run_vertical_pipeline(img, config=config)
    return result


def ocr_vertical_container_regions(img, config=None) -> dict[str, Any]:
    result, _candidates, _prepared, _best_index = _run_vertical_pipeline(img, config=config)
    return result


def _annotate_boxes(img: np.ndarray, boxes: Sequence[Box]) -> np.ndarray:
    if img is None or getattr(img, "size", 0) == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    canvas = core._ensure_bgr(img).copy()
    for idx, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            canvas,
            str(idx),
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _render_profile_image(profile: dict[str, Any], minima: Sequence[int]) -> np.ndarray:
    canvas = np.full((140, 360, 3), 255, dtype=np.uint8)
    cv2.putText(
        canvas,
        "YOLOv11 boxes",
        (18, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (40, 70, 200),
        2,
        cv2.LINE_AA,
    )
    return canvas
