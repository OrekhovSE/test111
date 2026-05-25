#!/usr/bin/env python3

from __future__ import annotations

import json
import mimetypes
import os
import re
import time

import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from typing import Any
from urllib import error, request


SERVER_IP = "10.3.50.27"
SERVICE_PORT = 8000
RECOGNIZE_URL = os.getenv("RECOGNIZE_URL", f"http://{SERVER_IP}:{SERVICE_PORT}/recognize")
DATASET_DIR = r"C:\Users\GPuzhalin\Desktop\container_seals_data\AutoReceive"
TIMEOUT_SEC = float(os.getenv("TIMEOUT_SEC", "120"))
LOG_DIR = Path(os.getenv("TEST_LOG_DIR", "logs"))

IMAGE_EXTS = {".jpg", ".jpeg"}

@dataclass
class EvalResult:
    image: str
    status: str
    expected: str | None
    best_got: str | None
    reason: str
    processed: int | None
    total_results: int | None
    matched_index: int | None
    payload: dict[str, Any] | None
    time: float | None


def guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def normalize(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", "", text).upper()


def load_expected(txt_path: Path) -> str | None:
    if not txt_path.exists():
        return None
    return txt_path.read_text(encoding="utf-8-sig", errors="replace").strip()


def build_multipart_payload(field_name: str, file_path: Path, mime: str) -> tuple[bytes, str]:
    boundary = f"----ocr-eval-{uuid.uuid4().hex}"
    line_break = b"\r\n"
    file_bytes = file_path.read_bytes()

    chunks = [
        f"--{boundary}".encode("utf-8"),
        f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"'.encode("utf-8"),
        f"Content-Type: {mime}".encode("utf-8"),
        b"",
        file_bytes,
        f"--{boundary}--".encode("utf-8"),
        b"",
    ]

    body = line_break.join(chunks)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def call_recognize(image_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    mime = guess_mime(image_path)
    try:
        body, content_type = build_multipart_payload("files", image_path, mime)
        req = request.Request(
            RECOGNIZE_URL,
            data=body,
            method="POST",
            headers={"Content-Type": content_type, "Content-Length": str(len(body))},
        )
        with request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            status_code = getattr(resp, "status", 200)
    except error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        return None, f"http_{exc.code}: {err_body}"
    except Exception as exc:
        return None, f"request_failed: {exc}"

    if status_code != 200:
        return None, f"http_{status_code}: {raw}"

    try:
        return json.loads(raw), None
    except Exception:
        return None, f"bad_json: {raw}"


def evaluate_image(image_path: Path) -> EvalResult:
    expected = load_expected(image_path.with_suffix(".txt"))
    if not expected:
        return EvalResult(
            image=image_path.name,
            status="SKIP",
            expected=None,
            best_got=None,
            reason="txt_missing_or_empty",
            processed=None,
            total_results=None,
            matched_index=None,
            payload=None,
            time=None,
        )

    payload, err = call_recognize(image_path)
    if err:
        return EvalResult(
            image=image_path.name,
            status="API_ERROR",
            expected=expected,
            best_got=None,
            reason=err,
            processed=None,
            total_results=None,
            matched_index=None,
            payload=None,
            time=None,
        )

    if not isinstance(payload, dict):
        return EvalResult(
            image=image_path.name,
            status="OTHER_ERROR",
            expected=expected,
            best_got=None,
            reason="response_is_not_json_object",
            processed=None,
            total_results=None,
            matched_index=None,
            payload=None,
            time=None,
        )

    processed = payload.get("processed")
    results = payload.get("results")

    if not isinstance(processed, int) or not isinstance(results, list):
        return EvalResult(
            image=image_path.name,
            status="OTHER_ERROR",
            expected=expected,
            best_got=None,
            reason="unexpected_response_schema",
            processed=processed if isinstance(processed, int) else None,
            total_results=len(results) if isinstance(results, list) else None,
            matched_index=None,
            payload=payload,
            time=None,
        )

    if processed == 0 or len(results) == 0:
        return EvalResult(
            image=image_path.name,
            status="YOLO_ERROR",
            expected=expected,
            best_got=None,
            reason="no_crops_or_no_results",
            processed=processed,
            total_results=len(results),
            matched_index=None,
            payload=payload,
            time=None,
        )

    expected_norm = normalize(expected)
    best_got: str | None = None
    best_score = -1.0
    matched_index: int | None = None
    non_empty_predictions = 0

    for idx, item in enumerate(results):
        if not isinstance(item, dict):
            continue

        got = str(item.get("result", "") or "").strip()
        got_norm = normalize(got)

        if got_norm:
            non_empty_predictions += 1

        score_raw = item.get("score", -1)
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            score = -1.0

        if score > best_score:
            best_score = score
            best_got = got

        if got_norm and got_norm == expected_norm:
            matched_index = idx
            break

    if matched_index is not None:
        return EvalResult(
            image=image_path.name,
            status="OK",
            expected=expected,
            best_got=best_got,
            reason="match_found_in_results",
            processed=processed,
            total_results=len(results),
            matched_index=matched_index,
            payload=payload,
            time=None,
        )

    if non_empty_predictions == 0:
        return EvalResult(
            image=image_path.name,
            status="OCR_ERROR",
            expected=expected,
            best_got=best_got,
            reason="all_predictions_empty",
            processed=processed,
            total_results=len(results),
            matched_index=None,
            payload=payload,
            time=None,
        )

    return EvalResult(
        image=image_path.name,
        status="OCR_ERROR",
        expected=expected,
        best_got=best_got,
        reason="predictions_present_but_no_exact_match",
        processed=processed,
        total_results=len(results),
        matched_index=None,
        payload=payload,
        time=None,
    )


def get_img_eval_stat(img_path: Path) -> EvalResult:
    start = time.perf_counter()
    eval_stat = evaluate_image(img_path)
    end = time.perf_counter()
    eval_stat.time = round((end - start) * 100)
    return eval_stat


def write_logs(run_id: str, rows: list[EvalResult], dataset_dir: Path) -> tuple[Path, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = LOG_DIR / f"ocr_eval_{run_id}.jsonl"
    txt_path = LOG_DIR / f"ocr_eval_{run_id}.log"

    with jsonl_path.open("w", encoding="utf-8") as jf, txt_path.open("w", encoding="utf-8") as tf:
        tf.write(f"RUN_ID: {run_id}\n")
        tf.write(f"DATASET_DIR: {dataset_dir}\n")
        tf.write(f"RECOGNIZE_URL: {RECOGNIZE_URL}\n")
        tf.write(f"TIMEOUT_SEC: {TIMEOUT_SEC}\n")
        tf.write("=" * 100 + "\n")

        for row in rows:
            jf.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            payload_dump = json.dumps(row.payload, ensure_ascii=False) if row.payload is not None else "null"
            tf.write(
                f"[{row.status}] image={row.image} expected={row.expected!r} best_got={row.best_got!r} "
                f"reason={row.reason} processed={row.processed} total_results={row.total_results} "
                f"matched_index={row.matched_index}\n"
            )
            tf.write(f"payload: {payload_dump}\n")
            tf.write("-" * 100 + "\n")

    return jsonl_path, txt_path


def print_infomsg(infomsg: str):
    print(f"[INFO] {infomsg}", file=sys.stdout, flush=True)


def print_errmsg(errmsg: str):
    print(f"[ERR] {errmsg}", file=sys.stderr, flush=True)


_stop_processing = False
def stop_processing():
    global _stop_processing
    _stop_processing = True

def reset_stop_flag():
    global _stop_processing
    _stop_processing = False


def process_dataset() -> int:
    reset_stop_flag()

    print_infomsg("Оценка работы сервиса OCR началась")

    dataset_dir = DATASET_DIR if isinstance(DATASET_DIR, Path) else Path(str(DATASET_DIR))

    if str(dataset_dir) == "PUT_DATASET_FOLDER_PATH_HERE":
        print_errmsg("Укажите путь к датасету в переменной DATASET_DIR внутри test.py")
        return 2

    if not dataset_dir.exists():
        print_errmsg(f"Папка не найдена: {dataset_dir.resolve()}")
        return 2

    images = sorted([p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if not images:
        print_errmsg(f"В папке нет файлов поддерживаемых расширений: '{dataset_dir.resolve()}'")
        return 2

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_stats: list[EvalResult] = []

    counters = {
        "TOTAL": 0,
        "OK": 0,
        "SKIP": 0,
        "YOLO_ERROR": 0,
        "OCR_ERROR": 0,
        "API_ERROR": 0,
        "OTHER_ERROR": 0,
    }

    print(f"RUN_ID       : {run_id}", flush=True)
    print(f"DATASET_DIR  : {dataset_dir.resolve()}", flush=True)
    print(f"RECOGNIZE_URL: {RECOGNIZE_URL}", flush=True)
    print(f"TIMEOUT_SEC  : {TIMEOUT_SEC}", flush=True)
    print("-" * 100, flush=True)


    for image_path in images:
        if _stop_processing:
            print("\nСтоп. Обработка датасета была прервана пользователем", flush=True)
            break
        counters["TOTAL"] += 1
        eval_stat = get_img_eval_stat(image_path)
        eval_stats.append(eval_stat)
        counters[eval_stat.status] += 1
        print("[%-10s] %-20s %-20s %-8s %s\n" % (eval_stat.status,
                                            f"expected={eval_stat.expected!r}",
                                            f"best_got={eval_stat.best_got!r}",
                                            f"elapsed={eval_stat.time}ms",
                                            f"reason={eval_stat.reason}"))

    jsonl_path, txt_path = write_logs(run_id, eval_stats, dataset_dir)

    print("-" * 100, flush=True)
    for key, value in counters.items():
        print(f"{key:<11}: {value}", flush=True)

    print(f"LOG_JSONL   : {jsonl_path}", flush=True)
    print(f"LOG_DETAILS : {txt_path}", flush=True)

    failures = counters["YOLO_ERROR"] + counters["OCR_ERROR"] + counters["API_ERROR"] + counters["OTHER_ERROR"]
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(process_dataset())
