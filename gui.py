#!/usr/bin/env python3

from __future__ import annotations

import json
import mimetypes
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import requests
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_API_BASE_URL = "http://127.0.0.1:8010"
DEFAULT_TIMEOUT_SEC = 120.0
REQUEST_FILES_FIELD = "files"


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


CONFIG_PATH = app_dir() / "gui_config.json"


MODES: dict[str, dict[str, str]] = {
    "seal": {
        "title": "Пломба",
        "recognize_endpoint": "/RecognizeSealNumber",
        "feedback_endpoint": "/bad_recognize",
        "success_label": "Номер пломбы",
    },
    "container": {
        "title": "Контейнер",
        "recognize_endpoint": "/RecognizeContainerNumber",
        "feedback_endpoint": "/bad_recognize_container",
        "success_label": "Номер контейнера",
    },
}


try:
    from PIL import Image, ImageTk

    PIL_AVAILABLE = True
except ImportError:
    Image = None
    ImageTk = None
    PIL_AVAILABLE = False


@dataclass
class AppConfig:
    api_base_url: str = DEFAULT_API_BASE_URL
    timeout_sec: float = DEFAULT_TIMEOUT_SEC


@dataclass
class RecognitionSession:
    image_path: Path
    mode: str
    api_base_url: str
    response_data: dict[str, Any]
    expected_text: str = ""


def guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def load_expected_text(image_path: Path) -> str:
    txt_path = image_path.with_suffix(".txt")
    if not txt_path.exists():
        return ""
    try:
        return txt_path.read_text(encoding="utf-8-sig", errors="replace").strip()
    except Exception:
        return ""


def load_app_config() -> AppConfig:
    if not CONFIG_PATH.exists():
        cfg = AppConfig()
        save_app_config(cfg)
        return cfg
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return AppConfig(
            api_base_url=str(raw.get("api_base_url") or DEFAULT_API_BASE_URL).strip() or DEFAULT_API_BASE_URL,
            timeout_sec=float(raw.get("timeout_sec") or DEFAULT_TIMEOUT_SEC),
        )
    except Exception:
        cfg = AppConfig()
        save_app_config(cfg)
        return cfg


def save_app_config(config: AppConfig) -> None:
    CONFIG_PATH.write_text(pretty_json(asdict(config)), encoding="utf-8")


class DesktopClientApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Клиент распознавания пломб и контейнеров")
        self.root.geometry("1220x840")
        self.root.minsize(1080, 760)
        self.root.configure(bg="#08111f")

        self.config_data = load_app_config()
        self.current_image_path: Path | None = None
        self.current_session: RecognitionSession | None = None
        self.preview_photo = None
        self.is_busy = False
        self.detail_view_var = tk.BooleanVar(value=False)
        self.worker_queue: Queue[tuple[str, Any]] = Queue()
        self.colors = {
            "button_seal": "#4e6484",
            "button_seal_active": "#445875",
            "button_container": "#486b69",
            "button_container_active": "#3f5f5d",
            "button_primary": "#5a7098",
            "button_primary_active": "#506487",
            "button_secondary": "#414f66",
            "button_secondary_active": "#4b5b74",
            "button_feedback": "#7b5f69",
            "button_feedback_active": "#6d545d",
            "badge_idle": "#415066",
            "badge_progress": "#536989",
            "badge_success": "#416b65",
            "badge_warning": "#8a6f43",
            "badge_error": "#79515a",
            "toggle_on": "#607da8",
            "toggle_off": "#3b4657",
        }

        self.mode_var = tk.StringVar(value="seal")
        self.api_base_var = tk.StringVar(value=self.config_data.api_base_url)
        self.timeout_var = tk.StringVar(value=str(int(self.config_data.timeout_sec)))
        self.feedback_expected_var = tk.StringVar(value="")

        self._build_styles()
        self._build_ui()
        self._poll_worker_queue()

    def _build_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background="#08111f")
        style.configure("Card.TFrame", background="#111827")
        style.configure("App.TRadiobutton", background="#111827", foreground="#e2e8f0", font=("Segoe UI", 10))

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, style="App.TFrame", padding=16)
        outer.pack(fill=tk.BOTH, expand=True)

        self.page_host = tk.Frame(outer, bg="#08111f")
        self.page_host.pack(fill=tk.BOTH, expand=True)
        self.page_host.grid_rowconfigure(0, weight=1)
        self.page_host.grid_columnconfigure(0, weight=1)

        self.main_tab = tk.Frame(self.page_host, bg="#08111f")
        self.settings_tab = tk.Frame(self.page_host, bg="#08111f")
        for frame in (self.main_tab, self.settings_tab):
            frame.grid(row=0, column=0, sticky="nsew")

        self._build_main_tab()
        self._build_settings_tab()
        self._build_menu()
        self._update_mode_badges()
        self._show_page("main")

    def _build_menu(self) -> None:
        menu_bar = tk.Menu(self.root)

        app_menu = tk.Menu(menu_bar, tearoff=0)
        app_menu.add_command(label="Главная", command=lambda: self._show_page("main"))
        app_menu.add_command(label="Настройки", command=lambda: self._show_page("settings"))
        app_menu.add_separator()
        app_menu.add_command(label="Выход", command=self.root.destroy)
        menu_bar.add_cascade(label="Приложение", menu=app_menu)

        service_menu = tk.Menu(menu_bar, tearoff=0)
        service_menu.add_command(label="Сохранить настройки", command=self.save_settings)
        service_menu.add_command(label="Сбросить настройки", command=self.reset_settings)
        menu_bar.add_cascade(label="Сервис", menu=service_menu)

        self.root.config(menu=menu_bar)

    def _show_page(self, page_name: str) -> None:
        target = {
            "main": self.main_tab,
            "settings": self.settings_tab,
        }.get(page_name, self.main_tab)
        target.tkraise()

    def _build_main_tab(self) -> None:
        self.main_tab.grid_columnconfigure(0, weight=3)
        self.main_tab.grid_columnconfigure(1, weight=2)
        self.main_tab.grid_rowconfigure(0, weight=1)

        left = tk.Frame(self.main_tab, bg="#111827", padx=18, pady=18)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 0))

        left_top = tk.Frame(left, bg="#111827")
        left_top.pack(fill=tk.X)
        tk.Label(left_top, text="Изображение", bg="#111827", fg="#f8fafc", font=("Segoe UI", 13, "bold")).pack(anchor="w", side=tk.LEFT)

        mode_shell = tk.Frame(left_top, bg="#111827")
        mode_shell.pack(side=tk.RIGHT)

        self.mode_seal_button = tk.Button(
            mode_shell,
            text="Пломба",
            command=lambda: self._recognize_with_mode("seal"),
            bg=self.colors["button_seal"],
            fg="#f8fafc",
            activebackground=self.colors["button_seal_active"],
            activeforeground="#f8fafc",
            relief=tk.FLAT,
            padx=18,
            pady=10,
            bd=0,
            highlightthickness=0,
        )
        self.mode_seal_button.pack(side=tk.LEFT)
        self.mode_container_button = tk.Button(
            mode_shell,
            text="Контейнер",
            command=lambda: self._recognize_with_mode("container"),
            bg=self.colors["button_container"],
            fg="#f8fafc",
            activebackground=self.colors["button_container_active"],
            activeforeground="#f8fafc",
            relief=tk.FLAT,
            padx=18,
            pady=10,
            bd=0,
            highlightthickness=0,
        )
        self.mode_container_button.pack(side=tk.LEFT, padx=(8, 0))

        self.file_info_label = tk.Label(left, text="Файл не выбран", bg="#111827", fg="#94a3b8", font=("Segoe UI", 10))
        self.file_info_label.pack(anchor="w", pady=(6, 12))

        self.drop_frame = tk.Frame(left, bg="#0f172a", bd=2, relief=tk.GROOVE, cursor="hand2")
        self.drop_frame.pack(fill=tk.BOTH, expand=True)
        self.drop_frame.pack_propagate(False)

        self.preview_label = tk.Label(
            self.drop_frame,
            text="Нажмите, чтобы выбрать изображение",
            bg="#0f172a",
            fg="#cbd5e1",
            font=("Segoe UI", 16, "bold"),
        )
        self.preview_label.pack(expand=True)

        self.preview_hint_label = tk.Label(
            left,
            text="Поддерживаются JPG, PNG, BMP, WEBP",
            bg="#111827",
            fg="#64748b",
            font=("Segoe UI", 10),
        )
        self.preview_hint_label.pack(anchor="w", pady=(10, 0))

        for widget in (self.drop_frame, self.preview_label, self.preview_hint_label):
            widget.bind("<Button-1>", lambda _event: self.select_image())

        right = tk.Frame(self.main_tab, bg="#111827", padx=18, pady=18)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=(0, 0))

        tk.Label(right, text="Распознавание", bg="#111827", fg="#f8fafc", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Label(
            right,
            text="Сначала выберите изображение. Затем нажмите «Пломба» или «Контейнер» над фото, чтобы отправить запрос.",
            bg="#111827",
            fg="#94a3b8",
            font=("Segoe UI", 10),
            justify=tk.LEFT,
            wraplength=320,
        ).pack(anchor="w", pady=(6, 18))

        self.status_badge = tk.Label(
            right,
            text="Ожидание запроса",
            bg=self.colors["badge_idle"],
            fg="#f8fafc",
            font=("Segoe UI", 11, "bold"),
            padx=12,
            pady=8,
        )
        self.status_badge.pack(anchor="w", pady=(20, 12))

        self.result_value_label = tk.Label(
            right,
            text="Результат пока отсутствует",
            bg="#111827",
            fg="#f8fafc",
            font=("Segoe UI", 24, "bold"),
            justify=tk.LEFT,
            wraplength=340,
        )
        self.result_value_label.pack(anchor="w")

        self.result_meta_label = tk.Label(
            right,
            text="Выберите изображение кликом по рамке. Запрос отправится по нажатию на кнопку режима над фото.",
            bg="#111827",
            fg="#94a3b8",
            font=("Segoe UI", 10),
            justify=tk.LEFT,
            wraplength=340,
        )
        self.result_meta_label.pack(anchor="w", pady=(10, 22))

        toggle_row = tk.Frame(right, bg="#111827")
        toggle_row.pack(fill=tk.X, pady=(0, 14))
        tk.Label(toggle_row, text="Подробный результат", bg="#111827", fg="#cbd5e1", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)

        self.toggle_canvas = tk.Canvas(toggle_row, width=64, height=34, bg="#111827", highlightthickness=0, bd=0, cursor="hand2")
        self.toggle_canvas.pack(side=tk.RIGHT)
        self.toggle_canvas.bind("<Button-1>", lambda _event: self._toggle_detail_view())
        self._render_detail_toggle()

        detail_wrap = tk.Frame(right, bg="#111827")
        detail_wrap.pack(fill=tk.BOTH, expand=True)
        detail_wrap.pack_propagate(False)

        self.detail_text = tk.Text(
            detail_wrap,
            wrap=tk.WORD,
            bg="#020617",
            fg="#cbd5e1",
            insertbackground="#e2e8f0",
            relief=tk.FLAT,
            font=("Cascadia Code", 10),
            height=16,
        )
        self.detail_scroll = tk.Scrollbar(detail_wrap, command=self.detail_text.yview)
        self.detail_text.configure(yscrollcommand=self.detail_scroll.set)
        self.detail_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.detail_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._set_text(self.detail_text, "{\n  \"status\": \"idle\"\n}")
        self._update_detail_visibility()

        self.feedback_button = tk.Button(
            right,
            text="Отправить обратную связь",
            command=self.open_feedback_panel,
            state=tk.DISABLED,
            bg=self.colors["button_feedback"],
            fg="#f8fafc",
            activebackground=self.colors["button_feedback_active"],
            activeforeground="#f8fafc",
            relief=tk.FLAT,
            padx=14,
            pady=10,
        )
        self.feedback_button.pack(fill=tk.X, pady=(14, 0))
        self.feedback_button.pack_forget()

        self.feedback_panel = tk.Frame(right, bg="#0f172a", padx=14, pady=14)
        self.feedback_panel.pack(fill=tk.X, pady=(12, 0))
        self.feedback_panel.pack_forget()

        tk.Label(
            self.feedback_panel,
            text="Обратная связь",
            bg="#0f172a",
            fg="#f8fafc",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")
        self.feedback_service_label = tk.Label(
            self.feedback_panel,
            text="",
            bg="#0f172a",
            fg="#94a3b8",
            font=("Segoe UI", 10),
            wraplength=320,
            justify=tk.LEFT,
        )
        self.feedback_service_label.pack(anchor="w", pady=(6, 12))
        tk.Label(
            self.feedback_panel,
            text="Какой результат должен быть:",
            bg="#0f172a",
            fg="#e2e8f0",
            font=("Segoe UI", 10),
        ).pack(anchor="w")
        self.feedback_expected_entry = tk.Entry(self.feedback_panel, textvariable=self.feedback_expected_var)
        self.feedback_expected_entry.pack(fill=tk.X, pady=(6, 12))
        feedback_actions = tk.Frame(self.feedback_panel, bg="#0f172a")
        feedback_actions.pack(fill=tk.X)
        self.feedback_cancel_button = tk.Button(
            feedback_actions,
            text="Отмена",
            command=self.close_feedback_panel,
            bg=self.colors["button_secondary"],
            fg="#f8fafc",
            activebackground=self.colors["button_secondary_active"],
            activeforeground="#f8fafc",
            relief=tk.FLAT,
            padx=12,
            pady=8,
        )
        self.feedback_cancel_button.pack(side=tk.RIGHT, padx=(8, 0))
        self.feedback_send_button = tk.Button(
            feedback_actions,
            text="Отправить",
            command=self.submit_feedback_from_panel,
            bg=self.colors["button_primary"],
            fg="#f8fafc",
            activebackground=self.colors["button_primary_active"],
            activeforeground="#f8fafc",
            relief=tk.FLAT,
            padx=12,
            pady=8,
        )
        self.feedback_send_button.pack(side=tk.RIGHT)
        self.feedback_status_label = tk.Label(
            self.feedback_panel,
            text="",
            bg="#0f172a",
            fg="#94a3b8",
            font=("Segoe UI", 10),
            wraplength=320,
            justify=tk.LEFT,
        )
        self.feedback_status_label.pack(anchor="w", pady=(12, 0))

        self.status_line = tk.Label(
            right,
            text="Готово к работе.",
            bg="#111827",
            fg="#64748b",
            font=("Segoe UI", 10),
            justify=tk.LEFT,
            wraplength=340,
        )
        self.status_line.pack(anchor="w", pady=(16, 0))

    def _build_settings_tab(self) -> None:
        self.settings_tab.grid_columnconfigure(0, weight=1)

        card = tk.Frame(self.settings_tab, bg="#111827", padx=18, pady=18)
        card.grid(row=0, column=0, sticky="new")

        tk.Label(card, text="Настройки подключения", bg="#111827", fg="#f8fafc", font=("Segoe UI", 13, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        tk.Label(
            card,
            text=f"Конфигурация хранится в файле: {CONFIG_PATH.name}",
            bg="#111827",
            fg="#94a3b8",
            font=("Segoe UI", 10),
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 16))

        tk.Label(card, text="URL сервиса", bg="#111827", fg="#e2e8f0").grid(row=2, column=0, sticky="w")
        tk.Entry(card, textvariable=self.api_base_var, width=44).grid(row=3, column=0, sticky="ew", pady=(6, 14))

        tk.Label(card, text="Таймаут, сек", bg="#111827", fg="#e2e8f0").grid(row=4, column=0, sticky="w")
        tk.Entry(card, textvariable=self.timeout_var, width=14).grid(row=5, column=0, sticky="w", pady=(6, 14))

        buttons = tk.Frame(card, bg="#111827")
        buttons.grid(row=6, column=0, sticky="w")
        tk.Button(
            buttons,
            text="Сохранить настройки",
            command=self.save_settings,
            bg=self.colors["button_primary"],
            fg="#f8fafc",
            activebackground=self.colors["button_primary_active"],
            activeforeground="#f8fafc",
            relief=tk.FLAT,
            padx=14,
            pady=10,
        ).pack(side=tk.LEFT)
        tk.Button(
            buttons,
            text="Сбросить по умолчанию",
            command=self.reset_settings,
            bg=self.colors["button_secondary"],
            fg="#f8fafc",
            activebackground=self.colors["button_secondary_active"],
            activeforeground="#f8fafc",
            relief=tk.FLAT,
            padx=14,
            pady=10,
        ).pack(side=tk.LEFT, padx=(10, 0))

        self.settings_status_label = tk.Label(
            card,
            text="Настройки загружены.",
            bg="#111827",
            fg="#64748b",
            font=("Segoe UI", 10),
        )
        self.settings_status_label.grid(row=7, column=0, sticky="w", pady=(16, 0))

        card.grid_columnconfigure(0, weight=1)

    def _set_text(self, widget: tk.Text, value: str) -> None:
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", value)
        widget.config(state=tk.DISABLED)

    def _set_mode(self, mode: str) -> None:
        self.mode_var.set(mode)
        if self.current_session is None:
            self._set_idle_result_state()

    def _recognize_with_mode(self, mode: str) -> None:
        self._set_mode(mode)
        self.start_recognition()

    def _update_mode_badges(self) -> None:
        self.mode_seal_button.config(
            bg=self.colors["button_seal"],
            fg="#f8fafc",
            activebackground=self.colors["button_seal_active"],
            activeforeground="#f8fafc",
        )
        self.mode_container_button.config(
            bg=self.colors["button_container"],
            fg="#f8fafc",
            activebackground=self.colors["button_container_active"],
            activeforeground="#f8fafc",
        )

    def select_image(self) -> None:
        if self.is_busy:
            messagebox.showinfo("Идёт обработка", "Дождитесь завершения текущего распознавания.")
            return
        filetypes = [("Изображения", " ".join(f"*{ext}" for ext in sorted(IMAGE_EXTS))), ("Все файлы", "*.*")]
        path = filedialog.askopenfilename(title="Выберите изображение", filetypes=filetypes)
        if not path:
            return

        image_path = Path(path)
        if image_path.suffix.lower() not in IMAGE_EXTS:
            messagebox.showerror("Неподдерживаемый файл", f"Поддерживаются: {', '.join(sorted(IMAGE_EXTS))}")
            return

        self.current_image_path = image_path
        self.current_session = None
        self.feedback_button.pack_forget()
        self.close_feedback_panel()
        expected_text = load_expected_text(image_path)
        expected_suffix = f" | эталон: {expected_text}" if expected_text else ""
        self.file_info_label.config(text=f"{image_path.name} | {image_path.stat().st_size} байт{expected_suffix}")
        self._render_preview()
        self._set_idle_result_state()

    def _render_preview(self) -> None:
        if self.current_image_path is None:
            self.preview_label.config(image="", text="Нажмите, чтобы выбрать изображение")
            self.preview_hint_label.config(text="Поддерживаются JPG, PNG, BMP, WEBP")
            self.preview_photo = None
            return

        if not PIL_AVAILABLE:
            self.preview_label.config(image="", text=f"{self.current_image_path.name}\n\nДля превью установите Pillow")
            self.preview_hint_label.config(text="")
            self.preview_photo = None
            return

        try:
            image = Image.open(self.current_image_path)
            image.thumbnail((620, 560), Image.Resampling.LANCZOS)
            self.preview_photo = ImageTk.PhotoImage(image)
            self.preview_label.config(image=self.preview_photo, text="")
            self.preview_hint_label.config(text="Нажмите на рамку, чтобы выбрать другое изображение")
        except Exception as exc:
            self.preview_label.config(image="", text=f"Не удалось открыть изображение:\n{exc}")
            self.preview_hint_label.config(text="")
            self.preview_photo = None

    def _set_idle_result_state(self) -> None:
        mode_title = MODES[self.mode_var.get()]["title"]
        self.status_badge.config(text=f"Режим: {mode_title}", bg=self.colors["badge_idle"])
        self.result_value_label.config(text="Результат пока отсутствует")
        self.result_meta_label.config(text="После выбора изображения нажмите кнопку режима над фото, чтобы запустить запрос.")
        self.status_line.config(text="Изображение выбрано.")
        self._set_text(self.detail_text, "{\n  \"status\": \"idle\"\n}")

    def start_recognition(self) -> None:
        if self.is_busy:
            return
        if self.current_image_path is None:
            messagebox.showwarning("Нет файла", "Сначала выберите изображение.")
            return

        api_base_url = self.api_base_var.get().strip().rstrip("/")
        if not api_base_url:
            messagebox.showwarning("Нет адреса", "Укажите URL сервиса в разделе «Настройки».")
            return

        try:
            timeout_sec = float(self.timeout_var.get().strip())
        except ValueError:
            messagebox.showwarning("Неверный таймаут", "Таймаут должен быть числом.")
            return

        mode = self.mode_var.get()
        endpoint = MODES[mode]["recognize_endpoint"]
        self.is_busy = True
        self._set_image_picker_enabled(False)
        self.feedback_button.pack_forget()
        self.close_feedback_panel()
        self.status_badge.config(text="Идёт запрос...", bg=self.colors["badge_progress"])
        self.result_value_label.config(text="Выполняется распознавание")
        self.result_meta_label.config(text=f"Запрос отправлен в {endpoint}")
        self.status_line.config(text="Ожидаем ответ от сервиса...")
        self._set_text(self.detail_text, "Распознавание выполняется...")

        worker = threading.Thread(
            target=self._recognize_worker,
            args=(self.current_image_path, mode, api_base_url, timeout_sec),
            daemon=True,
        )
        worker.start()

    def _recognize_worker(self, image_path: Path, mode: str, api_base_url: str, timeout_sec: float) -> None:
        endpoint = MODES[mode]["recognize_endpoint"]
        try:
            with image_path.open("rb") as fh:
                response = requests.post(
                    f"{api_base_url}{endpoint}",
                    files={REQUEST_FILES_FIELD: (image_path.name, fh, guess_mime(image_path))},
                    timeout=timeout_sec,
                )
            response.raise_for_status()
            data = response.json()
            self.worker_queue.put(("recognize_ok", (image_path, mode, api_base_url, data)))
        except Exception as exc:
            self.worker_queue.put(("recognize_error", str(exc)))

    def _poll_worker_queue(self) -> None:
        try:
            while True:
                event, payload = self.worker_queue.get_nowait()
                if event == "recognize_ok":
                    image_path, mode, api_base_url, data = payload
                    self._on_recognition_success(image_path, mode, api_base_url, data)
                elif event == "recognize_error":
                    self._on_recognition_error(str(payload))
                elif event == "feedback_ok":
                    self._on_feedback_success(payload)
                elif event == "feedback_error":
                    self._on_feedback_error(str(payload))
        except Empty:
            pass
        finally:
            self.root.after(120, self._poll_worker_queue)

    def _on_recognition_success(self, image_path: Path, mode: str, api_base_url: str, data: dict[str, Any]) -> None:
        self.is_busy = False
        self._set_image_picker_enabled(True)
        self.current_session = RecognitionSession(
            image_path=image_path,
            mode=mode,
            api_base_url=api_base_url,
            response_data=data,
            expected_text=load_expected_text(image_path),
        )
        self.feedback_button.config(state=tk.NORMAL, text="Отправить обратную связь")
        self.feedback_button.pack(fill=tk.X, pady=(14, 0))
        self.feedback_expected_var.set(self.current_session.expected_text or "")
        self.feedback_service_label.config(text="")
        self.feedback_status_label.config(text="", fg="#94a3b8")

        result_value = str(data.get("result") or "")
        raw_text = str(data.get("raw_text") or "")
        failure_reason = str(data.get("failure_reason") or "")
        timings = data.get("timings_ms") or {}

        if result_value and result_value != "NOT_FOUND":
            title = MODES[mode]["success_label"]
            self.status_badge.config(text="Успешно", bg=self.colors["badge_success"])
            self.result_value_label.config(text=result_value)
            self.result_meta_label.config(text=f"{title} найден.")
        else:
            self.status_badge.config(text="Не распознано", bg=self.colors["badge_warning"])
            self.result_value_label.config(text="NOT_FOUND")
            self.result_meta_label.config(text=f"Причина: {failure_reason or 'не указана'}")

        summary_lines = [
            f"Режим: {MODES[mode]['title']}",
            f"Файл: {image_path.name}",
            f"Результат: {result_value or 'пусто'}",
            f"Raw OCR: {raw_text or 'пусто'}",
            f"Причина: {failure_reason or '-'}",
        ]
        if isinstance(timings, dict) and timings:
            summary_lines.append("")
            summary_lines.append("Тайминги, мс:")
            for key, value in timings.items():
                summary_lines.append(f"  {key}: {value}")

        self.status_line.config(text="Распознавание завершено.")
        self._set_text(self.detail_text, "\n".join(summary_lines) + "\n\n" + pretty_json(data))

    def _on_recognition_error(self, error_text: str) -> None:
        self.is_busy = False
        self._set_image_picker_enabled(True)
        self.feedback_button.pack_forget()
        self.close_feedback_panel()
        self.current_session = None
        self.status_badge.config(text="Ошибка", bg=self.colors["badge_error"])
        self.result_value_label.config(text="Сервис не ответил")
        self.result_meta_label.config(text="Проверьте настройки подключения и доступность API.")
        self.status_line.config(text="Не удалось выполнить запрос к сервису.")
        self._set_text(self.detail_text, pretty_json({"status": "error", "detail": error_text}))

    def open_feedback_panel(self) -> None:
        if self.current_session is None:
            messagebox.showwarning("Нет данных", "Сначала выполните распознавание.")
            return
        recognized_text = str(self.current_session.response_data.get("result") or "")
        if recognized_text == "NOT_FOUND":
            recognized_text = str(self.current_session.response_data.get("raw_text") or "")
        self.feedback_expected_var.set(self.current_session.expected_text or "")
        self.feedback_service_label.config(text=f"Сервис вернул: {recognized_text or 'пусто'}")
        self.feedback_status_label.config(text="Укажите корректный результат и нажмите «Отправить».", fg="#94a3b8")
        self.feedback_expected_entry.config(state=tk.NORMAL)
        self.feedback_send_button.config(state=tk.NORMAL, text="Отправить")
        self.feedback_cancel_button.config(state=tk.NORMAL)
        self.feedback_panel.pack(fill=tk.X, pady=(12, 0))
        self.feedback_expected_entry.focus_set()

    def close_feedback_panel(self) -> None:
        self.feedback_expected_entry.config(state=tk.NORMAL)
        self.feedback_send_button.config(state=tk.NORMAL, text="Отправить")
        self.feedback_cancel_button.config(state=tk.NORMAL)
        self.feedback_status_label.config(text="", fg="#94a3b8")
        self.feedback_panel.pack_forget()

    def submit_feedback_from_panel(self) -> None:
        if self.current_session is None:
            return
        expected_text = self.feedback_expected_var.get().strip()
        if not expected_text:
            messagebox.showwarning("Нужны данные", "Введите корректный результат, который должен был вернуть сервис.")
            return
        self.feedback_status_label.config(text="Отправляем обратную связь...", fg="#94a3b8")
        self.feedback_expected_entry.config(state=tk.DISABLED)
        self.feedback_send_button.config(state=tk.DISABLED, text="Отправляем...")
        self.feedback_cancel_button.config(state=tk.DISABLED)
        self.feedback_button.config(state=tk.DISABLED, text="Отправляем feedback...")
        worker = threading.Thread(
            target=self._feedback_worker,
            args=(self.current_session, expected_text),
            daemon=True,
        )
        worker.start()

    def _feedback_worker(self, session: RecognitionSession, expected_text: str) -> None:
        mode_cfg = MODES[session.mode]
        feedback_endpoint = mode_cfg["feedback_endpoint"]
        recognized_text = str(session.response_data.get("result") or "")
        if recognized_text == "NOT_FOUND":
            recognized_text = str(session.response_data.get("raw_text") or "")

        payload_json = {
            "expected_text": expected_text,
            "recognized_text": recognized_text,
            "source_endpoint": mode_cfg["recognize_endpoint"],
            "service_response": session.response_data,
        }

        try:
            with session.image_path.open("rb") as fh:
                response = requests.post(
                    f"{session.api_base_url}{feedback_endpoint}",
                    files={"image": (session.image_path.name, fh, guess_mime(session.image_path))},
                    data={
                        "expected_text": expected_text,
                        "recognized_text": recognized_text,
                        "endpoint": mode_cfg["recognize_endpoint"],
                        "payload": json.dumps(payload_json, ensure_ascii=False),
                    },
                    timeout=float(self.timeout_var.get().strip() or DEFAULT_TIMEOUT_SEC),
                )
            response.raise_for_status()
            self.worker_queue.put(("feedback_ok", response.json()))
        except Exception as exc:
            self.worker_queue.put(("feedback_error", str(exc)))

    def _on_feedback_success(self, response_data: dict[str, Any]) -> None:
        self.feedback_button.config(state=tk.NORMAL, text="Отправить обратную связь")
        self.feedback_expected_var.set("")
        self.feedback_service_label.config(text="")
        self.feedback_status_label.config(text="Обратная связь отправлена.", fg="#86efac")
        self.status_line.config(text="Обратная связь отправлена.")
        self.close_feedback_panel()

    def _on_feedback_error(self, error_text: str) -> None:
        self.feedback_button.config(state=tk.NORMAL, text="Отправить обратную связь")
        self.feedback_expected_entry.config(state=tk.NORMAL)
        self.feedback_send_button.config(state=tk.NORMAL, text="Отправить")
        self.feedback_cancel_button.config(state=tk.NORMAL)
        self.feedback_status_label.config(text=f"Не удалось отправить обратную связь: {error_text}", fg="#fca5a5")
        self.status_line.config(text="Не удалось отправить обратную связь.")

    def save_settings(self) -> None:
        api_base_url = self.api_base_var.get().strip().rstrip("/")
        if not api_base_url:
            messagebox.showwarning("Неверные настройки", "URL сервиса не должен быть пустым.")
            return
        try:
            timeout_sec = float(self.timeout_var.get().strip())
        except ValueError:
            messagebox.showwarning("Неверные настройки", "Таймаут должен быть числом.")
            return

        self.config_data = AppConfig(api_base_url=api_base_url, timeout_sec=timeout_sec)
        save_app_config(self.config_data)
        self.settings_status_label.config(text=f"Настройки сохранены в {CONFIG_PATH.name}.", fg="#86efac")

    def reset_settings(self) -> None:
        self.config_data = AppConfig()
        self.api_base_var.set(self.config_data.api_base_url)
        self.timeout_var.set(str(int(self.config_data.timeout_sec)))
        save_app_config(self.config_data)
        self.settings_status_label.config(text="Настройки сброшены к значениям по умолчанию.", fg="#fcd34d")

    def _set_image_picker_enabled(self, enabled: bool) -> None:
        cursor = "hand2" if enabled else "watch"
        bg = "#0f172a" if enabled else "#172033"
        self.drop_frame.config(cursor=cursor, bg=bg)
        self.preview_label.config(cursor=cursor, bg=bg)
        self.preview_hint_label.config(cursor=cursor, bg=bg)

    def _toggle_detail_view(self) -> None:
        self.detail_view_var.set(not self.detail_view_var.get())
        self._render_detail_toggle()
        self._update_detail_visibility()

    def _render_detail_toggle(self) -> None:
        self.toggle_canvas.delete("all")
        enabled = self.detail_view_var.get()
        track_color = self.colors["toggle_on"] if enabled else self.colors["toggle_off"]
        x1, y1, x2, y2 = 4, 6, 60, 30
        radius = (y2 - y1) / 2
        self.toggle_canvas.create_rectangle(x1 + radius, y1, x2 - radius, y2, outline="", fill=track_color)
        self.toggle_canvas.create_oval(x1, y1, x1 + 2 * radius, y2, outline="", fill=track_color)
        self.toggle_canvas.create_oval(x2 - 2 * radius, y1, x2, y2, outline="", fill=track_color)
        knob_radius = 9
        knob_center_x = x2 - radius if enabled else x1 + radius
        knob_center_y = (y1 + y2) / 2
        self.toggle_canvas.create_oval(
            knob_center_x - knob_radius,
            knob_center_y - knob_radius,
            knob_center_x + knob_radius,
            knob_center_y + knob_radius,
            outline="",
            fill="#f8fafc",
        )

    def _update_detail_visibility(self) -> None:
        if self.detail_view_var.get():
            self.detail_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.detail_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        else:
            self.detail_text.pack_forget()
            self.detail_scroll.pack_forget()


def main() -> None:
    root = tk.Tk()
    app = DesktopClientApp(root)
    _ = app
    root.mainloop()


if __name__ == "__main__":
    main()
