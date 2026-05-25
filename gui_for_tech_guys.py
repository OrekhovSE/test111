#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import os
import sys
import threading
import queue
from dataclasses import asdict
import json
import pprint

try:
    from test import get_img_eval_stat, IMAGE_EXTS, EvalResult
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print(f"Убедитесь, что файл test.py находится в той же папке, что и этот скрипт")
    sys.exit(1)


class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title = "Распознование номеров пломб"
        self.root.geometry("1000x600")
        self.root.configure(bg="#1e3a8a")
        self.current_image_path = None
        self.current_result = None
        self.center_window()
        self.setup_ui()


    def center_window(self):
        self.root.update_idletasks()
        width = 1000
        height = 600
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')


    def setup_ui(self):
        self.colors = {
                "bg_dark": "#1e3a8a",
                "bg_light": "#3b82f6",
                "text": "#ffffff",
                "text_secondary": "#93c5fd",
                "frame_bg": "#0f172a",
                "button_bg": "#eff6ff",
                "button_fg": "#1e3a8a",
                "success": "#10b981",
                "error": "#ef4444"
        }

        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=0)
        self.root.grid_columnconfigure(0, weight=1)

        title_label = tk.Label(
                self.root,
                text="Распознование номеров пломб",
                font=("Arial", 20, "bold"),
                bg=self.colors["bg_dark"],
                fg=self.colors["text"],
                pady=20
        )
        title_label.grid(row=0, column=0, sticky="ew")

        top_frame = tk.Frame(self.root, bg=self.colors["bg_dark"])
        top_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)

        top_frame.grid_columnconfigure(0, weight=1)
        top_frame.grid_columnconfigure(1, weight=1)
        top_frame.grid_rowconfigure(0, weight=1)

        self.left_frame = tk.Frame(top_frame, bg=self.colors["frame_bg"], relief=tk.RAISED, bd=2)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.left_frame.grid_propagate(False)
        self.left_frame.config(width=400)

        text_frame = tk.Frame(self.left_frame, bg=self.colors["frame_bg"])
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.result_text = tk.Text(
                text_frame,
                font=("Courier", 10),
                bg=self.colors["frame_bg"],
                fg=self.colors["text_secondary"],
                wrap=tk.WORD,
                state=tk.DISABLED
        )
        scrollbar = tk.Scrollbar(text_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.right_frame = tk.Frame(top_frame, bg=self.colors["frame_bg"], relief=tk.RAISED, bd=2)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.right_frame.grid_propagate(False)
        self.right_frame.config(width=400)

        self.photo_label = tk.Label(
                self.right_frame,
                bg=self.colors["frame_bg"],
                fg=self.colors["text"],
                text="Фото не выбрано",
                font=("Arial", 16, "bold")
        )
        self.photo_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)


        buttons_frame = tk.Frame(self.root, bg=self.colors["bg_dark"])
        buttons_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=10)

        buttons_frame.grid_columnconfigure(0, weight=1)
        buttons_frame.grid_columnconfigure(1, weight=1)
        buttons_frame.grid_columnconfigure(2, weight=1)
        buttons_frame.grid_columnconfigure(3, weight=1)

        btn_next = tk.Button(
                buttons_frame,
                text="Далее",
                font=("Arial", 10, "bold"),
                bg=self.colors["button_bg"],
                fg=self.colors["button_fg"],
                activebackground="#dbeafe",
                cursor="hand2",
                padx=20,
                pady=5,
                command=self.clear_all
        )
        btn_next.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.btn_report = tk.Button(
                buttons_frame,
                text="Сообщить о неверном результате",
                font=("Arial", 10, "bold"),
                bg=self.colors["button_bg"],
                fg=self.colors["button_fg"],
                activebackground="#dc2626",
                cursor="hand2",
                padx=20,
                pady=5,
                command=self.report_error
        )
        self.btn_report.grid(row=0, column=1, sticky="ew", padx=(5, 5))

        self.btn_select = tk.Button(
                buttons_frame,
                text="Выбрать фото",
                font=("Arial", 10, "bold"),
                bg=self.colors["button_bg"],
                fg=self.colors["button_fg"],
                activebackground="#dbeafe",
                cursor="hand2",
                padx=20,
                pady=5,
                command=self.select_file
        )
        self.btn_select.grid(row=0, column=2, sticky="ew", padx=(5, 5))

        self.btn_recognize = tk.Button(
                buttons_frame,
                text="Распознать",
                font=("Arial", 10, "bold"),
                bg=self.colors["button_bg"],
                fg=self.colors["button_fg"],
                activebackground="#dbeafe",
                cursor="hand2",
                padx=20,
                pady=5,
                command=self.recognize_file
        )
        self.btn_recognize.grid(row=0, column=3, sticky="ew", padx=(5, 0))


    def recognize_file(self):
        if not self.current_image_path:
            messagebox.showwarning("Предупреждение", "Сначала выберите файл")
            return

        self.btn_recognize.config(state=tk.DISABLED, text="Распознование..")

        try:
            result = get_img_eval_stat(self.current_image_path)
            self.current_result = result

            import json
            result_dict = {
                    "image": str(result.image),
                    "status": result.status,
                    "expected": result.expected,
                    "best_got": result.best_got,
                    "reason": result.reason,
                    "processed": result.processed,
                    "total_results": result.total_results,
                    "matched_index": result.matched_index,
                    "payload": result.payload,
                    "time": result.time
            }
            output = json.dumps(result_dict, indent=2, ensure_ascii=False)

            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, output)
            self.result_text.config(state=tk.DISABLED)

            self.show_match_plate(result.expected, result.best_got)

        except Exception as e:
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Ошибка: {str(e)}")
            self.result_text.config(state=tk.DISABLED)

        finally:
            self.btn_recognize.config(state=tk.NORMAL, text="Распознать")


    def show_match_plate(self, expected, best_got):
        if hasattr(self, "match_plate"):
            self.match_plate.destroy()

        if expected is not None and best_got is not None:
            if str(expected) == str(best_got):
                text = "Совпадение"
                color = "#10b981"
            else:
                text = "Несовпадение. Сообщите нам"
                color = "#ef4444"

            self.match_plate = tk.Label(
                    self.left_frame,
                    text=text,
                    font=("Arial", 12, "bold"),
                    bg=color,
                    fg="white",
                    pady=5
            )
            self.match_plate.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)


    def clear_all(self):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)
        self.current_image_path = None
        self.current_result = None

        self.photo_label.config(image='', text="Файл не выбран")
        self.photo_label.image = None

        left_width = self.left_frame.winfo_width()
        self.right_frame.config(width=left_width)
        self.right_frame.grid_propagate(False)

        if hasattr(self, "match_plate"):
            self.match_plate.destroy()


    def report_error(self):
        if not self.current_result:
            messagebox.showwarning("Предупреждение", "Нет результатов для отправки")
            return

        if str(self.current_result.expected) == str(self.current_result.best_got):
            messagebox.showinfo("Информация", "Результат верный, нечего сообщать")
            return

        if not messagebox.askyesno("Подтверждение", "Отправить жалобу на неверное распознование?"):
            return

        self.btn_report.config(state=tk.DISABLED, text="Отправка...")
        self.root.update()

        try:
            import requests
            import json

            with open(self.current_image_path, "rb") as f:
                image_data = f.read()

            files = {
                    "image": (self.current_image_path.name, image_data, "image/jpeg"),
                    "expected": (None, str(self.current_result.expected)),
                    "best_got": (None, str(self.current_result.best_got)),
                    "reason": (None, str(self.current_result.reason)),
                    "payload": (None, json.dumps(self.current_result.payload))
            }

            from test import SERVER_IP, SERVICE_PORT
            response = requests.post(f"http://{SERVER_IP}:{SERVICE_PORT}/bad_recognize",
                                     files=files,
                                     timeout=30)

            if response.status_code == 200:
                messagebox.showinfo("Успех", "Жалоба отправлена успешно")
            else:
                messagebox.showerror("Ошибка", f"Сервер вернул код: {response.status_code}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось отправить: {str(e)}")
        finally:
            self.btn_report.config(state=tk.NORMAL, text="Сообщить о неверном результате"),


    def select_file(self):
        filetypes = []
        all_exts = " ".join([f"*{ext}" for ext in IMAGE_EXTS])
        filetypes.append(("Все изображения", all_exts))

        for ext in sorted(IMAGE_EXTS):
            ext_name = ext.upper().replace('.', '')
            filetypes.append((f"{ext_name} файлы", f"*{ext}"))

        filetypes.append(("Все файлы", "*.*"))

        file_path = filedialog.askopenfilename(
                title="Выберите изображение",
                initialdir=os.path.expanduser("~"),
                filetypes=filetypes
        )

        if not file_path:
            return

        self.current_image_path = Path(file_path)

        extension = self.current_image_path.suffix.lower()
        if extension not in IMAGE_EXTS:
            messagebox.showerr(
                    "Ошибка",
                    f"Тип файла не поддерживается!\n\n",
                    f"Расширение: {extension}\n",
                    f"Поддерживаемые расширения: {', '.join(sorted(IMAGE_EXTS))}"
            )
            return

        self.root.update_idletasks()

        try:
            from PIL import Image, ImageTk

            img = Image.open(self.current_image_path)

            label_width = self.photo_label.winfo_width()
            label_height = self.photo_label.winfo_height()

            if label_width <= 1:
                label_width = 350
                label_height = 350

            background = Image.new('RGB', (label_width, label_height), color="#0f172a")
            img.thumbnail((label_width, label_height), Image.Resampling.LANCZOS)

            x = (label_width - img.width) // 2
            y = (label_height - img.height) // 2

            background.paste(img, (x, y))

            photo = ImageTk.PhotoImage(img)
            self.photo_label.config(image=photo, text="")
            self.photo_label.image = photo

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть изображение:\n{str(e)}")


def main():
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
