from __future__ import annotations

from pathlib import Path
import tkinter as tk

import cv2
from PIL import Image, ImageChops, ImageTk

from .config import STEP_TITLES, image_path
from .state_machine import AppState, SessionStateMachine


WHITE = "#FFFFFF"
FOREST = "#064E3B"
TEAL = "#10B58B"
MINT = "#DDF7EC"
SKY = "#DFF6FF"
BLUE = "#49BEE8"
CORAL = "#FF6B5F"
GREEN = "#2DBE63"
PANEL = "#F2F5F5"
MUTED = "#5E786F"


def _contain(image: Image.Image, width: int, height: int) -> Image.Image:
    result = image.copy()
    result.thumbnail((max(1, width), max(1, height)), Image.Resampling.LANCZOS)
    return result


def _trim_white(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    flat = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    difference = ImageChops.difference(rgba, flat).convert("L")
    bbox = difference.point(lambda value: 255 if value > 12 else 0).getbbox()
    return rgba.crop(bbox) if bbox else rgba


class LaiaView(tk.Canvas):
    def __init__(self, master: tk.Misc, machine: SessionStateMachine, **kwargs) -> None:
        super().__init__(master, background=WHITE, highlightthickness=0, **kwargs)
        self.machine = machine
        self.frame_bgr = None
        self.camera_detail = ""
        self.hardware_notice = ""
        self._source_images: dict[Path, Image.Image] = {}
        self._photo_cache: dict[tuple[Path, int, int], ImageTk.PhotoImage] = {}
        self._photos: list[ImageTk.PhotoImage] = []
        self._camera_item = None
        self._camera_box = None
        self._camera_photo = None
        self.bind("<Configure>", lambda _event: self.render())

    def set_frame(self, frame_bgr) -> None:
        self.frame_bgr = frame_bgr
        self._update_camera_image()

    def set_camera_detail(self, detail: str) -> None:
        self.camera_detail = detail

    def render(self) -> None:
        # El DSI 800x480 entrega 480x800 rotado; no forzar ancho de escritorio.
        # Hacerlo recortaría la interfaz vertical.
        width = max(320, self.winfo_width())
        height = max(480, self.winfo_height())
        self.delete("all")
        self._photos.clear()
        self._camera_item = None
        self._camera_box = None
        self._camera_photo = None
        self.configure(background=WHITE)
        self._decorations(width, height)
        self._logo(width, height)

        if self.machine.state is AppState.WAITING_FOR_HANDS:
            self._waiting(width, height)
        elif self.machine.state is AppState.INCOMPLETE:
            self._incomplete(width, height)
        elif self.machine.state in {AppState.WASHING, AppState.CORRECTION}:
            self._tutorial(width, height)
        else:
            self._success(width, height)

    def _font(self, width: int, size: int, weight: str = "normal"):
        if width <= 600:
            scale = max(0.82, min(1.12, width / 480))
        else:
            scale = max(0.72, min(1.45, width / 1024))
        return ("DejaVu Sans", max(10, int(size * scale)), weight)

    def _decorations(self, width: int, height: int) -> None:
        for x, y, radius in (
            (0.055, 0.10, 15),
            (0.92, 0.18, 10),
            (0.10, 0.72, 11),
            (0.88, 0.78, 18),
        ):
            cx, cy = width * x, height * y
            self.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill=SKY, outline="#BDEBFA", width=2)
            self.create_oval(cx - radius * 0.45, cy - radius * 0.55, cx, cy - radius * 0.05, fill=WHITE, outline="")
        self._star(width * 0.82, height * 0.13, 11, MINT)
        self._star(width * 0.13, height * 0.84, 8, SKY)

    def _star(self, x: float, y: float, radius: float, color: str) -> None:
        points = [x, y - radius, x + radius * 0.26, y - radius * 0.26, x + radius, y,
                  x + radius * 0.26, y + radius * 0.26, x, y + radius,
                  x - radius * 0.26, y + radius * 0.26, x - radius, y,
                  x - radius * 0.26, y - radius * 0.26]
        self.create_polygon(points, fill=color, outline="")

    def _load(self, path: Path) -> Image.Image:
        if path not in self._source_images:
            image = Image.open(path).convert("RGBA")
            self._source_images[path] = _trim_white(image) if path.name == "LOGO_LAIA-1.png" else image
        return self._source_images[path]

    def _place_image(self, path: Path, x: float, y: float, max_width: int, max_height: int) -> None:
        key = (path, max_width, max_height)
        photo = self._photo_cache.get(key)
        if photo is None:
            image = _contain(self._load(path), max_width, max_height)
            photo = ImageTk.PhotoImage(image)
            self._photo_cache[key] = photo
        self.create_image(x, y, image=photo, anchor="center")

    def _update_camera_image(self) -> None:
        if self.frame_bgr is None or self._camera_item is None or self._camera_box is None:
            return
        left, top, right, bottom = self._camera_box
        rgb = cv2.cvtColor(self.frame_bgr, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(rgb)
        frame.thumbnail((int(right - left), int(bottom - top)), Image.Resampling.LANCZOS)
        self._camera_photo = ImageTk.PhotoImage(frame)
        self.itemconfigure(self._camera_item, image=self._camera_photo)

    def _logo(self, width: int, height: int) -> None:
        if height > width:
            self._place_image(image_path("logo"), width / 2, height * 0.06, int(width * 0.42), int(height * 0.10))
        else:
            self._place_image(image_path("logo"), width / 2, height * 0.07, int(width * 0.22), int(height * 0.105))

    def _waiting(self, width: int, height: int) -> None:
        portrait = height > width
        title_y = 0.17 if portrait else 0.205
        self.create_text(width / 2, height * title_y, text="Acerca tus manos", fill=FOREST,
                         font=self._font(width, 30, "bold"), width=width * 0.82, justify="center")
        if portrait:
            left, top, right, bottom = width * 0.07, height * 0.23, width * 0.93, height * 0.88
        else:
            left, top, right, bottom = width * 0.11, height * 0.28, width * 0.89, height * 0.88
        self.create_rectangle(left, top, right, bottom, fill=PANEL, outline=TEAL, width=max(2, int(width / 420)))

        if self.frame_bgr is not None:
            self._camera_box = (left, top, right, bottom)
            self._camera_item = self.create_image((left + right) / 2, (top + bottom) / 2, anchor="center")
            self._update_camera_image()
        else:
            message = "Esperando cámara" if not self.camera_detail else "Cámara no disponible"
            self.create_text((left + right) / 2, bottom - height * 0.06, text=message,
                             fill=MUTED, font=self._font(width, 14, "bold"))

        self._place_image(
            image_path("waiting"),
            (left + right) / 2,
            (top + bottom) / 2 - height * 0.02,
            int((right - left) * (0.68 if portrait else 0.52)),
            int((bottom - top) * (0.52 if portrait else 0.70)),
        )

    def _tutorial(self, width: int, height: int) -> None:
        portrait = height > width
        first_y = 0.16 if portrait else 0.205
        second_y = 0.215 if portrait else 0.275
        correction = self.machine.state is AppState.CORRECTION
        step = self.machine.expected_step
        if correction:
            self.create_text(width / 2, height * first_y, text="Casi, casi", fill=CORAL,
                             font=self._font(width, 25, "bold"))
            self.create_text(width / 2, height * second_y, text=f"Haz el paso {step}", fill=FOREST,
                             font=self._font(width, 29, "bold"))
        else:
            self.create_text(width / 2, height * first_y, text=f"Paso {step} de 6", fill=TEAL,
                             font=self._font(width, 20, "bold"))
            self.create_text(width / 2, height * second_y, text=STEP_TITLES[step], fill=FOREST,
                             font=self._font(width, 28, "bold"), width=width * 0.86, justify="center")

        self._place_image(
            self.machine.asset,
            width / 2,
            height * (0.56 if portrait else 0.58),
            int(width * (0.76 if portrait else 0.46)),
            int(height * (0.47 if portrait else 0.50)),
        )
        self._progress(width, height, step, CORAL if correction else TEAL)

    def _progress(self, width: int, height: int, current: int, active_color: str) -> None:
        portrait = height > width
        start_x = width * (0.10 if portrait else 0.25)
        end_x, y = width * (0.90 if portrait else 0.75), height * 0.91
        self.create_line(start_x, y, end_x, y, fill=TEAL, width=max(2, int(width / 400)))
        gap = (end_x - start_x) / 5
        radius = max(11, int(width * (0.032 if portrait else 0.017)))
        for step in range(1, 7):
            x = start_x + gap * (step - 1)
            fill = active_color if step == current else (MINT if step < current else WHITE)
            self.create_oval(x - radius, y - radius, x + radius, y + radius,
                             fill=fill, outline=TEAL, width=2)
            self.create_text(x, y, text=str(step), fill=WHITE if step == current else FOREST,
                             font=self._font(width, 12, "bold"))

    def _incomplete(self, width: int, height: int) -> None:
        portrait = height > width
        self.create_text(
            width / 2,
            height * (0.19 if portrait else 0.23),
            text="¡Casi!",
            fill=CORAL,
            font=self._font(width, 31, "bold"),
        )
        self.create_text(
            width / 2,
            height * (0.285 if portrait else 0.34),
            text="Nos faltaron algunos movimientos.",
            fill=FOREST,
            font=self._font(width, 22, "bold"),
            width=width * 0.84,
            justify="center",
        )
        self.create_text(
            width / 2,
            height * (0.37 if portrait else 0.43),
            text="¿Lo intentamos otra vez?",
            fill=TEAL,
            font=self._font(width, 20, "bold"),
            width=width * 0.84,
            justify="center",
        )
        self._place_image(
            image_path("waiting"),
            width / 2,
            height * (0.63 if portrait else 0.69),
            int(width * (0.58 if portrait else 0.34)),
            int(height * (0.40 if portrait else 0.43)),
        )
        self.create_text(
            width / 2,
            height * (0.90 if portrait else 0.93),
            text="Acerca tus manos para comenzar de nuevo",
            fill=MUTED,
            font=self._font(width, 15, "bold"),
            width=width * 0.86,
            justify="center",
        )

    def _success(self, width: int, height: int) -> None:
        portrait = height > width
        check_y = height * (0.145 if portrait else 0.185)
        radius = 18 if portrait else 20
        self.create_oval(width / 2 - radius, check_y - radius, width / 2 + radius, check_y + radius,
                         fill=GREEN, outline="")
        self.create_line(width / 2 - 9, check_y, width / 2 - 2, check_y + 8,
                         width / 2 + 11, check_y - 8, fill=WHITE, width=4, joinstyle="round")
        self.create_text(width / 2, height * (0.205 if portrait else 0.265), text="¡Yupi!", fill=GREEN,
                         font=self._font(width, 31, "bold"))
        self.create_text(width / 2, height * (0.265 if portrait else 0.34), text="¡Muy bien!", fill=FOREST,
                         font=self._font(width, 27, "bold"))
        self.create_text(width / 2, height * (0.315 if portrait else 0.405), text="Lavado completado", fill=TEAL,
                         font=self._font(width, 18, "bold"))
        self.create_text(width / 2, height * (0.355 if portrait else 0.46), text="Tus manos están limpias", fill=FOREST,
                         font=self._font(width, 16, "bold"), width=width * 0.86, justify="center")
        self._place_image(
            image_path("success"),
            width / 2,
            height * (0.61 if portrait else 0.69),
            int(width * (0.76 if portrait else 0.42)),
            int(height * (0.40 if portrait else 0.39)),
        )
        self.create_text(width / 2, height * (0.90 if portrait else 0.94),
                         text="Gracias por cuidar tu salud", fill=GREEN,
                         font=self._font(width, 16, "bold"), width=width * 0.86, justify="center")
