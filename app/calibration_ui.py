"""Vista Tkinter de calibración, visualmente alineada con la app de LAIA."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk

import cv2
from PIL import Image, ImageTk

from .calibration_config import CalibrationState
from .config import image_path
from .ui import BLUE, CORAL, FOREST, GREEN, MINT, MUTED, PANEL, SKY, TEAL, WHITE, _contain, _trim_white


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


class CalibrationView(tk.Canvas):
    """Canvas con video, guía suave y landmarks pequeños; sin overlays técnicos."""

    def __init__(self, master: tk.Misc, **kwargs) -> None:
        super().__init__(master, background=WHITE, highlightthickness=0, **kwargs)
        self.frame_bgr = None
        self.hands: tuple[tuple[tuple[float, float], ...], ...] = ()
        self.hand_valid: tuple[bool, ...] = ()
        self.state = CalibrationState.POSITIONING
        self.message = "Buscando tus manos…"
        self.countdown: int | None = None
        self.camera_message = ""

        self._source_images: dict[Path, Image.Image] = {}
        self._photo_cache: dict[tuple[Path, int, int], ImageTk.PhotoImage] = {}
        self._camera_item = None
        self._camera_box: tuple[float, float, float, float] | None = None
        self._video_geometry: tuple[float, float, float, float] | None = None
        self._camera_photo = None
        self._landmark_items: list[int] = []
        self._guide_item = None
        self._last_content_key = None
        self.bind("<Configure>", lambda _event: self.render())

    def set_frame(self, frame_bgr) -> None:
        self.frame_bgr = frame_bgr
        self._update_camera_image()
        self._draw_landmarks()

    def set_landmarks(
        self,
        hands: tuple[tuple[tuple[float, float], ...], ...],
        hand_valid: tuple[bool, ...],
    ) -> None:
        self.hands = hands
        self.hand_valid = hand_valid
        self._draw_landmarks()

    def set_feedback(
        self,
        state: CalibrationState,
        message: str,
        countdown: int | None,
        camera_message: str = "",
    ) -> None:
        key = (state, message, countdown, camera_message)
        self.state = state
        self.message = message
        self.countdown = countdown
        self.camera_message = camera_message
        if key != self._last_content_key:
            self._last_content_key = key
            self.render()

    def render(self) -> None:
        width = max(320, self.winfo_width())
        height = max(480, self.winfo_height())
        self.delete("all")
        self._landmark_items.clear()
        self._camera_item = None
        self._camera_box = None
        self._video_geometry = None
        self._guide_item = None
        self.configure(background=WHITE)
        self._decorations(width, height)
        self._logo(width, height)

        portrait = height > width
        self._header(width, height, portrait)
        if portrait:
            box = (width * 0.065, height * 0.235, width * 0.935, height * 0.775)
        else:
            box = (width * 0.10, height * 0.30, width * 0.90, height * 0.83)
        self._camera_box = box
        left, top, right, bottom = box
        self.create_rectangle(left, top, right, bottom, fill=PANEL, outline="#BFEBDD", width=2)
        self._camera_item = self.create_image((left + right) / 2, (top + bottom) / 2, anchor="center")
        self._update_camera_image()

        # La silueta es una guía amistosa, no la ROI matemática.
        guide_w = int((right - left) * (0.70 if portrait else 0.52))
        guide_h = int((bottom - top) * (0.64 if portrait else 0.72))
        self._place_image(image_path("waiting"), (left + right) / 2, (top + bottom) / 2, guide_w, guide_h)
        self._draw_guide_halo(left, top, right, bottom)
        self._draw_landmarks()
        self._draw_state_overlay(width, height, portrait)

        if self.camera_message:
            self.create_text(
                (left + right) / 2,
                bottom - height * 0.035,
                text=self.camera_message,
                fill=CORAL,
                font=self._font(width, 13, "bold"),
            )
        self._footer(width, height, portrait)

    def _font(self, width: int, size: int, weight: str = "normal"):
        if width <= 600:
            scale = max(0.82, min(1.12, width / 480))
        else:
            scale = max(0.72, min(1.45, width / 1024))
        return ("DejaVu Sans", max(10, int(size * scale)), weight)

    def _decorations(self, width: int, height: int) -> None:
        for x, y, radius in ((0.055, 0.10, 15), (0.92, 0.18, 10), (0.10, 0.72, 11), (0.88, 0.78, 18)):
            cx, cy = width * x, height * y
            self.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill=SKY, outline="#BDEBFA", width=2)
            self.create_oval(cx - radius * 0.45, cy - radius * 0.55, cx, cy - radius * 0.05, fill=WHITE, outline="")
        self._star(width * 0.82, height * 0.13, 11, MINT)
        self._star(width * 0.13, height * 0.84, 8, SKY)

    def _star(self, x: float, y: float, radius: float, color: str) -> None:
        points = [
            x, y - radius, x + radius * 0.26, y - radius * 0.26,
            x + radius, y, x + radius * 0.26, y + radius * 0.26,
            x, y + radius, x - radius * 0.26, y + radius * 0.26,
            x - radius, y, x - radius * 0.26, y - radius * 0.26,
        ]
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
            photo = ImageTk.PhotoImage(_contain(self._load(path), max_width, max_height))
            self._photo_cache[key] = photo
        self.create_image(x, y, image=photo, anchor="center")

    def _logo(self, width: int, height: int) -> None:
        if height > width:
            self._place_image(image_path("logo"), width / 2, height * 0.06, int(width * 0.42), int(height * 0.10))
        else:
            self._place_image(image_path("logo"), width / 2, height * 0.07, int(width * 0.22), int(height * 0.105))

    def _header(self, width: int, height: int, portrait: bool) -> None:
        if self.state is CalibrationState.CALIBRATION_OK:
            title, color = "¡Perfecto!", GREEN
            subtitle = "Así colocaremos las manos"
        elif self.state is CalibrationState.COUNTDOWN:
            title, color = "¡Listo!", TEAL
            subtitle = "Mantén tus manos aquí"
        elif self.state is CalibrationState.READY:
            title, color = "¡Muy bien!", TEAL
            subtitle = "Mantén tus manos aquí"
        else:
            title, color = "Coloca tus manos aquí", FOREST
            subtitle = self.message
        first_y = 0.155 if portrait else 0.205
        second_y = 0.205 if portrait else 0.265
        self.create_text(width / 2, height * first_y, text=title, fill=color, font=self._font(width, 28, "bold"), width=width * 0.84)
        self.create_text(width / 2, height * second_y, text=subtitle, fill=FOREST, font=self._font(width, 17, "bold"), width=width * 0.86)

    def _draw_guide_halo(self, left: float, top: float, right: float, bottom: float) -> None:
        color = GREEN if self.state in {CalibrationState.READY, CalibrationState.COUNTDOWN, CalibrationState.CALIBRATION_OK} else TEAL
        inset_x = (right - left) * 0.16
        inset_y = (bottom - top) * 0.14
        self.create_oval(left + inset_x, top + inset_y, right - inset_x, bottom - inset_y, outline=color, width=2)

    def _video_point(self, x: float, y: float) -> tuple[float, float] | None:
        if self._video_geometry is None:
            return None
        left, top, width, height = self._video_geometry
        return left + x * width, top + y * height

    def _draw_landmarks(self) -> None:
        for item in self._landmark_items:
            self.delete(item)
        self._landmark_items.clear()
        for index, hand in enumerate(self.hands):
            valid = index < len(self.hand_valid) and self.hand_valid[index]
            color = GREEN if valid else BLUE
            points = [self._video_point(x, y) for x, y in hand]
            if len(points) != 21 or any(point is None for point in points):
                continue
            concrete = [point for point in points if point is not None]
            for start, end in HAND_CONNECTIONS:
                self._landmark_items.append(self.create_line(*concrete[start], *concrete[end], fill=color, width=1.5, capstyle="round", joinstyle="round"))
            for x, y in concrete:
                radius = 2.4
                self._landmark_items.append(self.create_oval(x - radius, y - radius, x + radius, y + radius, fill=WHITE, outline=color, width=1))

    def _update_camera_image(self) -> None:
        if self.frame_bgr is None or self._camera_item is None or self._camera_box is None:
            return
        left, top, right, bottom = self._camera_box
        frame_height, frame_width = self.frame_bgr.shape[:2]
        box_width, box_height = right - left, bottom - top
        scale = min(box_width / frame_width, box_height / frame_height)
        display_width = max(1, int(frame_width * scale))
        display_height = max(1, int(frame_height * scale))
        rgb = cv2.cvtColor(self.frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
        actual_width, actual_height = image.size
        self._video_geometry = (
            left + (box_width - actual_width) / 2,
            top + (box_height - actual_height) / 2,
            actual_width,
            actual_height,
        )
        self._camera_photo = ImageTk.PhotoImage(image)
        self.itemconfigure(self._camera_item, image=self._camera_photo)

    def _footer(self, width: int, height: int, portrait: bool) -> None:
        if self.state is CalibrationState.CALIBRATION_OK:
            text = "Puedes repetir la calibración cuando quieras"
        elif self.state is CalibrationState.COUNTDOWN:
            text = "No muevas tus manos"
        else:
            text = "Las dos manos, juntas y dentro de la guía"
        self.create_text(width / 2, height * (0.885 if portrait else 0.91), text=text, fill=MUTED, font=self._font(width, 14, "bold"), width=width * 0.88, justify="center")

    def _draw_state_overlay(self, width: int, height: int, portrait: bool) -> None:
        """Dibuja el número/check encima del video, nunca debajo del panel."""

        if self.state is CalibrationState.COUNTDOWN and self.countdown is not None:
            self.create_text(
                width / 2,
                height * (0.52 if portrait else 0.56),
                text=str(self.countdown),
                fill=GREEN,
                font=self._font(width, 82, "bold"),
            )
        elif self.state is CalibrationState.CALIBRATION_OK:
            center_y = height * (0.29 if portrait else 0.37)
            self.create_oval(width / 2 - 18, center_y - 18, width / 2 + 18, center_y + 18, fill=GREEN, outline="")
            self.create_line(width / 2 - 9, center_y, width / 2 - 2, center_y + 8, width / 2 + 11, center_y - 8, fill=WHITE, width=4, joinstyle="round")
