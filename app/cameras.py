from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraOption:
    source: str
    label: str


def _video_number(path: Path) -> int:
    try:
        return int(path.name.removeprefix("video"))
    except ValueError:
        return 9999


def discover_camera_options() -> list[CameraOption]:
    """Devuelve cámaras reales y evita los nodos internos de códec/ISP."""
    options = [CameraOption("auto", "Automática")]

    try:
        from picamera2 import Picamera2

        if Picamera2.global_camera_info():
            options.append(CameraOption("picamera2", "Cámara Raspberry Pi"))
    except Exception as exc:
        LOGGER.debug("No se pudo consultar Picamera2: %s", exc)

    for node in sorted(Path("/sys/class/video4linux").glob("video*"), key=_video_number):
        try:
            resolved = node.resolve()
            # Los nodos pisp/codec de la Raspberry no son cámaras seleccionables.
            if "/usb" not in str(resolved).lower():
                continue
            name_path = node / "name"
            name = name_path.read_text(encoding="utf-8").strip() if name_path.is_file() else "Cámara USB"
            device = f"/dev/{node.name}"
            options.append(CameraOption(f"v4l2:{device}", f"USB · {name} ({node.name})"))
        except OSError as exc:
            LOGGER.debug("No se pudo inspeccionar %s: %s", node, exc)

    return options


def v4l2_device(source: str, fallback_index: int = 0) -> str:
    if source.startswith("v4l2:"):
        device = source.split(":", 1)[1]
        if device.startswith("/dev/video"):
            return device
        raise ValueError(f"dispositivo V4L2 inválido: {device}")
    return f"/dev/video{fallback_index}"
