from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import shutil
import subprocess


LOGGER = logging.getLogger(__name__)
ROTATIONS = ("normal", "90", "180", "270")


@dataclass(frozen=True)
class DisplayOutput:
    name: str
    transform: str
    width: int = 0
    height: int = 0

    @property
    def logical_size(self) -> tuple[int, int]:
        if self.transform in {"90", "270"}:
            return self.height, self.width
        return self.width, self.height


def parse_wlr_randr(output: str, preferred_output: str | None = None) -> DisplayOutput:
    candidates: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for raw_line in output.splitlines():
        if raw_line and not raw_line[0].isspace():
            name = raw_line.split(maxsplit=1)[0]
            current = {"name": name, "enabled": False, "transform": "normal", "width": 0, "height": 0}
            candidates.append(current)
            continue
        if current is None:
            continue
        line = raw_line.strip()
        if line == "Enabled: yes":
            current["enabled"] = True
        elif line.startswith("Transform:"):
            current["transform"] = line.split(":", 1)[1].strip()

        elif " px" in line and "current)" in line:
            size = line.split(maxsplit=1)[0]
            try:
                width, height = (int(value) for value in size.split("x", 1))
                current["width"], current["height"] = width, height
            except (ValueError, TypeError):
                pass
    enabled = [item for item in candidates if item["enabled"]]
    if preferred_output:
        enabled = [item for item in enabled if item["name"] == preferred_output]
    if not enabled:
        target = f" {preferred_output}" if preferred_output else ""
        raise RuntimeError(f"no hay un output Wayland activo{target}")
    selected = enabled[0]
    return DisplayOutput(
        str(selected["name"]),
        str(selected["transform"]),
        int(selected["width"]),
        int(selected["height"]),
    )


class DisplayRotationController:
    def __init__(self, output_name: str | None = None) -> None:
        self.output_name = output_name
        self.command = shutil.which("wlr-randr")

    @staticmethod
    def next_transform(current: str) -> str:
        try:
            index = ROTATIONS.index(current)
        except ValueError:
            return "normal"
        return ROTATIONS[(index + 1) % len(ROTATIONS)]

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        environment.setdefault("WAYLAND_DISPLAY", "wayland-0")
        return environment

    def current(self) -> DisplayOutput:
        if self.command is None:
            raise RuntimeError("wlr-randr no está instalado")
        completed = subprocess.run(
            [self.command],
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
            env=self._environment(),
        )
        return parse_wlr_randr(completed.stdout, self.output_name)

    def rotate_next(self) -> DisplayOutput:
        current = self.current()
        transform = self.next_transform(current.transform)
        completed = subprocess.run(
            [self.command, "--output", current.name, "--transform", transform],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
            env=self._environment(),
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(detail or f"no se pudo girar {current.name}")
        LOGGER.info("Display %s: %s -> %s", current.name, current.transform, transform)
        return DisplayOutput(current.name, transform, current.width, current.height)
