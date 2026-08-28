from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
IMAGES = ASSETS / "images"
AUDIO = ASSETS / "audio"
DEPLOYMENT = ROOT / "deployment" / "laia_model"
LOGS = ROOT / "logs"


IMAGE_FILES = {
    "logo": "LOGO_LAIA-1.png",
    "waiting": "silueta.png",
    "step_1": "Paso1.png",
    "step_2": "Paso2.png",
    "step_3": "Paso3.png",
    "step_4": "Paso4.png",
    "step_5": "Paso5.png",
    "step_6": "Paso6.png",
    "success": "manos_limpias.png",
}

AUDIO_FILES = {
    "welcome": "Bienvenido.mp3",
    "almost": "casi_bien.mp3",
    "recover": "ya_va_bien.mp3",
    "start": "comencemos.mp3",
    "retry": "ups_intentar_otravez.mp3",
    "hands_lost": "ya_no_veo_manos_acercalas.mp3",
    "success": "manos_limpias.mp3",
}

STEP_TITLES = {
    1: "Frota palma con palma",
    2: "Limpia el dorso de tus manos",
    3: "Entrelaza tus dedos",
    4: "Frota el dorso de tus dedos",
    5: "Limpia tus pulgares",
    6: "Frota las puntas de tus dedos",
}


@dataclass(frozen=True)
class RuntimeConfig:
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: float = 20.0
    inference_threads: int = 4
    # Cableado de señalización: acierto -> BCM22; error/corrección -> BCM17.
    # Estos son los dos valores que hay que cambiar si se modifica el cableado.
    success_gpio_bcm: int = 22
    error_gpio_bcm: int = 17
    gpio_blink_count: int = 3
    gpio_blink_seconds: float = 0.20
    gpio_hold_seconds: float = 3.0
    correct_predictions_required: int = 2
    incorrect_predictions_required: int = 2
    hands_lost_seconds: float = 3.0
    # Un lavado nunca puede aprobarse antes de este tiempo continuo con las
    # manos visibles, aunque el modelo haya encadenado varias predicciones.
    minimum_washing_seconds: float = 20.0
    # Compatibilidad con configuraciones antiguas. Estos valores pueden
    # conservarse como referencia/diagnóstico, pero nunca autorizan SUCCESS.
    minimum_steps_for_success: int = 4
    minimum_step_coverage: float = 0.80
    success_hold_seconds: float = 8.0
    incomplete_hold_seconds: float = 5.0


def image_path(key: str) -> Path:
    return IMAGES / IMAGE_FILES[key]


def audio_path(key: str) -> Path:
    return AUDIO / AUDIO_FILES[key]


def tutorial_for_step(step: int) -> Path:
    if step not in range(1, 7):
        raise ValueError(f"paso fuera de rango: {step}")
    return image_path(f"step_{step}")


def expected_asset(state: str, expected_step: int) -> Path:
    """Contrato visual único; CORRECTION depende del paso esperado."""
    if state == "WAITING_FOR_HANDS":
        return image_path("waiting")
    if state in {"WASHING", "CORRECTION"}:
        return tutorial_for_step(expected_step)
    if state == "SUCCESS":
        return image_path("success")
    if state == "INCOMPLETE":
        return image_path("waiting")
    raise ValueError(f"estado desconocido: {state}")


def required_paths() -> list[Path]:
    model_files = [
        DEPLOYMENT / "model" / "laia_lightstgcnv2_fp32.onnx",
        DEPLOYMENT / "model" / "hand_landmarker.task",
    ]
    return (
        [IMAGES / filename for filename in IMAGE_FILES.values()]
        + [AUDIO / filename for filename in AUDIO_FILES.values()]
        + model_files
    )

