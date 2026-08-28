#!/usr/bin/env python3

import time
import math
from pathlib import Path

import cv2
import mediapipe as mp
from picamera2 import Picamera2
from gpiozero import LED


# ============================================================
# CONFIGURACIÓN
# ============================================================

WIDTH = 640
HEIGHT = 480

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "deployment"
    / "laia_model"
    / "model"
    / "hand_landmarker.task"
)

# LEDs BCM
GPIO_RED = 17
GPIO_YELLOW = 27
GPIO_GREEN = 22

# Zona donde queremos colocar las manos ANTES de iniciar.
# Coordenadas normalizadas: 0.0 - 1.0
#
#        x1    y1    x2    y2
ROI = (0.16, 0.14, 0.84, 0.86)

# Qué porcentaje de los landmarks de CADA mano
# debe estar dentro de la zona objetivo.
MIN_INSIDE_RATIO = 0.90

# Evita iniciar si las manos están demasiado lejos/pequeñas.
MIN_HAND_WIDTH = 0.08
MIN_HAND_HEIGHT = 0.10

# Evita iniciar si una mano llena prácticamente toda la cámara.
MAX_HAND_WIDTH = 0.60
MAX_HAND_HEIGHT = 0.75

# Deben permanecer correctamente posicionadas este tiempo
# antes de ponerse verde.
STABLE_SECONDS = 1.0

# Countdown posterior.
COUNTDOWN_SECONDS = 3.0

# Confianzas MediaPipe
MIN_DETECTION_CONFIDENCE = 0.50
MIN_PRESENCE_CONFIDENCE = 0.50
MIN_TRACKING_CONFIDENCE = 0.50


# ============================================================
# CONEXIONES DE LA MANO PARA DIBUJAR EL ESQUELETO
# ============================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),       # índice
    (5, 9), (9, 10), (10, 11), (11, 12),  # medio
    (9, 13), (13, 14), (14, 15), (15, 16),# anular
    (13, 17), (17, 18), (18, 19), (19, 20),# meñique
    (0, 17),
]


# ============================================================
# GPIO
# ============================================================

red = LED(GPIO_RED)
yellow = LED(GPIO_YELLOW)
green = LED(GPIO_GREEN)


def leds_red():
    red.on()
    yellow.off()
    green.off()


def leds_green():
    red.off()
    yellow.off()
    green.on()


def leds_off():
    red.off()
    yellow.off()
    green.off()


# ============================================================
# UTILIDADES
# ============================================================

def normalized_to_pixel(x, y):
    px = int(x * WIDTH)
    py = int(y * HEIGHT)
    return px, py


def hand_bbox(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    return (
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    )


def inside_ratio(landmarks, roi):
    x1, y1, x2, y2 = roi

    inside = 0

    for lm in landmarks:
        if x1 <= lm.x <= x2 and y1 <= lm.y <= y2:
            inside += 1

    return inside / len(landmarks)


def validate_hand(landmarks):
    """
    Comprueba que una mano:
    1. esté mayormente dentro del ROI
    2. no esté demasiado lejos
    3. no esté excesivamente cerca
    """

    ratio = inside_ratio(landmarks, ROI)

    x1, y1, x2, y2 = hand_bbox(landmarks)

    hand_width = x2 - x1
    hand_height = y2 - y1

    if ratio < MIN_INSIDE_RATIO:
        return False, f"fuera de zona ({ratio:.0%})"

    if hand_width < MIN_HAND_WIDTH or hand_height < MIN_HAND_HEIGHT:
        return False, "acerca las manos"

    if hand_width > MAX_HAND_WIDTH or hand_height > MAX_HAND_HEIGHT:
        return False, "aleja las manos"

    return True, "OK"


def validate_two_hands(result):
    """
    Para comenzar:
      - exactamente 2 manos
      - ambas cumplen geometría
    """

    hands = result.hand_landmarks

    if len(hands) == 0:
        return False, "No veo manos"

    if len(hands) == 1:
        return False, "Necesito ver las dos manos"

    if len(hands) < 2:
        return False, "Necesito ver las dos manos"

    reasons = []

    for i, hand in enumerate(hands[:2]):
        ok, reason = validate_hand(hand)

        if not ok:
            reasons.append(f"Mano {i + 1}: {reason}")

    if reasons:
        return False, " | ".join(reasons)

    return True, "Posicion correcta"


def draw_hand(frame, landmarks):
    points = []

    for lm in landmarks:
        x, y = normalized_to_pixel(lm.x, lm.y)
        points.append((x, y))

    # conexiones
    for a, b in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[a],
            points[b],
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

    # landmarks
    for x, y in points:
        cv2.circle(
            frame,
            (x, y),
            4,
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            frame,
            (x, y),
            2,
            (0, 100, 255),
            -1,
            cv2.LINE_AA,
        )

    # bounding box real de esa mano
    x1, y1, x2, y2 = hand_bbox(landmarks)

    p1 = normalized_to_pixel(x1, y1)
    p2 = normalized_to_pixel(x2, y2)

    cv2.rectangle(
        frame,
        p1,
        p2,
        (255, 255, 0),
        2,
    )


def draw_target_roi(frame, valid=False):
    x1, y1, x2, y2 = ROI

    p1 = normalized_to_pixel(x1, y1)
    p2 = normalized_to_pixel(x2, y2)

    if valid:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    cv2.rectangle(
        frame,
        p1,
        p2,
        color,
        3,
    )

    cv2.putText(
        frame,
        "ZONA DE INICIO",
        (p1[0], max(30, p1[1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_center_text(frame, text, scale=2.0):
    font = cv2.FONT_HERSHEY_SIMPLEX

    size, _ = cv2.getTextSize(
        text,
        font,
        scale,
        4,
    )

    x = (WIDTH - size[0]) // 2
    y = (HEIGHT + size[1]) // 2

    # borde negro
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        (0, 0, 0),
        8,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_hand_presence_confidence=MIN_PRESENCE_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)

landmarker = HandLandmarker.create_from_options(options)


# ============================================================
# CÁMARA
# ============================================================

camera = Picamera2()

camera_config = camera.create_preview_configuration(
    main={
        "size": (WIDTH, HEIGHT),
        "format": "RGB888",
    }
)

camera.configure(camera_config)
camera.start()

time.sleep(1.5)


# ============================================================
# STATE MACHINE DE PRUEBA
# ============================================================

state = "WAITING"

stable_since = None
countdown_started = None

leds_red()

print()
print("==========================================")
print(" LAIA - TEST POSICIONAMIENTO DE MANOS")
print("==========================================")
print()
print("Rojo     GPIO17")
print("Amarillo GPIO27 [apagado en esta prueba]")
print("Verde    GPIO22")
print()
print("Q / ESC = salir")
print("R       = reiniciar")
print()


try:

    while True:

        # Picamera2 entrega el buffer del stream RGB888 en orden BGR en esta
        # Raspberry. Mantener el contrato BGR del backend y convertir una sola
        # vez al formato SRGB que exige MediaPipe Tasks.
        frame_bgr = camera.capture_array()

        # Mirror: más intuitivo para una persona frente a la pantalla.
        frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        timestamp_ms = int(time.monotonic() * 1000)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        frame = frame_bgr.copy()

        # ----------------------------------------------------
        # Dibujar landmarks de todas las manos detectadas
        # ----------------------------------------------------

        for hand_landmarks in result.hand_landmarks:
            draw_hand(frame, hand_landmarks)

        # ----------------------------------------------------
        # Validar posición
        # ----------------------------------------------------

        placement_ok, reason = validate_two_hands(result)

        now = time.monotonic()

        # ====================================================
        # WAITING
        # ====================================================

        if state == "WAITING":

            leds_red()

            if placement_ok:

                if stable_since is None:
                    stable_since = now

                elapsed = now - stable_since

                remaining = max(
                    0.0,
                    STABLE_SECONDS - elapsed,
                )

                reason = (
                    f"Manten las manos ahi... "
                    f"{remaining:.1f}s"
                )

                if elapsed >= STABLE_SECONDS:

                    state = "COUNTDOWN"
                    countdown_started = now

                    leds_green()

            else:
                stable_since = None

        # ====================================================
        # COUNTDOWN
        # ====================================================

        elif state == "COUNTDOWN":

            if not placement_ok:

                print("Posicion perdida -> reiniciando")

                state = "WAITING"
                stable_since = None
                countdown_started = None

                leds_red()

                reason = "Posicion perdida"

            else:

                leds_green()

                elapsed = now - countdown_started

                remaining = (
                    COUNTDOWN_SECONDS - elapsed
                )

                if remaining > 0:

                    number = max(
                        1,
                        math.ceil(remaining),
                    )

                    draw_center_text(
                        frame,
                        str(number),
                        scale=3.5,
                    )

                    reason = (
                        "LISTO - manten las manos"
                    )

                else:

                    state = "STARTED"

                    leds_green()

                    print()
                    print("========================")
                    print("   INICIO AUTORIZADO")
                    print("========================")
                    print()

        # ====================================================
        # STARTED
        # ====================================================

        elif state == "STARTED":

            leds_green()

            draw_center_text(
                frame,
                "INICIO OK",
                scale=1.7,
            )

            reason = (
                "Aqui comenzaria el modelo LAIA"
            )

        # ----------------------------------------------------
        # Dibujar ROI
        # ----------------------------------------------------

        roi_green = (
            state in ("COUNTDOWN", "STARTED")
        )

        draw_target_roi(
            frame,
            valid=roi_green,
        )

        # ----------------------------------------------------
        # Información de debug
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (0, HEIGHT - 80),
            (WIDTH, HEIGHT),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            f"Estado: {state}",
            (15, HEIGHT - 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            reason[:75],
            (15, HEIGHT - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # Mostrar
        # ----------------------------------------------------

        cv2.imshow(
            "LAIA - Camera Positioning Test",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        if key == ord("r"):

            print("Reinicio manual")

            state = "WAITING"
            stable_since = None
            countdown_started = None

            leds_red()


finally:

    print("Cerrando...")

    leds_off()

    camera.stop()
    landmarker.close()

    cv2.destroyAllWindows()
