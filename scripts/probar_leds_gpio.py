#!/usr/bin/env python3
"""Prueba secuencial de los GPIO usados como LEDs en el proyecto LAIA."""

import argparse
import time

from gpiozero import LED


# Mapeo del conector Raspberry Pi de 40 pines. Se excluyen alimentacion, tierra
# y los pines fisicos 27/28 (GPIO0/GPIO1), reservados para identificar HATs.
PHYSICAL_BY_BCM = {
    2: 3, 3: 5, 4: 7, 14: 8, 15: 10,
    17: 11, 18: 12, 27: 13, 22: 15, 23: 16, 24: 18,
    10: 19, 9: 21, 25: 22, 11: 23, 8: 24, 7: 26,
    5: 29, 6: 31, 12: 32, 13: 33, 19: 35, 16: 36,
    26: 37, 20: 38, 21: 40,
}

ALL_GPIO_PINS = list(PHYSICAL_BY_BCM)

# Pines encontrados en la implementacion actual y en la prueba antigua.
PIN_INFO = {
    17: "configuracion principal: LED verde (pin fisico 11)",
    22: "configuracion principal: LED rojo (pin fisico 15)",
    26: "prueba antigua: LED verde (pin fisico 37)",
    20: "prueba antigua: LED rojo (pin fisico 38)",
}


def probar_pin(pin: int, segundos: float) -> None:
    descripcion = PIN_INFO.get(pin, "pin solicitado manualmente")
    fisico = PHYSICAL_BY_BCM.get(pin)
    ubicacion = f"pin fisico {fisico}" if fisico else "pin fisico desconocido"
    led = LED(pin, active_high=True, initial_value=False)
    try:
        print(
            f"\n>>> ENCENDIDO: GPIO{pin} / {ubicacion} - {descripcion}",
            flush=True,
        )
        led.on()
        time.sleep(segundos)
        led.off()
        print(f"<<< APAGADO:   GPIO{pin}", flush=True)
    finally:
        led.off()
        led.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enciende, uno por uno, los GPIO candidatos para los LEDs de LAIA."
    )
    parser.add_argument(
        "--pins",
        type=int,
        nargs="+",
        default=[17, 22, 26, 20],
        help="GPIO en numeracion BCM (predeterminado: 17 22 26 20)",
    )
    parser.add_argument(
        "--todos",
        action="store_true",
        help="prueba todos los GPIO utilizables del conector de 40 pines",
    )
    parser.add_argument(
        "--segundos",
        type=float,
        default=3.0,
        help="tiempo encendido por pin (predeterminado: 3)",
    )
    parser.add_argument(
        "--pausa",
        type=float,
        default=1.0,
        help="pausa entre pines (predeterminado: 1)",
    )
    args = parser.parse_args()

    if args.todos:
        args.pins = ALL_GPIO_PINS

    if args.segundos <= 0 or args.pausa < 0:
        parser.error("--segundos debe ser mayor que 0 y --pausa no puede ser negativa")

    print("Prueba de LEDs LAIA (numeracion BCM)")
    print("Asegurate de que cada LED tenga una resistencia de 220-330 ohmios.")
    if args.todos:
        print("Modo TODOS: no se probaran 5 V, 3.3 V, GND ni los pines ID 27/28.")
    print("Presiona Ctrl+C para detener; el GPIO activo se apagara.")

    try:
        for indice, pin in enumerate(args.pins):
            probar_pin(pin, args.segundos)
            if indice < len(args.pins) - 1:
                time.sleep(args.pausa)
    except KeyboardInterrupt:
        print("\nPrueba interrumpida; salida apagada.")
    except Exception as error:
        print(f"\nNo se pudo probar el GPIO: {error}")
        raise SystemExit(1) from error
    else:
        print("\nPrueba terminada; todos los GPIO probados quedaron apagados.")


if __name__ == "__main__":
    main()
