#!/usr/bin/env python3

import time

from gpiozero import LED


# Numeracion BCM y su pin fisico correspondiente.
PINES = [
    (17, 11),
    (22, 15),
    (26, 37),
    (20, 38),
]

SEGUNDOS_ENCENDIDO = 3
SEGUNDOS_PAUSA = 1


def main():
    print("Prueba secuencial de GPIO 17, 22, 26 y 20")
    print("Presiona Ctrl+C para detener la prueba.\n")

    try:
        for gpio, pin_fisico in PINES:
            led = LED(gpio, active_high=True, initial_value=False)
            try:
                print(
                    f">>> ENCENDIDO: GPIO{gpio} / pin fisico {pin_fisico}",
                    flush=True,
                )
                led.on()
                time.sleep(SEGUNDOS_ENCENDIDO)
                led.off()
                print(f"<<< APAGADO: GPIO{gpio}\n", flush=True)
            finally:
                led.off()
                led.close()

            time.sleep(SEGUNDOS_PAUSA)

    except KeyboardInterrupt:
        print("\nPrueba interrumpida; el GPIO activo quedo apagado.")
    else:
        print("Prueba terminada; todos los GPIO quedaron apagados.")


if __name__ == "__main__":
    main()
