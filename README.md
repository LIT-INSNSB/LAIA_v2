# LAIA en Raspberry Pi 5

Workspace integrado para la interfaz, el deployment ONNX, los assets definitivos,
las pruebas y el diagnóstico de hardware de LAIA.

## Ejecutar

```bash
cd /home/labinnovaciont/Desktop/LAIA
./scripts/run_laia.sh
```

Modo simulación sin cámara, display físico, GPIO ni audio:

```bash
./scripts/run_laia.sh --simulate --windowed
```

En simulación: `Espacio` o `→` confirma el paso esperado, `x` simula una clase
incorrecta, `r` reinicia y `Esc` sale. Estos controles no aparecen en pantalla.

## Contrato visual

- `WAITING_FOR_HANDS`: cámara live con `silueta.png`.
- `WASHING`: `Paso1.png` a `Paso6.png` según el paso esperado.
- `CORRECTION`: conserva el PNG del paso esperado; nunca usa la clase errónea.
- `SUCCESS`: usa únicamente `manos_limpias.png` y reproduce
  `manos_limpias.mp3`; señal de acierto en BCM22 con secuencia temporizada.

Los PNG se cargan con alpha, se escalan con `contain` y no se recortan ni se
deforman. El resto de la interfaz (check, textos, progreso, burbujas y estrellas)
es UI nativa de Tkinter.

## Criterio de aprobación

Para tolerar errores puntuales del modelo, la sesión conserva los pasos distintos
identificados durante todo el lavado. LAIA solo entra en `SUCCESS` cuando se
cumplen ambas condiciones:

- han transcurrido al menos 20 segundos continuos con las manos visibles;
- se han identificado al menos 4 de los 6 pasos, o el modelo ha comunicado una
  cobertura de pasos de al menos 80 %.

Si la primera predicción activa `CORRECTION`, una evidencia posterior válida
puede recuperar la sesión sin usar la clase equivocada para escoger la imagen.
Perder las manos reinicia el reloj de 20 segundos; así nunca se aprueba una
sesión de menos de 20 segundos efectivos.

## Validar

```bash
deployment/laia_model/.venv/bin/python -m unittest discover -s tests -v
cd deployment/laia_model
.venv/bin/python smoke_test.py --threads 4
.venv/bin/python -m unittest discover -s tests -v
```

Diagnóstico de hardware:

```bash
./scripts/hardware_diagnostics.sh
```


## Pantalla vertical y giro

El panel DSI activo (actualmente `DSI-1`) tiene 800x480 píxeles físicos
(154x86 mm). Con la transformación 270, la interfaz es vertical: 480x800.

El botón táctil `↻ Girar`, visible arriba a la derecha, cambia entre
`normal → 90 → 180 → 270` y reajusta la interfaz sin reiniciar LAIA.
Después de girar, las imágenes mantienen sus proporciones con `contain`.

Para fijar explícitamente el output:

```bash
./LAIA --display-output DSI-1
```

## Ejecutable

`./LAIA` es el ejecutable de arranque y activa el entorno del deployment.
También se creó `/home/labinnovaciont/Desktop/LAIA.desktop` para iniciar LAIA
con doble clic desde el escritorio. Ambos conservan las dependencias del sistema
necesarias para Picamera2, GPIO, audio y Wayland.

## Selección de cámara

El botón `Cámara: ...`, arriba a la izquierda, abre un menú táctil con las
cámaras disponibles y permite reconectar la fuente seleccionada.

- `Automática`: prueba Picamera2 y después V4L2.
- `Cámara Raspberry Pi`: fuerza Picamera2 cuando está disponible.
- `USB · nombre (videoN)`: fuerza ese dispositivo USB sin abrir Picamera2.
- `Actualizar cámaras`: repite el descubrimiento tras conectar una webcam.

El cambio se realiza en segundo plano para no congelar la interfaz. Los nodos
internos de códec/ISP de la Raspberry Pi no aparecen como cámaras seleccionables.

También puede fijarse una cámara desde el ejecutable:

```bash
./LAIA --camera-source v4l2:/dev/video0
```

## Salir y señalización GPIO

El botón rojo `Salir`, arriba a la derecha debajo de `↻ Girar`, cierra LAIA y
apaga los GPIO. `Esc` mantiene el mismo comportamiento como atajo.

La señalización actual es:

- acierto (`SUCCESS`): BCM22, tres parpadeos, encendido durante 3 segundos y apagado;
- error (`CORRECTION`): BCM17, tres parpadeos, encendido durante 3 segundos y apagado;
- los tiempos no bloquean la interfaz y se cancelan al cambiar de estado o salir.

Para cambiar el cableado, edita exactamente estos campos en
`app/config.py`, dentro de `RuntimeConfig`:

```python
success_gpio_bcm = 22
error_gpio_bcm = 17
```

También se pueden ajustar `gpio_blink_count`, `gpio_blink_seconds` y
`gpio_hold_seconds` en ese mismo bloque.
