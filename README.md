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

Para preparar la comprobación de audio solicitada:

```bash
./scripts/run_laia.sh --simulate --windowed --no-gpio
```

Pulsa `Espacio` o `→` una vez para iniciar, `x` para entrar en `CORRECTION` y
varias veces más para verificar que los `retry` no reinician “Oops”. Después
recupera el paso con `Espacio` o `→` y pulsa `x` en el siguiente paso para
confirmar que una nueva entrada en `CORRECTION` puede reproducirlo otra vez.

## Diagnóstico de inferencia

LAIA registra una línea `Prediction:` por cada evaluación efectiva del modelo
(`prediction` o `insufficient_pose`), no por cada frame. Incluye clase superior,
nombre, confianza, cobertura de pose, rachas, estado, pasos aceptados, eventos y
diagnósticos ligeros de tracking. Los valores ausentes se escriben como `none`.

Para extraer el intervalo de una prueba sin tratar el log como binario:

```bash
grep -a '^2026-09-01' logs/laia-app.log | grep -E 'Prediction:|Evento:|Estado:|Inicio LAIA'
```

## Diagnóstico invisible de laboratorio

El launcher normal (`./scripts/run_laia.sh` o `./LAIA`) activa la capa
diagnóstica sin añadir controles a la pantalla ni cambiar la inferencia. No se
graba nada en `WAITING_FOR_HANDS`. Cuando aparece `session_started`, se genera
un ID local `YYYYMMDD_HHMMSS`; si ya existe un artefacto, se usa `_02`, `_03`,
etc. sin sobrescribirlo.

Durante una sesión activa se crean, en local y sin subir ni borrar archivos:

- `logs/recordings/session_<id>_raw.mp4`: frames BGR originales, sin overlays ni audio.
- `logs/recordings/session_<id>_annotated.mp4`: copia anotada con landmarks, estado, predicción y térmicos cacheados.
- `logs/telemetry/session_<id>.csv`: muestras térmicas aproximadamente cada 5 segundos.

Ambos MP4 usan los mismos frames muestreados a 10 FPS desde la cadencia normal
de cámara/inferencia, con cola acotada y no bloqueante. El log `Prediction:`,
`Evento:` y `Thermal:` lleva el mismo `session_id`; `SessionSummary:` incluye
duración, contadores de sampling/cola/escritura, tamaños, FPS efectivos,
latencia de actualización, intervalos de predicción y resumen térmico. La
temperatura mostrada es SoC, no temperatura ambiente.

El CSV tiene el contrato fijo de columnas térmicas (`soc_temp_c`, fuente
`sysfs_cpu_thermal` o fallback `vcgencmd_cpu_thermal`, throttling actual y
ocurrido, frecuencia, carga y número de CPUs). Una lectura ausente se conserva
como `none`; no se usa `psutil` ni se consulta el sensor desde MediaPipe, ONNX
o Tkinter.

Para desactivar selectivamente la capa, sin cambiar `RuntimeConfig`:

```bash
LAIA_DIAGNOSTICS_ENABLED=0 ./scripts/run_laia.sh
LAIA_RECORDING_ENABLED=0 ./scripts/run_laia.sh
LAIA_THERMAL_ENABLED=0 ./scripts/run_laia.sh
```

Los modos `--simulate` y `--preview-state` desactivan grabación y lectura
térmica para no convertir una vista previa en un artefacto de laboratorio.
Los archivos generados están excluidos por `.gitignore`.

### Procedimiento de validación técnica

1. Ejecuta `./scripts/hardware_diagnostics.sh` y registra sus resultados.
2. Arranca el launcher normal y confirma en `WAITING_FOR_HANDS` que no aparecen
   MP4 ni CSV y que la interfaz permanece igual.
3. Realiza únicamente una prueba de laboratorio informada/controlada. Conserva
   el ID del primer `Prediction:` con `session_started` y comprueba que coincide
   con los dos MP4, el CSV y los logs terminales.
4. Comprueba los streams sin reproducirlos en pantalla:

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,codec_name,avg_frame_rate \
  -of default=noprint_wrappers=1 logs/recordings/session_<id>_raw.mp4
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,codec_name,avg_frame_rate \
  -of default=noprint_wrappers=1 logs/recordings/session_<id>_annotated.mp4
```

Verifica que el raw no contiene dibujos, que el annotated sí contiene la
anotación, que los dos streams tienen el mismo número de frames y que no hay
audio. No se deben ajustar umbrales, pasos ni tiempos de aprobación a partir de
una sola prueba.

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
- el estado ha confirmado los seis pasos esperados, del 1 al 6. La cobertura
  comunicada por el modelo se conserva como diagnóstico y no sustituye esas
  confirmaciones.

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

La suite de aplicación incluye pruebas de snapshots de pose sin segunda
inferencia, correlación de sesiones, colas no bloqueantes, overlays seguros,
thermal fallback, límites de disco y MP4 sintéticos verificables con
`ffprobe`.

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
