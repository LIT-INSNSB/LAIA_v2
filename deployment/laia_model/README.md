# LAIA hand-washing inference package

This directory is a self-contained inference package for the selected
`LightSTGCNV2` PSKUS checkpoint. It does not contain a trainer and it is not an
Ultralytics/YOLO model.

## Model contract

- Classes: `0..6`; raw PSKUS class `7` was preserved upstream but excluded.
- Temporal context: 1.5 seconds, resampled to 32 steps.
- Prediction stride: 0.375 seconds.
- Pose: two slots of 21 MediaPipe landmarks, filled by compacting the currently
  observed stable track IDs. Track association is temporal, but slots are not
  anatomical Left/Right and can compact after a dropout.
- Features: normalized joints, first temporal differences per second, bone
  vectors, and explicit validity masks.
- Model inputs: three float32 arrays `(N,3,32,42)` and one boolean mask
  `(N,32,42)`.
- Model output: seven raw logits. Softmax values are reported as
  `confidence_uncalibrated`; no deployment threshold has been calibrated.

The valid operating context is a known hand-washing episode. Class 0 means
`other_hand_washing_movement`; it does **not** mean `not washing`. The model
does not recognize paper/faucet class 7.

## Selected model

The source is `best.pt` at epoch 47, selected by validation macro-F1. It is not
`last.pt` (epoch 59), because the latter has lower validation macro-F1. The
runtime artifact is `model/laia_lightstgcnv2_fp32.onnx`; the small `.pt` file is
an optional inference-only PyTorch fallback and provenance artifact.

## Raspberry Pi OS 64-bit installation

The target is Raspberry Pi 5 with 64-bit Raspberry Pi OS and Python 3.11.
Official ARM64 wheels exist for the pinned MediaPipe Tasks and ONNX Runtime
versions, so this package does not require PyTorch on the Pi.

```bash
cd laia_model
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-inference.txt
python smoke_test.py --threads 4
```

Run a video:

```bash
python inference.py video \
  --input /path/to/video.mp4 \
  --pose-api tasks \
  --task-model model/hand_landmarker.task \
  --model model/laia_lightstgcnv2_fp32.onnx \
  --threads 4
```

Each output line is JSON. There is no prediction until the 1.5-second temporal
buffer is ready. Calling the image command intentionally returns
`status=warming_up`, because one frame cannot identify a motion:

```bash
python inference.py image --input /path/to/frame.jpg --pose-api tasks
```

If a complete window contains no observed hand pose, the API returns
`status=insufficient_pose` and leaves class, name, and confidence as `null`.
This integrity gate avoids turning a wholly missing signal into class 0.

For integration with an existing camera loop, instantiate
`StreamingHandwashingRecognizer` and call `update_frame(frame_bgr,
timestamp_s=...)`. Call `reset_temporal_state()` when starting a new movement
window; it clears MediaPipe, tracking and temporal-buffer state without
reopening Picamera2. `reset()` remains an alias for a full recognizer reset.

## Pose API compatibility

The PSKUS feature store was extracted with legacy
`mediapipe.solutions.hands==0.10.21`. That exact PyPI release has no official
Linux ARM64 wheel. Raspberry Pi therefore uses the officially supported
MediaPipe Tasks `HandLandmarker` from `mediapipe==1.0.1` and the bundled
`hand_landmarker.task`.

This substitution is explicit, never silent. A controlled three-video probe
covering 927 frames found 94.71% agreement in hand count and 97.30% agreement
in downstream window class. Those are agreement proxies, not ground truth and
not proof of full equivalence. See
`reports/mediapipe_legacy_vs_tasks_probe.json`.

The legacy x86 reference can be recreated separately with:

```bash
python3 -m venv .venv-legacy
source .venv-legacy/bin/activate
pip install -r requirements/server-legacy-reference.txt
```

Do not install MediaPipe 0.10.21 into the W&B training environment: it requires
`protobuf<5`, while the installed W&B release requires `protobuf>=5`.

## Pose-window API and portable smoke test

Classifier-only inference from an explicit pose window:

```bash
python inference.py pose-window \
  --input examples/pose_windows/smoke_window.npz \
  --model model/laia_lightstgcnv2_fp32.onnx
```

The 21 real validation examples contain pose arrays only, not RGB or personal
images. `parity_test.py` compares source PyTorch with ONNX and requires 100%
argmax agreement:

```bash
conda activate DL-env
PYTHONPATH=. python parity_test.py \
  --checkpoint /path/to/best.pt
```

## Raspberry Pi benchmark

First benchmark the classifier:

```bash
python benchmark_pi.py --iterations 1000 --threads 4
```

Then benchmark complete video/camera inference separately. Decode, MediaPipe,
feature construction, classifier latency, end-to-end FPS, RSS, temperature,
and `vcgencmd get_throttled` should be recorded for at least 30 minutes before
production promotion. Server timing must not be extrapolated to the Pi.

Run the deployment unit tests with:

```bash
python -m unittest discover -s tests -v
```

## Files

- `model_manifest.json`: immutable model/preprocessing/dependency contract.
- `model/laia_lightstgcnv2_fp32.onnx`: recommended Raspberry artifact.
- `model/laia_lightstgcnv2_best_state.pt`: optional PyTorch fallback.
- `model/hand_landmarker.task`: official MediaPipe Tasks hand model.
- `laia_inference/`: frozen model-specific preprocessing and streaming API.
- `smoke_test.py`: portable known-output check.
- `parity_test.py`: source PyTorch versus ONNX comparison.
- `reports/parity.json`: measured numerical parity.
