#!/usr/bin/env bash
set -u

echo "== Cámara =="
rpicam-hello --list-cameras 2>&1 || true
v4l2-ctl --list-devices 2>&1 || true

echo "== Display =="
for item in /sys/class/drm/*/status /sys/class/drm/*/modes; do
    if [[ -r "$item" ]]; then
        echo "$item"
        tr '\n' ' ' < "$item"
        echo
    fi
done

echo "== GPIO =="
pinctrl get 17 2>&1 || true
pinctrl get 22 2>&1 || true

echo "== Audio =="
aplay -l 2>&1 || true
wpctl status 2>&1 || true

