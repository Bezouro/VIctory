#!/usr/bin/env bash
set -euo pipefail
if (( $# < 1 || $# > 2 )); then
    echo "Usage: run.sh mesh.obj|model.glb|scene.json [frame.png]" >&2
    exit 2
fi
project=$(dirname -- "$(realpath -- "$0")")
source_file=$(realpath -e -- "$1")
options=(--max-fps 60)
files=("$source_file")
if (( $# == 2 )); then
    files+=("$(realpath -m -- "$2")")
    options=(--disable-vsync)
fi
exec "${GODOT_BIN:-godot}" --path "$project" --rendering-method mobile --rendering-driver vulkan \
    "${options[@]}" --script "$project/scene.gd" -- "${files[@]}"
