#!/usr/bin/env bash
# Launch the native runtime with a read-only title and per-user saves.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || ( $# == 2 && $2 != --bind-only ) ]]; then
    echo "Usage: $0 /path/to/title.exe [--bind-only]" >&2
    exit 2
fi
exe=$(realpath -e -- "$1")
[[ -f $exe ]] || { echo "Expected an executable file" >&2; exit 2; }
title=$(dirname -- "$exe")
repo=$(dirname -- "$(dirname -- "$(realpath -- "$0")")")
bin=$(realpath -e -- "${VICTORY_BIN:-$repo/target/release/victory}")
cache=$(realpath -m -- "${VICTORY_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/victory/cache}")
data=$(realpath -m -- "${VICTORY_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/victory}")
for writable in "$cache" "$data"; do
    if [[ $writable == / || $title == / || $writable == "$title" || $writable == "$title/"* || $title == "$writable/"* ]]; then
        echo "Cache and save directories must be separate from the title directory" >&2
        exit 2
    fi
done
mkdir -p -- "$cache" "$data"
args=(--execute)
if [[ ${2:-} == --bind-only ]]; then args=(); fi
exec bwrap --die-with-parent --unshare-pid --ro-bind / / --dev-bind /dev /dev --proc /proc \
    --bind "$cache" "$cache" --bind "$data" "$data" --ro-bind "$title" "$title" \
    --chdir "$cache" --setenv VICTORY_GAME_DIR "$title" \
    --setenv VICTORY_CACHE_DIR "$cache" --setenv VICTORY_DATA_DIR "$data" \
    --setenv VKD3D_SHADER_CACHE_PATH "$cache" -- "$bin" run "$exe" "${args[@]}"
