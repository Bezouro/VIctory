#!/usr/bin/env bash
# Launch the native runtime with a read-only title and per-user saves.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 /path/to/title.exe [--bind-only] [--console series-x|series-s]" >&2
    exit 2
fi
exe=$(realpath -e -- "$1")
[[ -f $exe ]] || { echo "Expected an executable file" >&2; exit 2; }
shift
mode=(--execute)
console_args=()
while (( $# )); do
    case "$1" in
        --bind-only) mode=(); shift ;;
        --console)
            [[ ${2:-} == series-x || ${2:-} == series-s ]] || { echo "Expected series-x or series-s" >&2; exit 2; }
            console_args=(--console "$2"); shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
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
exec bwrap --die-with-parent --unshare-pid --ro-bind / / --dev-bind /dev /dev --proc /proc \
    --bind "$cache" "$cache" --bind "$data" "$data" --ro-bind "$title" "$title" \
    --chdir "$cache" --setenv VICTORY_GAME_DIR "$title" \
    --setenv VICTORY_CACHE_DIR "$cache" --setenv VICTORY_DATA_DIR "$data" \
    --setenv VKD3D_SHADER_CACHE_PATH "$cache" -- "$bin" run "$exe" "${mode[@]}" "${console_args[@]}"
