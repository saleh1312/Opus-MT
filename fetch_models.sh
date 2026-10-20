#!/usr/bin/env bash
# Download models named in services.json when their files are not already on disk.
# The Docker entrypoint runs this before server.py, which loads every SentencePiece
# model at startup and exits if one of those files is missing.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
CONFIG="${1:-services.json}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Model config not found: $CONFIG" >&2
    exit 1
fi

mapfile -t MISSING < <(python3 - "$CONFIG" <<'PY'
import json
import os
import sys

with open(sys.argv[1]) as handle:
    services = json.load(handle)

seen = []
for targets in services.values():
    for config in targets.values():
        needed = [config["configuration"]]
        for key in ("sourcespm", "targetspm", "sourcebpe", "targetbpe"):
            path = config.get(key)
            if path:
                needed.append(path)
        if all(os.path.isfile(path) for path in needed):
            continue
        pair = os.path.basename(os.path.dirname(config["configuration"]))
        if pair not in seen:
            seen.append(pair)
            print(pair)
PY
)

if [[ ${#MISSING[@]} -eq 0 ]]; then
    echo "All models listed in ${CONFIG} are already present."
    exit 0
fi

declare -A URLS=(
    [ar-en]="https://object.pouta.csc.fi/OPUS-MT-models/ar-en/opus-2019-12-18.zip"
    [ar-fr]="https://object.pouta.csc.fi/OPUS-MT-models/ar-fr/opus-2020-01-24.zip"
    [en-ar]="https://object.pouta.csc.fi/Tatoeba-MT-models/eng-ara/opus+bt-2021-04-13.zip"
    [en-fr]="https://object.pouta.csc.fi/OPUS-MT-models/en-fr/opus-2020-02-26.zip"
    [en-es]="https://object.pouta.csc.fi/OPUS-MT-models/en-es/opus-2019-12-04.zip"
    [fr-en]="https://object.pouta.csc.fi/OPUS-MT-models/fr-en/opus-2020-02-26.zip"
    [fr-es]="https://object.pouta.csc.fi/OPUS-MT-models/fr-es/opus-2020-01-09.zip"
    [es-en]="https://object.pouta.csc.fi/OPUS-MT-models/es-en/opus-2019-12-04.zip"
    [es-fr]="https://object.pouta.csc.fi/OPUS-MT-models/es-fr/opus-2020-01-08.zip"
)

download_one() {
    local pair="$1"
    local url="${URLS[$pair]:-}"
    local dest="${ROOT}/models/${pair}"
    if [[ -z "$url" ]]; then
        echo "No download URL for ${pair}. Add the files under models/${pair}/ or extend fetch_models.sh." >&2
        return 1
    fi
    mkdir -p "$dest"
    local tmp
    tmp="$(mktemp -d)"
    echo "Downloading ${pair} from ${url}"
    curl -fL --retry 4 --retry-delay 3 -o "${tmp}/model.zip" "$url"
    unzip -q -o "${tmp}/model.zip" -d "$dest"
    rm -rf "$tmp"
    if [[ ! -f "${dest}/decoder.yml" ]]; then
        local packed
        packed="$(find "$dest" -name decoder.yml -print -quit || true)"
        if [[ -n "$packed" && "$packed" != "${dest}/decoder.yml" ]]; then
            mv "$(dirname "$packed")"/* "$dest"/
        fi
    fi
    if [[ ! -f "${dest}/decoder.yml" ]]; then
        echo "Downloaded ${pair} but decoder.yml is missing." >&2
        return 1
    fi
    # The container runs as root. Keep new files owned by whoever owns ./models
    # so they stay usable from the host.
    local owner
    owner="$(stat -c '%u:%g' "${ROOT}/models" 2>/dev/null || true)"
    if [[ -n "$owner" && "$owner" != "0:0" ]]; then
        chown -R "$owner" "$dest" || true
    fi
    echo "Installed ${pair}"
}

status=0
pids=()
for pair in "${MISSING[@]}"; do
    download_one "$pair" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
if [[ "$status" -ne 0 ]]; then
    echo "One or more models failed to download." >&2
    exit 1
fi

# Confirm the files server.py will open at startup are actually there.
fetch_models_recheck="$(python3 - "$CONFIG" <<'PY'
import json
import os
import sys

with open(sys.argv[1]) as handle:
    services = json.load(handle)
missing = []
for targets in services.values():
    for config in targets.values():
        needed = [config["configuration"]]
        for key in ("sourcespm", "targetspm", "sourcebpe", "targetbpe"):
            path = config.get(key)
            if path:
                needed.append(path)
        for path in needed:
            if not os.path.isfile(path):
                missing.append(path)
print("\n".join(missing))
PY
)"
if [[ -n "$fetch_models_recheck" ]]; then
    echo "Model files are still missing after download:" >&2
    echo "$fetch_models_recheck" >&2
    exit 1
fi
echo "Model download complete."
