#!/usr/bin/env bash
# Fetch any missing translation models, then start the server.
set -euo pipefail

cd /usr/src/app
config="services.json"
previous=""
for arg in "$@"; do
    if [[ "$previous" == "-c" || "$previous" == "--config" ]]; then
        config="$arg"
    fi
    previous="$arg"
done

/usr/src/app/fetch_models.sh "$config"
exec "$@"
