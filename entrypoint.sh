#!/bin/sh
set -e

# Ensure the visualization output directory exists (important when the
# directory is bind-mounted from the host and may not have been created yet).
mkdir -p /work/visualization

exec uv run python -m apps.example1 "$@"
