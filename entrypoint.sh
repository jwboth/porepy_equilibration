#!/bin/sh
set -e

# Ensure the output directory exists (important when the directory is
# bind-mounted from the host and may not have been created yet).
mkdir -p /work/output

uv run python -m example2 --without-reference-states --instant-bc
uv run python -m example2 --with-reference-states --instant-bc
uv run python -m example2 --with-reference-states --instant-bc --low-friction
uv run python -m example2 --with-reference-states --instant-bc --no-friction
