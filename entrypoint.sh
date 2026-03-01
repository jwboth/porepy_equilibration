#!/bin/sh
set -e

# Ensure the output directory exists (important when the directory is
# bind-mounted from the host and may not have been created yet).
mkdir -p /work/output

uv run python -m example1 --with-reference-state --gradual-bc
uv run python -m example1 --with-reference-state --instant-bc
uv run python -m example1 --with-reference-state --instant-bc
uv run python -m example1 --without-reference-state --instant-bc
