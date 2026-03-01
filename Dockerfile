# syntax=docker/dockerfile:1
FROM python:3.13-slim

# Install git (needed to initialise the porepy submodule) and build tools
# required by some of porepy's native dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libglu1-mesa \
        libgl1 \
        libx11-6 \
        libxrender1 \
        libxi6 \
        libxcursor1 \
        libxinerama1 \
        libxrandr2 \
        libxext6 \
        libxfixes3 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /work

# Copy repository (including .git so submodules can be initialised)
COPY . /work

# Initialise the porepy git submodule so uv can install it from the local path
RUN git submodule update --init --recursive

# Install all project dependencies (including the local porepy editable install)
RUN uv sync

# Create the output directory so it is ready for a bind-mount
RUN mkdir -p /work/output

RUN chmod +x /work/entrypoint.sh

ENTRYPOINT ["/work/entrypoint.sh"]
CMD []
