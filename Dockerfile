# syntax=docker/dockerfile:1
FROM python:3.13-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install git (needed to initialise the porepy submodule), build tools,
# and all runtime libraries required by gmsh and porepy simulations.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        vim \
        libglu1-mesa \
        libgl1 \
        libglx-mesa0 \
        libx11-6 \
        libxrender1 \
        libxi6 \
        libxcursor1 \
        libxinerama1 \
        libxrandr2 \
        libxext6 \
        libxfixes3 \
        libxft2 \
        libxss1 \
        libxau6 \
        libxdmcp6 \
        libsm6 \
        ffmpeg \
        python3-tk \
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

# Install pypardiso (and its MKL dependencies) for high-performance sparse solves
RUN uv pip install pypardiso

# Create the output directory so it is ready for a bind-mount
RUN mkdir -p /work/output

RUN chmod +x /work/entrypoint.sh

ENTRYPOINT ["/work/entrypoint.sh"]
CMD []
