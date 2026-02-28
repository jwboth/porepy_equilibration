# porepy_equilibration

Run scripts for the paper *"Consistent initialization of mixed-dimensional multiphysics models for fractured reservoirs under geomechanical constraints and field measurements"*.

## Installation

### Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
# On Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### 1. Clone the repository with submodules

```bash
git clone --recurse-submodules https://github.com/jwboth/porepy_equilibration.git
cd porepy_equilibration
```

If you have already cloned the repository without submodules, initialise them with:

```bash
git submodule update --init --recursive
```

### 2. Install with uv

Create a virtual environment and install `porepy_equilibration` together with all
dependencies (including [PorePy](https://github.com/pmgbergen/porepy) from the
bundled submodule):

```bash
uv sync
```

To install in an existing environment without creating a new virtual environment:

```bash
uv pip install -e .
```

### 3. Run the examples

After installation, each example can be executed as a Python module.

**Example 1:**

```bash
python -m example1
```

Or, using the uv-managed virtual environment directly:

```bash
uv run python -m example1
```

## Project structure

```
porepy_equilibration/
├── apps/
│   └── example1/          # Example application (python -m example1)
├── external/
│   └── porepy/            # PorePy git submodule (https://github.com/pmgbergen/porepy)
├── src/
│   └── porepy_equilibration/   # Main library package
├── pyproject.toml
└── README.md
```

## Docker

A `Dockerfile` is provided so that the example can be run in a fully
reproducible environment without installing anything locally (other than
Docker).

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) ≥ 20.10

### 1. Clone the repository with submodules

```bash
git clone --recurse-submodules https://github.com/jwboth/porepy_equilibration.git
cd porepy_equilibration
```

> **Note** – the `external/porepy` git submodule must be initialised before
> building the image so that the build context contains the source code.
> If you cloned without `--recurse-submodules`, run:
>
> ```bash
> git submodule update --init --recursive
> ```

### 2. Build the image

```bash
docker build -t porepy-equilibration .
```

### 3. Run the container

Run `apps.example1` and write any output files produced under `visualization/`
to the **host** directory `./visualization` via a bind-mount:

```bash
mkdir -p visualization
docker run --rm \
  -v "$PWD/visualization:/work/visualization" \
  porepy-equilibration
```

After the container exits, `./visualization/` on the host will contain any
files that `apps.example1` wrote to `/work/visualization` inside the container.

### 4. Run with Docker Compose (alternative)

```bash
docker compose up --build
```

The `docker-compose.yml` file already includes the `./visualization` bind-mount.

### Environment variables

No environment variables are required for the default run.  Custom flags can
be appended after the image name and are forwarded verbatim to
`python -m apps.example1`:

```bash
docker run --rm -v "$PWD/visualization:/work/visualization" \
  porepy-equilibration --some-flag
```

## Development

Update the PorePy submodule to the latest version:

```bash
git submodule update --remote external/porepy
```
