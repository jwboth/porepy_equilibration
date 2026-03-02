# porepy_equilibration

Run scripts for the paper *"Consistent initialization of mixed-dimensional multiphysics models for fractured reservoirs under geomechanical constraints and field measurements"* by Jakub W. Both and Inga Berre.

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

Run all four `example2` variants in sequence. Output is written to
`/work/output` inside the container; bind-mount the host's `./output`
directory to get the files locally:

```bash
mkdir -p output
docker run --rm \
  -v "$PWD/output:/work/output" \
  porepy-equilibration
```

The container runs the following commands in order:

```
uv run python -m example2 --without-reference-states --instant-bc
uv run python -m example2 --with-reference-states --instant-bc
uv run python -m example2 --with-reference-states --instant-bc --low-friction
uv run python -m example2 --with-reference-states --instant-bc --no-friction
```

After the container exits, `./output/` on the host will contain all
generated subfolders and files.

### 4. Run with Docker Compose (alternative)

```bash
docker compose up --build
```

The `docker-compose.yml` file already includes the `./output` bind-mount.

### Environment variables

No environment variables are required.

## Development

Update the PorePy submodule to the latest version:

```bash
git submodule update --remote external/porepy
```
