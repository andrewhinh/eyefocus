# eyefocus

**Stay focused!**

Built with:

- uv for project management.
- PyTorch for model training.
- Modal for model infra.
- FastHTML for the frontend.
- Ruff for linting and formatting.

## Set Up

Set up the environment:

```bash
uv sync --all-extras --dev
uvx pre-commit install
```

Optionally, set up Modal:

```bash
modal setup
```

## Repository Structure

```bash
.
├── frontend            # landing page.
├── ft                  # classifier training.
├── src                 # pypi package.
```

## Development

### PyPI

Run locally:

```bash
uv run eyefocus -vv
```

Build the package:

```bash
uvx --from build pyproject-build --installer uv
```

Upload the package:

```bash
uvx twine upload dist/*
```

Test the uploaded package:

```bash
uv run --with eyefocus --no-project -- eyefocus -vv
```

### Frontend

Run the app:

```bash
modal serve --env=dev frontend/app.py
```

Deploy on Modal in dev:

```bash
modal deploy --env=dev frontend/app.py
```

Deploy on Modal in main:

```bash
modal deploy --env=main frontend/app.py
```

### Training

Optionally, manually collect screenshots:

```bash
uv run ft/collect.py
```

Run ETL on HF dataset (or collected screenshots if available):

```bash
uv run ft/etl.py
```

or

```bash
uv run modal run ft/etl.py
```

Train the model:

```bash
uv run torchrun --standalone --nproc_per_node=<n-gpus> ft/train.py
```

or

```bash
uv run modal run ft/train_modal.py
```
