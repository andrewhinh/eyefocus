# modeldemo

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
uv run pre-commit install
export PYTHONPATH=.
echo "export PYTHONPATH=.:$PYTHONPATH" >> ~/.bashrc
```

## Development

Check out the following docs:

- [uv](https://docs.astral.sh/uv/getting-started/features/#projects)
- [modal](https://modal.com/docs)
- [ruff](https://docs.astral.sh/ruff/tutorial/)
