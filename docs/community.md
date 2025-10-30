# Community

Fuse thrives on contributions from practitioners who care about readable infrastructure. This page outlines expectations for collaboration.

## Contributing workflow

!!! note "Conventional commits"
    Use the `type: summary` format (`feat: add numpy evaluator guard`). Keep each PR focused on one logical change and include reproduction steps in the description.

1. Fork the repository and create a feature branch.
2. Install the development extras (`pip install -e ".[dev]"`).
3. Run the full CI suite locally before submitting a PR.

```bash
ruff check src tests
mypy
pytest -q
mkdocs build --strict
```

## Development tips

- Prefer explicit imports between modules to keep dependency flow obvious.
- Thread `ExecutionConfig` and `RuntimePolicies` through execution paths when adding new features.
- Examples live under `examples/` and pair with `.fuse` programs—use them as fixtures for tests.

## Documentation

We use [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) with mkdocstrings.

=== "Preview locally"

    ```bash
    mkdocs serve
    ```

=== "Strict build"

    ```bash
    mkdocs build --strict
    ```

`mike` powers versioned deployments. Analytics keys live in `mkdocs.yml` (override them per environment).
