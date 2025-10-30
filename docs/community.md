# Community

## Contributing

- Use Conventional Commits (e.g., `feat: add numpy evaluator guard`)
- Keep PRs focused on one logical change
- Include reproduction/verification steps and sample outputs when relevant

## Development

```bash
pip install -e ".[dev]"
ruff check src tests
mypy --strict -p fuse
pytest -q
```

## Docs

We use MkDocs Material with mkdocstrings. Build locally:

```bash
pip install -e ".[dev]"
mkdocs build --strict
mkdocs serve
```

Versioned deployments can be handled with `mike`. Configure analytics and versioning in `mkdocs.yml` or an override file.

