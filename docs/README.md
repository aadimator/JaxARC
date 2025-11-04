# JaxARC Documentation

This directory contains the source files for JaxARC's documentation, built with
Sphinx and hosted on ReadTheDocs.

## Building Locally

```bash
# Build the documentation
pixi run -e docs docs-build

# Serve locally at http://localhost:8000
pixi run -e docs docs-serve

# Clean build artifacts
pixi run -e docs docs-clean
```

## Files

- **`conf.py`**: Sphinx configuration
- **`index.md`**: Documentation homepage
- **`environment.yml`**: Conda environment for ReadTheDocs (auto-generated from
  pixi)
- **`getting-started/`**: Installation and quickstart guides
- **`tutorials/`**: Step-by-step tutorials
- **`api/`**: API reference documentation
- **`_static/`**: Static assets (images, CSS)
- **`_build/`**: Generated HTML (gitignored)

## Updating the ReadTheDocs Environment

If you modify the docs dependencies in `pixi.toml`, regenerate the environment
file:

```bash
pixi run -e docs docs-export
```

This exports the pixi `docs` environment to `docs/environment.yml` and
automatically fixes ReadTheDocs compatibility issues:

- Removes invalid `*` from `jax[cpu]*`
- Fixes relative path for editable install (`-e ..` to reference project root)

## Writing Documentation

Documentation is written in MyST Markdown (`.md` files) and Jupyter Notebooks
(`.ipynb` files).

**MyST Markdown Features:**

- Standard Markdown syntax
- Directives: ` ```{directive} ` for special blocks
- Roles: `` {role}`text` `` for inline markup
- Cross-references: `` {doc}`path/to/file` ``

See [MyST documentation](https://myst-parser.readthedocs.io/) for more details.

## ReadTheDocs

The documentation is automatically built and published on
[ReadTheDocs](https://jaxarc.readthedocs.io/) when changes are pushed to the
main branch.

Configuration: `.readthedocs.yaml` in the repository root.
