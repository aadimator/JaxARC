# JaxARC

This project uses [Pixi](https://pixi.js.org) to manage the project structure
and build process, and [Jupyter Book](https://jupyterbook.org) for
documentation. Below are the steps to get started with the project.

```bash
# Clone the repository
git clone git@github.com:aadimator/jaxarc.git
cd jaxarc

# Install Pixi globally
curl -fsSL https://pixi.sh/install.sh | sh

# Install project dependencies
pixi install
```

## Useful Pixi Commands

Here are some useful Pixi commands to manage the project:

```bash
# Add dependencies
pixi add <package_name>

# Remove dependencies
pixi remove <package_name>

# Run linting
pixi run lint

# Serve the documentation
pixi run docs-serve

# Run tests
pixi run test
```

For detailed documentation on how the project was set up from scractch, refer to
the [](./setup.md). For more information on how to use Pixi, refer to the
[Pixi Documentation](https://pixi.js.org/docs/).
