# Installation

This guide will walk you through installing JaxARC on your system.

## Quick Install

### Using Pixi (Recommended)

[Pixi](https://pixi.sh) is a fast, cross-platform package manager that makes
environment management easy.

```bash
# Install Pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Add JaxARC to your project
pixi add jaxarc

# Or create a new project with JaxARC
pixi init my-arc-project
cd my-arc-project
pixi add jaxarc
```

### Using pip

```bash
# Install in your current environment
pip install jaxarc

# Or create a virtual environment first (recommended)
python -m venv jaxarc-env
source jaxarc-env/bin/activate  # On Windows: jaxarc-env\Scripts\activate
pip install jaxarc
```

## GPU Support

By default, JaxARC installs the CPU version of JAX. To enable GPU acceleration:

### CUDA Support (NVIDIA GPUs)

```bash
# Install JAX with CUDA support
pip install --upgrade "jax[cuda12]"  # For CUDA 12
# or
pip install --upgrade "jax[cuda11]"  # For CUDA 11
```

### TPU Support (Google Cloud)

```bash
# Install JAX with TPU support
pip install --upgrade "jax[tpu]"
```

For more details on JAX installation, see the
[official JAX installation guide](https://github.com/google/jax#installation).

## Verify Installation

After installation, verify that JaxARC is working correctly:

```python
import jax
import jaxarc

# Check versions
print(f"JaxARC version: {jaxarc.__version__}")
print(f"JAX version: {jax.__version__}")

# Verify JAX can see your devices
print(f"JAX devices: {jax.devices()}")

# Create a simple environment to test
env, env_params = jaxarc.make(
    "Mini-Most_Common_color_l6ab0lf3xztbyxsu3p", auto_download=True
)
print("Successfully created environment")

# Test reset
key = jax.random.PRNGKey(0)
state, timestep = env.reset(key, env_params=env_params)
print("Environment reset successful")
```

**Expected output:**

```
JaxARC version: 0.1.0
JAX version: 0.6.x
JAX devices: [CpuDevice(id=0)]  # or GpuDevice/TpuDevice if available
Successfully created environment
Environment reset successful
Observation shape: (3, 5, 5)
```

## Troubleshooting

### Import Error: No module named 'jaxarc'

**Cause**: JaxARC is not installed in the current Python environment.

**Solution**:

```bash
# Verify you're in the right environment
which python  # Should show your venv or pixi environment

# Reinstall JaxARC
pip install jaxarc
```

### JAX installation issues

**Cause**: JAX has specific platform requirements.

**Solution**: Visit the
[JAX installation guide](https://github.com/google/jax#installation) for
platform-specific instructions.

### GPU not detected

**Cause**: JAX is using the CPU version or CUDA drivers are not properly
installed.

**Solution**:

1. Check CUDA installation: `nvidia-smi` should show your GPU
2. Install JAX with CUDA support (see GPU Support section above)
3. Verify with `jax.devices()` - should show `GpuDevice`

## Next Steps

Now that JaxARC is installed, continue to:

- **[Quick Start](quickstart.md)** - Learn basic environment usage
- **[First Example](first-example.md)** - See a complete working example

## Additional Resources

- **[JAX Documentation](https://jax.readthedocs.io/)** - Learn more about JAX
- **[Pixi Documentation](https://pixi.sh)** - Learn more about Pixi
