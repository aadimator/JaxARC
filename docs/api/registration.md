# Registration API

The registration system manages environment creation, task IDs, and dataset
organization.

JaxARC uses a registration system to manage environments and tasks. Each
environment is registered with a unique ID that can be used with `make()`.

## Module Contents

```{eval-rst}
.. automodule:: jaxarc.registration
   :members:
   :undoc-members:
   :show-inheritance:
```

## Task ID Format

Task IDs follow the pattern: `Dataset-TaskName_taskId`

Examples:

- `Mini-Most_Common_color_l6ab0lf3xztbyxsu3p`
- `ConceptARC-denoising_0c9aba6e`
- `ARC-007bbfb7`
