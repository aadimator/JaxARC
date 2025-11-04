# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from __future__ import annotations

import os
import sys

# Add the project root to sys.path for imports
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "JaxARC"
# copyright = "2025, JaxARC Contributors"
author = "Aadam"
version = "0.1.0"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # API reference from docstrings
    "sphinx.ext.napoleon",  # Google-style docstrings
    "sphinx.ext.viewcode",  # Link to source code
    "sphinx.ext.intersphinx",  # Link to other projects
    "myst_nb",  # Jupyter notebook support (replaces myst_parser)
    "sphinx_copybutton",  # Copy code buttons
    "sphinx_design",  # Grid cards and other design elements
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST Parser configuration -----------------------------------------------
# https://myst-parser.readthedocs.io/en/latest/configuration.html

myst_enable_extensions = [
    "colon_fence",  # ::: fences
    "deflist",  # Definition lists
    "tasklist",  # Task lists
]

# -- MyST-NB configuration ---------------------------------------------------
# https://myst-nb.readthedocs.io/en/latest/configuration.html

# Execution settings
nb_execution_mode = "auto"  # Cache outputs to avoid re-running unchanged cells
nb_execution_timeout = 300  # 5 minutes per cell
nb_execution_allow_errors = False  # Stop on errors during build

# Cache location
nb_execution_cache_path = "_build/.jupyter_cache"

# Output formatting
nb_output_stderr = "show"  # Show stderr in outputs
nb_merge_streams = True  # Merge stdout and stderr

# Kernel settings
nb_kernel_rgx_aliases = {
    "python3": "python",
}

# File extensions to parse as notebooks
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = "JaxARC Documentation"
html_static_path = ["_static"]

# -- Autodoc configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# -- Intersphinx configuration -----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Link checking configuration ---------------------------------------------

linkcheck_ignore = [
    r"http://localhost:\d+/",  # Local development servers
]
