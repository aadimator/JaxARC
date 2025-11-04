#!/usr/bin/env bash
# Export pixi docs environment and fix known issues for ReadTheDocs compatibility

set -e  # Exit on error

echo "Exporting pixi docs environment..."
pixi project export conda-environment --environment docs > docs/environment.yml

echo "Fixing ReadTheDocs compatibility issues..."

# Fix 1: Invalid pip syntax: jax[cpu]* -> jax[cpu]
sed -i.bak 's/jax\[cpu\]\*/jax[cpu]/' docs/environment.yml

# Fix 2: Relative path for editable install: -e . -> -e ..
# (environment.yml is in docs/, so we need .. to get to project root)
sed -i.bak 's|- -e \.|- -e ..|' docs/environment.yml

# Remove backup file
rm -f docs/environment.yml.bak

echo "✅ docs/environment.yml exported and fixed successfully!"
