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

# Fix 3: Add chex if not present (needed by JaxARC but not in conda export)
if ! grep -q "chex" docs/environment.yml; then
  # Find the line with jax[cpu] and add chex after it
  sed -i.bak '/jax\[cpu\]/a\
  - chex
' docs/environment.yml
fi

# Fix 4: Ensure -e .. comes AFTER jax[cpu] and chex
# This ensures dependencies are installed before the package
awk '
  BEGIN { in_pip = 0; lines = ""; editable = "" }
  /^- pip:/ { in_pip = 1; print; next }
  in_pip && /^  - -e/ { editable = $0; next }
  in_pip && /^[^ -]/ {
    in_pip = 0
    if (editable != "") print editable
    print
    next
  }
  in_pip { print; next }
  { print }
  END { if (in_pip && editable != "") print editable }
' docs/environment.yml > docs/environment.yml.tmp && mv docs/environment.yml.tmp docs/environment.yml

# Fix 5: Remove any blank lines in the pip section
sed -i.bak '/^- pip:/,/^[^ ]/{/^$/d;}' docs/environment.yml

# Remove backup file
rm -f docs/environment.yml.bak

echo "✅ docs/environment.yml exported and fixed successfully!"
echo ""
echo "Installation order:"
grep -A 20 "^- pip:" docs/environment.yml | grep "^  -"
