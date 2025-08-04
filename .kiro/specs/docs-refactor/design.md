# Documentation Refactor Design

## Overview

This design outlines a comprehensive refactor of the JaxARC documentation to create a streamlined, user-focused structure that eliminates redundancy, improves discoverability, and aligns with the current codebase. The new structure will consolidate information into logical, concise documents while maintaining comprehensive coverage of all features.

## Architecture

### New Documentation Structure

```
docs/
├── README.md                    # Main project overview (updated)
├── getting-started.md          # Quick start guide with common use cases
├── datasets.md                 # All dataset information consolidated
├── configuration.md            # Complete configuration guide
├── api-reference.md            # Comprehensive API documentation
└── examples/                   # Practical examples directory
    ├── basic-usage.md
    ├── conceptarc-examples.md
    ├── miniarc-examples.md
    └── advanced-patterns.md
```

### Removed/Consolidated Files

- `CONFIG_API_README.md` → Merged into `configuration.md`
- `parser_usage.md` → Merged into `datasets.md`
- `KAGGLE_TO_GITHUB_MIGRATION.md` → Removed (migration complete)
- `testing_guide.md` → Condensed into `api-reference.md`
- `TROUBLESHOOTING.md` → Integrated contextually into relevant sections

## Components and Interfaces

### 1. Main README.md (Updated)

**Purpose**: Project overview, installation, and quick navigation
**Length**: ~200 lines (reduced from current 400+)
**Key Sections**:
- Brief project description
- Installation (simplified)
- Quick start (30-second example)
- Dataset overview (summary table)
- Navigation to detailed docs
- Contributing and support

### 2. getting-started.md (New)

**Purpose**: Comprehensive getting started guide for new users
**Length**: ~150 lines
**Key Sections**:
- Installation and setup
- Download your first dataset
- Run your first example
- Common patterns and next steps
- Links to detailed documentation

### 3. datasets.md (Consolidated)

**Purpose**: All dataset-related information in one place
**Length**: ~300 lines
**Key Sections**:
- Dataset overview and comparison table
- Download instructions (unified)
- Parser usage for each dataset type
- Configuration examples
- Performance characteristics
- Troubleshooting (contextual)

### 4. configuration.md (Consolidated)

**Purpose**: Complete configuration system documentation
**Length**: ~250 lines
**Key Sections**:
- Configuration overview and philosophy
- Factory functions and presets
- Hydra integration
- Action formats
- Environment types
- Advanced configuration patterns
- Troubleshooting (contextual)

### 5. api-reference.md (Streamlined)

**Purpose**: Complete API documentation with practical focus
**Length**: ~400 lines
**Key Sections**:
- Core classes and functions
- Parser API reference
- Environment API reference
- Data types and structures
- JAX compatibility notes
- Testing patterns (condensed)

### 6. examples/ Directory (New Structure)

**Purpose**: Practical, working examples organized by use case
**Files**:
- `basic-usage.md`: Core functionality examples
- `conceptarc-examples.md`: ConceptARC-specific patterns
- `miniarc-examples.md`: MiniARC rapid prototyping
- `advanced-patterns.md`: JAX transformations, batch processing

## Data Models

### Documentation Metadata

Each document will include frontmatter for consistency:

```yaml
---
title: "Document Title"
description: "Brief description"
last_updated: "2025-01-16"
related_docs: ["doc1.md", "doc2.md"]
examples: ["example1.py", "example2.py"]
---
```

### Cross-Reference System

Standardized linking patterns:
- `[Configuration Guide](configuration.md)` for internal links
- `[Example: Basic Usage](examples/basic-usage.md#section)` for example links
- `[API: ArcAgiParser](api-reference.md#arcagiparser)` for API references

### Code Example Standards

All code examples will follow this pattern:

```python
# Brief description of what this example demonstrates
import jax
from jaxarc.parsers import ArcAgiParser

# Setup (minimal, working)
config = create_config()
parser = ArcAgiParser(config)

# Main functionality
result = parser.get_random_task(jax.random.PRNGKey(42))

# Expected output or next steps
print(f"Task loaded with {result.num_train_pairs} training pairs")
```

## Error Handling

### Contextual Troubleshooting

Instead of separate troubleshooting documents, each section will include:

1. **Common Issues** subsection with 2-3 most frequent problems
2. **Quick Fixes** with one-line solutions
3. **See Also** references to related sections

Example pattern:
```markdown
### Common Issues

**"Dataset not found"**: Run `python scripts/download_dataset.py <dataset-name>`
**"Legacy format detected"**: Update config to use `path` instead of `challenges`/`solutions`
**Need help?**: See [Configuration Guide](configuration.md#troubleshooting) for more details
```

### Error Message Alignment

Documentation will reference actual error messages from the codebase to ensure accuracy.

## Testing Strategy

### Documentation Testing

1. **Link Validation**: Automated checking of all internal links
2. **Code Example Testing**: All code examples must be runnable
3. **Consistency Checking**: Automated validation of naming conventions
4. **Freshness Validation**: Automated detection of outdated information

### Content Quality Metrics

- **Readability**: Target 8th-grade reading level using automated tools
- **Completeness**: All public APIs documented
- **Accuracy**: All examples tested against current codebase
- **Conciseness**: Maximum line limits enforced per document type

## Migration Strategy

### Phase 1: Content Consolidation

1. Extract and merge content from existing documents
2. Remove duplicate information
3. Update all examples to current format
4. Remove outdated references

### Phase 2: Structure Implementation

1. Create new document structure
2. Implement cross-reference system
3. Add contextual troubleshooting
4. Create examples directory

### Phase 3: Quality Assurance

1. Test all code examples
2. Validate all links
3. Review for consistency
4. Optimize for readability

### Phase 4: Cleanup

1. Remove old documentation files
2. Update all references to new structure
3. Update CI/CD to validate new structure
4. Update contributing guidelines

## Design Decisions and Rationales

### Single-Page Approach for Major Topics

**Decision**: Consolidate related information into single, comprehensive documents
**Rationale**: Reduces navigation overhead and provides complete context in one place

### Contextual Troubleshooting

**Decision**: Embed troubleshooting information within relevant sections
**Rationale**: Users encounter problems in context, so solutions should be contextual too

### Examples-First Approach

**Decision**: Lead with practical examples, follow with explanation
**Rationale**: Developers prefer working code they can modify over abstract explanations

### Aggressive Deduplication

**Decision**: Remove all duplicate information, even if it means more cross-references
**Rationale**: Maintenance burden of duplicate information outweighs navigation convenience

### GitHub-First Documentation

**Decision**: Remove all Kaggle-related information and migration guides
**Rationale**: Migration period is over; maintaining legacy information creates confusion

## Performance Considerations

### Document Size Optimization

- Target maximum 400 lines per document
- Use collapsible sections for optional details
- Minimize embedded images and large code blocks
- Optimize for fast loading and searching

### Search Optimization

- Use consistent terminology throughout
- Include relevant keywords in headers
- Structure content for easy scanning
- Provide clear section hierarchies

### Mobile Responsiveness

- Ensure all documents are readable on mobile devices
- Use responsive tables and code blocks
- Minimize horizontal scrolling
- Test on various screen sizes

## Integration Points

### CI/CD Integration

- Automated link checking on PR
- Code example testing in CI pipeline
- Documentation freshness validation
- Consistency checking automation

### IDE Integration

- Ensure documentation works well with IDE preview
- Optimize for VS Code markdown preview
- Support for jump-to-definition from examples
- Integration with language servers

### Website Integration

- Structure compatible with static site generators
- SEO optimization for search engines
- Social media preview optimization
- Analytics integration for usage tracking