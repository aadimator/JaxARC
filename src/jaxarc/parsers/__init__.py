"""ARC dataset parsers for loading and processing task data.

This module provides parsers for different ARC dataset formats, including
ARC-AGI, ConceptARC, and MiniARC. All parsers inherit from a common base
class to ensure consistent functionality and reduce code duplication.

Examples:
    ```python
    from jaxarc.parsers import ArcAgiParser, ConceptArcParser, MiniArcParser

    # Load ARC-AGI dataset
    parser = ArcAgiParser(data_dir="data/arc-prize-2024")
    tasks = parser.load_tasks(split="training")

    # Load ConceptARC dataset
    concept_parser = ConceptArcParser(data_dir="data/ConceptARC")
    concept_tasks = concept_parser.load_tasks(split="corpus")

    # Load MiniARC dataset
    mini_parser = MiniArcParser(data_dir="data/MiniARC")
    mini_tasks = mini_parser.load_tasks(split="training")
    ```
"""

from __future__ import annotations

# Specific dataset parsers
from .arc_agi import ArcAgiParser

# Base parser class
from .base_parser import ArcDataParserBase
from .concept_arc import ConceptArcParser
from .mini_arc import MiniArcParser

__all__ = [
    "ArcAgiParser",
    "ArcDataParserBase",
    "ConceptArcParser",
    "MiniArcParser",
]
