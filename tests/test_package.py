from __future__ import annotations

import importlib.metadata

import jaxarc as m


def test_version():
    assert importlib.metadata.version("jaxarc") == m.__version__
