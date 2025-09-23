from __future__ import annotations

import types

import pytest

from jaxarc.configs.main_config import JaxArcConfig
from jaxarc.registration import available_task_ids, make
from jaxarc.utils.core import get_config

# Optional hydra import (not used if unavailable)
try:  # pragma: no cover - import resolution varies by env
    import hydra.core.global_hydra as gh  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    gh = None  # type: ignore[assignment]


@pytest.mark.fast
def test_make_mini_overrides_default_agi_incoming_config():
    """Given a config that defaults to ARC-AGI, make("Mini-<id>") must switch to MiniARC.

    This ensures dataset selection honors the dataset key and doesn't stick with defaults.
    """
    # Default Hydra config uses ARC-AGI-2 per conf/config.yaml
    cfg = JaxArcConfig.from_hydra(get_config())
    assert "ARC-AGI" in cfg.dataset.dataset_name

    # Use a single MiniARC task to keep test fast
    ids = available_task_ids("Mini", config=cfg, auto_download=False)
    assert isinstance(ids, list)
    assert len(ids) > 0, "MiniARC should provide some task ids"

    task_id = ids[0]
    env, params = make(f"Mini-{task_id}", config=cfg, auto_download=False)

    # After make, EnvParams must point to MiniARC (both on env and returned params)
    assert env.params.dataset.dataset_name == "MiniARC"
    assert "MiniArcParser" in env.params.dataset.parser_entry_point
    assert params.dataset.dataset_name == "MiniARC"
    assert "MiniArcParser" in params.dataset.parser_entry_point


@pytest.mark.fast
def test_make_mini_when_hydra_already_initialized(monkeypatch: pytest.MonkeyPatch):
    """Simulate external Hydra app already initialized; make should not re-init Hydra.

    We monkeypatch GlobalHydra.instance().is_initialized() -> True and ensure make works
    without raising and still selects MiniARC.
    """
    if gh is None:
        pytest.skip("hydra not available in test environment")

    # Build a dummy instance object with is_initialized() == True
    dummy = types.SimpleNamespace(is_initialized=lambda: True)
    monkeypatch.setattr(gh.GlobalHydra, "instance", lambda: dummy)

    env, params = make("Mini", auto_download=False)
    assert env.params.dataset.dataset_name == "MiniARC"
    assert "MiniArcParser" in env.params.dataset.parser_entry_point
    assert params.dataset.dataset_name == "MiniARC"
    assert "MiniArcParser" in params.dataset.parser_entry_point


@pytest.mark.fast
def test_available_task_ids_mini_nonempty():
    ids = available_task_ids("Mini", auto_download=False)
    assert isinstance(ids, list)
    assert len(ids) > 0
