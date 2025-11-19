"""
Visualization wrapper for JaxARC environments.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from stoa.core_wrappers.wrapper import Wrapper

from jaxarc.state import State
from jaxarc.types import EnvParams, Grid
from jaxarc.utils.task_manager import get_task_id_globally
from jaxarc.utils.visualization.core import detect_changed_cells
from jaxarc.utils.visualization.rl_display import (
    draw_rl_step_svg_enhanced,
    get_operation_display_name,
)


class StepVisualizationWrapper(Wrapper):
    """
    Wrapper that enables detailed step visualization by tracking transition history.

    Enables `env.render(mode="detailed")` which returns a rich SVG of the last transition.
    """

    def __init__(self, env):
        super().__init__(env)
        self._last_transition: tuple[Any, Any, Any, Any] | None = None

    def reset(  # type: ignore[override]
        self, key, env_params: EnvParams | None = None
    ) -> tuple[Any, Any]:
        # Cast env to Any to avoid type checking against stoa.Environment
        env: Any = self._env
        state, timestep = env.reset(key, env_params)
        self._last_transition = None
        return state, timestep

    def step(  # type: ignore[override]
        self, state: State, action: Any, env_params: EnvParams | None = None
    ) -> tuple[Any, Any]:
        # Cast env to Any to avoid type checking against stoa.Environment
        env: Any = self._env
        next_state, timestep = env.step(state, action, env_params)

        # Cache the transition
        self._last_transition = (state, action, next_state, timestep)

        return next_state, timestep

    def render(self, state: State, mode: str | None = None) -> Any:  # type: ignore[override]
        if mode == "detailed":
            return self._render_detailed()
        # Cast env to Any to avoid type checking against stoa.Environment
        env: Any = self._env
        return env.render(state, mode)

    def _render_detailed(self) -> str:
        if self._last_transition is None:
            return (
                '<svg width="200" height="50">'
                '<text x="10" y="30" font-family="sans-serif" fill="red">'
                "No transition available (call step() first)"
                "</text></svg>"
            )

        prev_state, action, next_state, timestep = self._last_transition

        # Construct Grid objects (using jnp.array to satisfy Equinox checks)
        before_grid = Grid(
            data=jnp.array(prev_state.working_grid),
            mask=jnp.array(prev_state.working_grid_mask),
        )
        after_grid = Grid(
            data=jnp.array(next_state.working_grid),
            mask=jnp.array(next_state.working_grid_mask),
        )

        # Extract info from timestep extras
        info = timestep.extras if timestep.extras is not None else {}

        # Get operation name
        op_id = -1
        if "operation_id" in info:
            # Handle JAX array scalar
            val = info["operation_id"]
            op_id = int(val)
        elif hasattr(action, "operation"):
            val = action.operation
            op_id = int(val)
        elif isinstance(action, dict) and "operation" in action:
            val = action["operation"]
            op_id = int(val)

        op_name = get_operation_display_name(op_id) if op_id >= 0 else "Unknown"

        # Calculate changed cells
        changed_cells = detect_changed_cells(before_grid, after_grid)

        # Task ID
        task_idx = int(next_state.task_idx)
        task_name = get_task_id_globally(task_idx)
        task_id_str = task_name if task_name else f"Task {task_idx}"

        return draw_rl_step_svg_enhanced(
            before_grid=before_grid,
            after_grid=after_grid,
            action=action,
            reward=float(timestep.reward),
            info=info,
            step_num=int(next_state.step_count),
            operation_name=op_name,
            changed_cells=changed_cells,
            task_id=task_id_str,
            task_pair_index=int(next_state.pair_idx),
        )
