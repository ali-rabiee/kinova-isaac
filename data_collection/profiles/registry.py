from __future__ import annotations

from typing import Dict

from .spec import ProfileSpec


def get_profiles() -> Dict[str, ProfileSpec]:
    # Keep CLI startup lightweight: do NOT import profile modules here.
    # Import them only when their functions are actually invoked.

    def _ticks_v0_add_args(parser) -> None:
        from . import ticks_v0

        ticks_v0.add_cli_args(parser)

    def _ticks_v0_run(args) -> int:
        from . import ticks_v0

        return ticks_v0.run(args)

    def _vla_v0_add_args(parser) -> None:
        from . import vla_v0

        vla_v0.add_cli_args(parser)

    def _vla_v0_run(args) -> int:
        from . import vla_v0

        return vla_v0.run(args)

    def _vla_v1_add_args(parser) -> None:
        from . import vla_v1

        vla_v1.add_cli_args(parser)

    def _vla_v1_run(args) -> int:
        from . import vla_v1

        return vla_v1.run(args)

    def _brace_v1_add_args(parser) -> None:
        from . import brace_v1

        brace_v1.add_cli_args(parser)

    def _brace_v1_run(args) -> int:
        from . import brace_v1

        return brace_v1.run(args)

    return {
        "ticks_v0": ProfileSpec(name="ticks_v0", add_cli_args=_ticks_v0_add_args, run=_ticks_v0_run),
        "vla_v0": ProfileSpec(name="vla_v0", add_cli_args=_vla_v0_add_args, run=_vla_v0_run),
        "vla_v1": ProfileSpec(name="vla_v1", add_cli_args=_vla_v1_add_args, run=_vla_v1_run),
        "brace_v1": ProfileSpec(name="brace_v1", add_cli_args=_brace_v1_add_args, run=_brace_v1_run),
    }


