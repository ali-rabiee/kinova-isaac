"""vla_lab: minimal VLA-TTC training/eval package for the kinova-isaac repo.

This package is intentionally self-contained. All implementation files live
under `vla_lab/` and reuse the rest of the repository (data collection,
environments, controllers, ...) only via standard imports - no edits to those
packages are required.

Public modules:
- vla_lab.dataset       : torch.utils.data.Dataset over `vla_v1` collect_data logs
- vla_lab.models        : `TinyVLA` baseline model
- vla_lab.losses        : optional DINOv2 feature-alignment loss
- vla_lab.ttc           : K-sample inference pipeline (Phase-1 fallback scorer)
- vla_lab.train         : training entrypoint (no Isaac dependency)
- vla_lab.eval_isaaclab : IsaacLab evaluation entrypoint
- vla_lab.dryrun        : VRAM/latency dry-fit (no Isaac dependency)
- vla_lab.inspect_data  : sanity-check tool for collected sessions
"""

__version__ = "0.0.1"
