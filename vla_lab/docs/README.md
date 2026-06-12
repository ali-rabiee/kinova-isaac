# vla_lab/docs — historical reports & design notes

These documents are kept for provenance; paths inside them may predate the 2026-06-11
reorganization (e.g. they may refer to files now under `vla_lab/docs/` or
`vla_lab/scripts/legacy/`).

| File | What it is |
| --- | --- |
| `EVAL_DEBUG_REPORT.md` | 2026-06-10 postmortem of the "robot flailing at eval" bug: root causes (unpowered settle, camera spawn failure, gripper-label bug, 15 Hz vs 5 Hz action-rate mismatch, start-pose mismatch), fixes, and verification results. Read this before touching `eval_isaaclab.py`. |
| `FABLE_INSTRUCTIONS.md` | The task brief that produced the eval debug report (kept for context). |
| `new_changes.md` | Earlier design notes for the eval protocol (Wilson CIs, latency reporting, occlusion axes) and SmolVLA comparison plan. |

Current documentation lives in:

- `vla_lab/README.md` — overview + commands for collection / training / evaluation.
- `vla_lab/data_collection_guide.md` — the authoritative data-collection reference
  (pipeline anatomy, data/action contract, 2026-06-11 bug fixes, final-dataset procedure).
