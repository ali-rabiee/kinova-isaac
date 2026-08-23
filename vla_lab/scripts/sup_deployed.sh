#!/usr/bin/env bash
# ★ Closed-loop evaluation: put trained checkpoints into the session as the robot's ear.
#
# The architecture sweep scores checkpoints on held-out dialogues. This scores them where it
# counts -- driving the real schedulers, belief module and estimand -- and it can fail in ways
# the offline metric cannot see (a model that grounds well on average but abstains exactly in
# the crossover band starves the estimand while looking healthy).
#
#   ./vla_lab/scripts/sup_deployed.sh                      # every tiny context mode
#   CKPTS="vla_lab/results/models_isaac_smolvla/smolvla__text" ./vla_lab/scripts/sup_deployed.sh
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${REPO_ROOT}"
CKPTS="${CKPTS:-vla_lab/results/models_isaac/tiny__none vla_lab/results/models_isaac/tiny__token vla_lab/results/models_isaac/tiny__film}"
exec python -m vla_lab.supervisory.run_deployed \
  --checkpoint ${CKPTS} \
  --supervisors "${SUPERVISORS:-24}" \
  --device "${DEVICE:-cpu}" \
  --out "${OUT:-vla_lab/results/deployed}" "$@"
