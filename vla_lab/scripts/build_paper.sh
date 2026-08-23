#!/usr/bin/env bash
# ★ Build the manuscript from the current results. Never build it any other way.
#
#   ./vla_lab/scripts/build_paper.sh
#   ALLOW_STALE=1 ./vla_lab/scripts/build_paper.sh    # build anyway when a stage has not run
#
# Refreshing figures and tables is part of the build rather than a step someone remembers, because
# the failure mode it prevents is silent: a re-run changes a result, the caption is updated by
# hand, the plot is not, and the paper disagrees with itself in a way that compiles cleanly.
set -uo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${REPO_ROOT}"
PAPER="vla_lab/paper/hri2027_carryover_vla"

"${REPO_ROOT}/vla_lab/scripts/sync_paper_figures.sh"
SYNC_RC=$?
if [[ ${SYNC_RC} -ne 0 && "${ALLOW_STALE:-0}" != "1" ]]; then
  echo
  echo "refusing to build with missing inputs. Re-run the stage above, or set ALLOW_STALE=1."
  exit ${SYNC_RC}
fi

cd "${PAPER}"
# Source-level checks first: these catch the two failure modes that compile cleanly and read
# wrong (see lint.py), so they are worth a second before spending a minute on pdflatex.
if ! python lint.py main.tex; then
  echo "source lint failed; fix before building."
  exit 1
fi
# Every number the manuscript quotes from a result file, re-derived from that file and looked for
# in the source. Generated tables cannot drift; prose can, and this is where it would.
python check_numbers.py

# latexmk's rerun heuristic does not always notice that an \input-ed table changed a label, so a
# freshly generated table can leave `Reference undefined' behind on an otherwise clean build.
# Re-run until the references settle rather than trusting one pass.
latexmk -pdf -interaction=nonstopmode main.tex > /dev/null 2>&1
for _ in 1 2; do
  n=$(grep "LaTeX Warning: Reference" main.log 2>/dev/null | wc -l | tr -d " ")
  [[ "${n}" == "0" ]] && break
  pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
done
echo
# Count with grep|wc rather than `grep -c`: this box's grep is ugrep, whose -c prints nothing
# (not "0") when there are no matches, which silently turned a clean build into a failure.
UNDEF=$(grep "LaTeX Warning: Reference" main.log 2>/dev/null | wc -l | tr -d " ")
CITE=$(grep "Citation.*undefined" main.log 2>/dev/null | wc -l | tr -d " ")
PAGES=$(pdfinfo main.pdf 2>/dev/null | awk '/^Pages:/{print $2}')
echo "pages: ${PAGES:-<none>}   undefined refs: ${UNDEF}   undefined citations: ${CITE}"
if [[ -z "${PAGES}" ]]; then
  echo "BUILD FAILED -- see ${PAPER}/main.log"
  exit 1
fi
if [[ "${UNDEF}" != "0" || "${CITE}" != "0" ]]; then
  echo "build produced dangling references; fix before circulating."
  exit 1
fi
echo "ok: ${PAPER}/main.pdf"
