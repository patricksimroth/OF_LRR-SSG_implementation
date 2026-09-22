#!/bin/bash
# run_postpro.sh - redraw a study's figures from its archived raw data, using
# the PERMANENT scripts in <case>/_postProcessing (not the copies in run/).
#
#   ./run_postpro.sh results/turbulentFlatPlate/compare_approaches
#   ./run_postpro.sh results/turbulentFlatPlate/compare_approaches cf_validation profile_validation
#
# No solver, no meshing: every script reads <study>/_rundata/ (written by
# run_study.sh) through PP_RUN_CASE_DIR and writes back into <study>/.  Same
# order as the drivers - postpro first, the rest alphabetically, dashboard
# last - and `_`-prefixed helpers are never executed.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/../.venv/bin/python3"

if [ $# -lt 1 ]; then
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi

STUDY_DIR="$(cd "$1" 2>/dev/null && pwd)" || { echo "[error] no such directory: $1"; exit 2; }
shift
RUNDATA="${STUDY_DIR}/_rundata"
STUDY="$(basename "${STUDY_DIR}")"
CASE="$(basename "$(dirname "${STUDY_DIR}")")"
PP_SRC="${SCRIPT_DIR}/${CASE}/_postProcessing"

[ -d "${RUNDATA}" ] || { echo "[error] ${RUNDATA} missing - was this folder written by run_study.sh?"; exit 2; }
[ -d "${PP_SRC}" ]  || { echo "[error] no post-processing scripts at ${PP_SRC}"; exit 2; }
[ -x "${VENV_PYTHON}" ] || { echo "[error] venv python not found: ${VENV_PYTHON}"; exit 2; }

# The scripts derive results/<case>/<study> from the REPO root and the study
# name in the archived manifest, not from this argument.  If either disagrees
# with the folder given here, the figures would silently land elsewhere.
EXPECTED="${SCRIPT_DIR}/results/${CASE}/${STUDY}"
if [ "${STUDY_DIR}" != "${EXPECTED}" ]; then
    echo "[error] ${STUDY_DIR}"
    echo "        is not ${EXPECTED}; the scripts would write there instead."
    exit 2
fi
MANIFEST_STUDY="$("${VENV_PYTHON}" -c "import json,sys; print(json.load(open(sys.argv[1]))['study'])" \
                  "${RUNDATA}/study_manifest.json" 2>/dev/null)"
if [ "${MANIFEST_STUDY}" != "${STUDY}" ]; then
    echo "[error] archived manifest says study '${MANIFEST_STUDY}', folder is '${STUDY}'"
    exit 2
fi

# Script order: explicit subset if given, else everything the drivers would run.
if [ $# -gt 0 ]; then
    NAMES=()
    for s in "$@"; do NAMES+=("${s%.py}.py"); done
else
    mapfile -t ALL < <(cd "${PP_SRC}" && ls -1 *.py | grep -v '^_' | sort)
    NAMES=()
    for n in "${ALL[@]}"; do [ "$n" = "postpro.py" ] && NAMES+=("$n"); done
    for n in "${ALL[@]}"; do [ "$n" != "postpro.py" ] && [ "$n" != "dashboard.py" ] && NAMES+=("$n"); done
    for n in "${ALL[@]}"; do [ "$n" = "dashboard.py" ] && NAMES+=("$n"); done
fi

echo "=========================================="
echo "study   : ${CASE}/${STUDY}  (redraw from archive, no solve)"
echo "scripts : ${PP_SRC}"
echo "data    : ${RUNDATA}"
echo "=========================================="

export PP_RUN_CASE_DIR="${RUNDATA}"
FAILED=()
for name in "${NAMES[@]}"; do
    if [ ! -f "${PP_SRC}/${name}" ]; then
        echo "  [FAILED] ${name}: not in ${PP_SRC}"
        FAILED+=("${name}"); continue
    fi
    log="${RUNDATA}/${name%.py}.log"
    printf "  -> %-24s" "${name}"
    if (cd "${PP_SRC}" && "${VENV_PYTHON}" "${name}") > "${log}" 2>&1; then
        echo "ok"
    else
        echo "FAILED (log: ${log})"
        FAILED+=("${name}")
    fi
done

# provenance.json records the code that produced the DATA.  Redrawn figures
# come from whatever post-processing code is checked out NOW, so record that
# too - otherwise the folder would claim one commit while its figures were
# drawn by another.  Kept PER SCRIPT and merged across calls, because a
# partial redraw (`run_postpro.sh <study> cf_validation`) refreshes only that
# script's figures and must not erase the record of the others.
(
    cd "${SCRIPT_DIR}"
    commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    dirty=false; [ -n "$(git status --porcelain -- "${CASE}/_postProcessing" 2>/dev/null)" ] && dirty=true
    "${VENV_PYTHON}" - "${STUDY_DIR}/postpro_rerun.json" "${commit}" "${dirty}" \
        "${NAMES[*]}" "${FAILED[*]:-}" <<'PY'
import json, os, sys
from datetime import datetime
out, commit, dirty, ran, failed = sys.argv[1:6]
failed = set(failed.split())
record = {}
if os.path.isfile(out):
    try:
        record = json.load(open(out))
    except ValueError:
        record = {}
if not isinstance(record.get("scripts"), dict):
    record["scripts"] = {}
record["note"] = ("figures redrawn from _rundata by run_postpro.sh, one entry "
                  "per script; the data itself is described by provenance.json")
now = datetime.now().astimezone().isoformat(timespec="seconds")
for name in ran.split():
    record["scripts"][name] = {"written": now, "postprocessing_commit": commit,
                               "postprocessing_dirty": dirty == "true",
                               "ok": name not in failed}
json.dump(record, open(out, "w"), indent=2)
PY
)

echo "=========================================="
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "Redraw of '${STUDY}' completed successfully -> ${STUDY_DIR}"
    exit 0
fi
echo "Redraw of '${STUDY}' completed with failures: ${FAILED[*]}"
exit 1
