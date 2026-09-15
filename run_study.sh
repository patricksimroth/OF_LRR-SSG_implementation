#!/bin/bash
if [ -z "${WM_PROJECT_DIR:-}" ]; then
    source /usr/lib/openfoam/openfoam2512/etc/bashrc
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/../.venv/bin/python3" "${SCRIPT_DIR}/turbulentFlatPlate/meshing/run_study.py" "$@"
