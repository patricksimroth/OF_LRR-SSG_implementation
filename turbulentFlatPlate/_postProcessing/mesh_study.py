"""
mesh_study.py – post-processing for config-driven mesh studies (stub).

Planned: read run/<case>/study_manifest.json (written by meshing/run_study.py),
group the <model>__<tag> run dirs by model and produce per model
  - Cf(x) overlay, one curve per mesh parameter set
  - a convergence table (cells, Cf interpolated at x=0.97008, wall y+
    min/max/avg, log-law RMS) vs the swept parameter
into results/<case>/mesh_study/.  Readers will be shared with cf_validation.py
via _pp_common.py.
"""
from __future__ import annotations

import os
import sys

import _pp_common as pp


def main():
    _, case_name, run_case_dir, _, results_root = pp.resolve_case_paths(
        os.path.dirname(os.path.abspath(__file__))
    )
    manifest = pp.load_manifest(run_case_dir)
    if manifest is None:
        print(f"[info] no study manifest in {run_case_dir} - nothing to do")
        return 0
    combos = manifest.get("combos", [])
    models = sorted({c.get("model", "?") for c in combos})
    print(f"[mesh_study] study '{manifest.get('study')}': "
          f"{len(combos)} combos, models: {models}")
    print("[warn] mesh_study plots not implemented yet (structure stub)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
