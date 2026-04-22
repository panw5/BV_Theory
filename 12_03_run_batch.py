#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import List


# =========================================================
# USER SETTINGS
# =========================================================
DEFAULT_ELEMENTS = [
    "V",
    "Ti",
    "Fe",
    "Mn",
    "Mo",
]

DOWNLOAD_SCRIPT = "12_01_download_structures.py"
ANALYZE_SCRIPT = "12_02_analyze_local_env.py"


# =========================================================
# HELPERS
# =========================================================
def check_script_exists(script_path: str):
    if not Path(script_path).exists():
        raise FileNotFoundError(f"Required script not found: {script_path}")


def run_command(cmd: List[str], env=None) -> int:
    print("\n[RUN]", " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    return result.returncode


def normalize_elements(elements_raw: List[str]) -> List[str]:
    clean = []
    for x in elements_raw:
        if not x:
            continue
        s = str(x).strip()
        if s:
            clean.append(s)
    return clean


# =========================================================
# MAIN
# =========================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch runner for downloading structures and/or analyzing local environments."
    )

    parser.add_argument(
        "--elements",
        nargs="+",
        default=DEFAULT_ELEMENTS,
        help="List of target elements, e.g. V Ti Fe Mn Mo"
    )

    parser.add_argument(
        "--mode",
        choices=["download", "analyze", "both"],
        default="both",
        help="What to run for each element"
    )

    parser.add_argument(
        "--data-dir",
        default="data_all_R0Bs",
        help="Directory containing df_consist_{ELEMENT}.csv"
    )

    parser.add_argument(
        "--out-root",
        default="out/12",
        help="Output root directory"
    )

    # download options
    parser.add_argument(
        "--max-download",
        type=int,
        default=None,
        help="Maximum number of material IDs to download per element"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-download CIF files"
    )
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Disable missing-only mode and re-check all materials"
    )

    # analysis options
    parser.add_argument(
        "--nn-cutoff",
        type=float,
        default=3.2,
        help="O neighbor cutoff for local environment analysis"
    )
    parser.add_argument(
        "--metal-search-cutoff",
        type=float,
        default=2.6,
        help="Cutoff used in bridging oxygen check"
    )
    parser.add_argument(
        "--min-cn-count-for-savg",
        type=int,
        default=20,
        help="Minimum sample count threshold for s_avg plot"
    )

    # grouped subplot options
    parser.add_argument(
        "--subplot-elements",
        nargs="+",
        default=None,
        help="Only include these elements in one combined subplot figure, e.g. Co Cr Fe"
    )
    parser.add_argument(
        "--subplot-output-name",
        default=None,
        help="Output file name for one grouped subplot figure"
    )
    parser.add_argument(
        "--subplot-groups",
        default=None,
        help='Semicolon-separated element groups, e.g. "Co,Cr,Fe;Hf,Ir,Mn;Mo,Nb,Re"'
    )
    parser.add_argument(
        "--combined-ncols",
        type=int,
        default=3,
        help="Number of subplot columns for grouped/all-elements figure"
    )

    # behavior
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the whole batch when one element fails"
    )

    args = parser.parse_args()

    elements = normalize_elements(args.elements)

    if not elements:
        raise ValueError("No valid elements provided.")

    check_script_exists(DOWNLOAD_SCRIPT)
    check_script_exists(ANALYZE_SCRIPT)

    env = os.environ.copy()

    if args.mode in ["download", "both"]:
        if not env.get("MP_API_KEY"):
            raise ValueError(
                "MP_API_KEY is not set in the environment, but download mode was requested."
            )

    print("=" * 70)
    print(f"[INFO] batch mode = {args.mode}")
    print(f"[INFO] elements = {elements}")
    print(f"[INFO] data_dir = {args.data_dir}")
    print(f"[INFO] out_root = {args.out_root}")
    print("=" * 70)

    success_elements = []
    failed_elements = []

    for idx, element in enumerate(elements, start=1):
        print("\n" + "=" * 70)
        print(f"[BATCH] ({idx}/{len(elements)}) element = {element}")
        print("=" * 70)

        element_failed = False

        # -------------------------------------------------
        # STEP 1: download
        # -------------------------------------------------
        if args.mode in ["download", "both"]:
            download_cmd = [
                sys.executable,
                DOWNLOAD_SCRIPT,
                "--element", element,
                "--data-dir", args.data_dir,
                "--out-root", args.out_root,
            ]

            if args.max_download is not None:
                download_cmd += ["--max-download", str(args.max_download)]

            if args.force_refresh:
                download_cmd.append("--force-refresh")

            if args.download_all:
                download_cmd.append("--download-all")

            code = run_command(download_cmd, env=env)
            if code != 0:
                print(f"[ERROR] download failed for element {element} (exit code={code})")
                element_failed = True

                if args.stop_on_error:
                    raise RuntimeError(f"Batch stopped because download failed for {element}")

        # -------------------------------------------------
        # STEP 2: analyze single element
        # -------------------------------------------------
        if (not element_failed) and args.mode in ["analyze", "both"]:
            analyze_cmd = [
                sys.executable,
                ANALYZE_SCRIPT,
                "--element", element,
                "--data-dir", args.data_dir,
                "--out-root", args.out_root,
                "--nn-cutoff", str(args.nn_cutoff),
                "--metal-search-cutoff", str(args.metal_search_cutoff),
                "--min-cn-count-for-savg", str(args.min_cn_count_for_savg),

                # 关键：单元素分析时，不重复生成总拼图
                "--skip-combined-subplots",
            ]

            code = run_command(analyze_cmd, env=env)
            if code != 0:
                print(f"[ERROR] analysis failed for element {element} (exit code={code})")
                element_failed = True

                if args.stop_on_error:
                    raise RuntimeError(f"Batch stopped because analysis failed for {element}")

        if element_failed:
            failed_elements.append(element)
        else:
            success_elements.append(element)

    # -------------------------------------------------
    # STEP 3: after all analyses, generate combined/grouped subplot figures once
    # -------------------------------------------------
    if args.mode in ["analyze", "both"] and success_elements:
        print("\n" + "=" * 70)
        print("[BATCH] final combined/grouped subplot plotting")
        print("=" * 70)

        plot_cmd = [
            sys.executable,
            ANALYZE_SCRIPT,
            "--element", success_elements[0],
            "--data-dir", args.data_dir,
            "--out-root", args.out_root,
            "--nn-cutoff", str(args.nn_cutoff),
            "--metal-search-cutoff", str(args.metal_search_cutoff),
            "--min-cn-count-for-savg", str(args.min_cn_count_for_savg),
            "--plot-only-combined",
            "--combined-ncols", str(args.combined_ncols),
        ]

        if args.subplot_elements:
            plot_cmd += ["--subplot-elements", *args.subplot_elements]

        if args.subplot_output_name:
            plot_cmd += ["--subplot-output-name", args.subplot_output_name]

        if args.subplot_groups:
            plot_cmd += ["--subplot-groups", args.subplot_groups]

        code = run_command(plot_cmd, env=env)
        if code != 0:
            print(f"[ERROR] final combined/grouped plotting failed (exit code={code})")
            if args.stop_on_error:
                raise RuntimeError("Batch stopped because final combined plotting failed")

    print("\n" + "=" * 70)
    print("[SUMMARY]")
    print(f"Success: {success_elements}")
    print(f"Failed : {failed_elements}")
    print("=" * 70)

    if failed_elements:
        sys.exit(1)

    print("[DONE] Batch run finished successfully.")


if __name__ == "__main__":
    main()

"""
1. Batch download first, then analyze, and automatically generate the final combined figure for all elements
python 12_03_run_batch.py --elements V Ti Fe Mn Mo --mode both

2. Batch download only
python 12_03_run_batch.py --elements V Ti Fe --mode download

3. Batch analysis only
python 12_03_run_batch.py --elements V Ti Fe --mode analyze

4. Adjust parameters during batch analysis
python 12_03_run_batch.py \
  --elements V Ti Fe \
  --mode analyze \
  --nn-cutoff 3.0 \
  --metal-search-cutoff 2.5 \
  --min-cn-count-for-savg 10

5. Test downloading only the first few materials
python 12_03_run_batch.py --elements V Ti --mode download --max-download 50

6. Stop immediately if one element raises an error
python 12_03_run_batch.py --elements V Ti Fe --mode both --stop-on-error

7. Generate only one grouped subplot for the specified elements at the end
python 12_03_run_batch.py \
  --elements Co Cr Fe Hf Ir Mn Mo Nb Re \
  --mode analyze \
  --subplot-elements Co Cr Fe \
  --subplot-output-name group_1_Co_Cr_Fe.png

8. Generate multiple grouped subplots at once at the end
python 12_03_run_batch.py \
  --elements Co Cr Fe Hf Ir Mn Mo Nb Re Rh Ru Ta V W Zr \
  --mode analyze \
  --subplot-groups "Co,Cr,Fe;Hf,Ir,Mn;Mo,Nb,Re;Rh,Ru,Ta;V,W,Zr"

9. Generate grouped subplots only from existing summary files, without re-analyzing
python 12_02_analyze_local_env.py \
  --element Co \
  --data-dir data_all_R0Bs \
  --out-root out/12 \
  --plot-only-combined \
  --subplot-groups "Co,Cr,Fe;Hf,Ir,Mn;Mo,Nb,Re"

10. Generate only one specified element group figure, without re-analyzing
python 12_02_analyze_local_env.py \
  --element Co \
  --data-dir data_all_R0Bs \
  --out-root out/12 \
  --plot-only-combined \
  --subplot-elements Co Cr Fe \
  --subplot-output-name group_1_Co_Cr_Fe.png


python 12_03_run_batch.py \
  --elements Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Ru Rh Pd Ag Cd La Hf Ta W Re Os Ir Pt Au Hg \
  --mode analyze

python 12_02_analyze_local_env.py \
  --element Cr \
  --data-dir data_all_R0Bs \
  --out-root out/12 \
  --plot-only-combined \
  --subplot-groups "Cr,Mo,Re,Ru,V,W;Sc,Ti,Mn,Fe,Co,Ni,Cu,Zn,Y;Zr,Nb,Rh,Pd,Ag,Cd,La,Hf,Ta;Os,Ir,Pt,Au,Hg"
"""