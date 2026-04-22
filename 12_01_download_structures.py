#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import os

import pandas as pd
from dotenv import load_dotenv
from mp_api.client import MPRester

# Load .env
load_dotenv()

api_key: Optional[str] = os.getenv("MP_API_KEY")
if not api_key:
    raise ValueError("MP_API_KEY is not set. Please set it in your .env file.")


# =========================================================
# CONFIG
# =========================================================
@dataclass
class DownloadConfig:
    element: str
    data_dir: Path = Path("data_all_R0Bs")
    out_root: Path = Path("out/12")

    max_download: Optional[int] = None

    # True: download only missing local CIF files
    # False: recheck download status even if the manifest already exists
    download_missing_only: bool = True

    # True: force re-download and overwrite existing CIF files
    force_refresh: bool = False

    @property
    def input_csv(self) -> Path:
        return self.data_dir / f"df_consist_{self.element}.csv"

    @property
    def out_dir(self) -> Path:
        return self.out_root / self.element

    @property
    def struct_dir(self) -> Path:
        return self.out_dir / "structures_cif"

    @property
    def cache_dir(self) -> Path:
        return self.out_dir / "cache"

    @property
    def manifest_path(self) -> Path:
        return self.cache_dir / "download_manifest.csv"


# =========================================================
# HELPERS
# =========================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_dirs(cfg: DownloadConfig) -> None:
    ensure_dir(cfg.out_dir)
    ensure_dir(cfg.struct_dir)
    ensure_dir(cfg.cache_dir)


def load_manifest(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=[
        "material_id", "status", "file_path", "error", "updated_at"
    ])


def save_manifest(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def get_structure_file_path(struct_dir: Path, material_id: str) -> Path:
    return struct_dir / f"{material_id}.cif"


def load_material_ids_from_csv(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "material_id" not in df.columns:
        raise ValueError(f"'material_id' column not found in {csv_path}")

    ids = (
        df["material_id"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    return ids


# =========================================================
# CORE
# =========================================================
def fetch_and_cache_structures(material_ids: List[str], cfg: DownloadConfig) -> pd.DataFrame:
    ensure_dirs(cfg)

    ids = list(dict.fromkeys(map(str, material_ids)))
    if cfg.max_download is not None:
        ids = ids[:cfg.max_download]

    manifest = load_manifest(cfg.manifest_path)
    manifest_map: Dict[str, Dict] = {
        str(row["material_id"]): row.to_dict()
        for _, row in manifest.iterrows()
    }

    to_download = []
    records = []

    for mid in ids:
        cif_path = get_structure_file_path(cfg.struct_dir, mid)

        # 1) Skip directly if the local file already exists and force refresh is disabled
        if cif_path.exists() and not cfg.force_refresh:
            records.append({
                "material_id": mid,
                "status": "skipped_existing",
                "file_path": str(cif_path),
                "error": "",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            })
            continue

        # 2) Skip if the manifest shows a successful download and the file exists
        if (
            cfg.download_missing_only
            and not cfg.force_refresh
            and mid in manifest_map
            and manifest_map[mid].get("status") == "downloaded"
            and cif_path.exists()
        ):
            records.append({
                "material_id": mid,
                "status": "skipped_manifest",
                "file_path": str(cif_path),
                "error": "",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            })
            continue

        to_download.append(mid)

    print(f"[INFO] element = {cfg.element}")
    print(f"[INFO] total unique material_ids = {len(ids)}")
    print(f"[INFO] need download = {len(to_download)}")
    print(f"[INFO] skip cached = {len(records)}")

    if not api_key:
        raise ValueError("Materials Project API key is missing.")

    if to_download:
        with MPRester(api_key) as mpr:
            for i, mid in enumerate(to_download, start=1):
                cif_path = get_structure_file_path(cfg.struct_dir, mid)
                try:
                    structure = mpr.get_structure_by_material_id(mid)
                    structure.to(filename=str(cif_path))
                    print(f"[{i}/{len(to_download)}] downloaded {mid} -> {cif_path.name}")

                    records.append({
                        "material_id": mid,
                        "status": "downloaded",
                        "file_path": str(cif_path),
                        "error": "",
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    })
                except Exception as e:
                    print(f"[WARN] failed to fetch {mid}: {e}")
                    records.append({
                        "material_id": mid,
                        "status": "failed",
                        "file_path": str(cif_path),
                        "error": str(e),
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    })

    out_df = pd.DataFrame(records)

    # Keep only the latest record for each material_id
    out_df = out_df.drop_duplicates(subset=["material_id"], keep="last")
    out_df = out_df.sort_values(["status", "material_id"]).reset_index(drop=True)

    save_manifest(out_df, cfg.manifest_path)
    print(f"[SAVE] manifest -> {cfg.manifest_path}")

    return out_df


# =========================================================
# CLI
# =========================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Materials Project structures incrementally."
    )
    parser.add_argument("--element", required=True, help="Target element, e.g. V, Ti, Fe")
    parser.add_argument("--data-dir", default="data_all_R0Bs", help="Directory containing df_consist_{ELEMENT}.csv")
    parser.add_argument("--out-root", default="out/12", help="Output root directory")
    parser.add_argument("--max-download", type=int, default=None, help="Maximum number of material IDs to download")
    parser.add_argument("--force-refresh", action="store_true", help="Force re-download even if cif already exists")
    parser.add_argument("--download-all", action="store_true", help="Disable missing-only mode")

    args = parser.parse_args()

    cfg = DownloadConfig(
        element=args.element,
        data_dir=Path(args.data_dir),
        out_root=Path(args.out_root),
        max_download=args.max_download,
        download_missing_only=not args.download_all,
        force_refresh=args.force_refresh,
    )

    material_ids = load_material_ids_from_csv(cfg.input_csv)
    fetch_and_cache_structures(material_ids, cfg)

    print("\n[DONE] Structure download step finished.")


if __name__ == "__main__":
    main()