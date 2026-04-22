from __future__ import annotations

import re
import ast
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pymatgen.core import Structure


# plotting style
sns.set(style="whitegrid", context="talk")


# =========================================================
# CONFIG
# =========================================================
@dataclass
class AnalysisConfig:
    element: str
    data_dir: Path = Path("data_all_R0Bs")
    out_root: Path = Path("out/12")

    nn_cutoff: float = 3.2
    metal_search_cutoff: float = 2.6
    min_cn_count_for_savg: int = 20

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
    def table_dir(self) -> Path:
        return self.out_dir / "tables"

    @property
    def plot_dir(self) -> Path:
        return self.out_dir / "plots"

    @property
    def normalized_detail_out(self) -> Path:
        return self.table_dir / f"{self.element}_all_CNs_detailed_rows.csv"

    @property
    def material_summary_out(self) -> Path:
        return self.table_dir / f"{self.element}_all_CNs_materials_summary.csv"

    @property
    def cn_input_summary_out(self) -> Path:
        return self.table_dir / f"{self.element}_CN_input_summary.csv"

    @property
    def detail_out(self) -> Path:
        return self.table_dir / f"{self.element}_local_descriptors_detailed.csv"

    @property
    def cn_summary_out(self) -> Path:
        return self.table_dir / f"{self.element}_cn_summary.csv"

    @property
    def cn_geom_summary_out(self) -> Path:
        return self.table_dir / f"{self.element}_cn_geometry_summary.csv"

    @property
    def cn_dominant_out(self) -> Path:
        return self.table_dir / f"{self.element}_cn_dominant_geometry.csv"

    @property
    def cn_charge_summary_out(self) -> Path:
        return self.table_dir / f"{self.element}_cn_mean_charge_summary.csv"

    @property
    def cn_geom_charge_summary_out(self) -> Path:
        return self.table_dir / f"{self.element}_cn_geom_mean_charge_summary.csv"

    @property
    def cn_charge_fraction_detail_out(self) -> Path:
        return self.table_dir / f"{self.element}_cn_charge_fraction_details.csv"

    @property
    def cn_savg_out(self) -> Path:
        return self.table_dir / f"{self.element}_cn_savg_summary.csv"

    @property
    def cn_savg_plot_out(self) -> Path:
        return self.plot_dir / f"{self.element}_savg_vs_CN_min_count_gt_{self.min_cn_count_for_savg}.png"

    @property
    def cn_savg_colored_plot_out(self) -> Path:
        return self.plot_dir / f"{self.element}_savg_vs_CN_colored_by_gt1_min_count_gt_{self.min_cn_count_for_savg}.png"

    @property
    def cn_savg_flagged_table_out(self) -> Path:
        return self.table_dir / f"{self.element}_cn_savg_summary_with_flag.csv"

    @property
    def combined_savg_subplot_table_out(self) -> Path:
        return self.out_root / f"all_elements_cn_savg_summary_min_count_gt_{self.min_cn_count_for_savg}.csv"

    @property
    def combined_savg_subplot_plot_out(self) -> Path:
        return self.out_root / f"all_elements_savg_subplots_min_count_gt_{self.min_cn_count_for_savg}.png"


# =========================================================
# HELPERS
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def ensure_dirs(cfg: AnalysisConfig):
    ensure_dir(cfg.out_root)
    ensure_dir(cfg.out_dir)
    ensure_dir(cfg.struct_dir)
    ensure_dir(cfg.table_dir)
    ensure_dir(cfg.plot_dir)


def save_plot(fig, out_path: Path):
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[SAVE] {out_path}")


def parse_list_cell(x):
    """
    Parse a cell that may contain list-like content into list[float].
    Examples:
      "[4, 6]" -> [4.0, 6.0]
      "4"      -> [4.0]
      "4, 6"   -> [4.0, 6.0]
    """
    if pd.isna(x):
        return []

    s = str(x).strip()

    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return [float(v) for v in val]
        if isinstance(val, (int, float)):
            return [float(val)]
    except Exception:
        pass

    nums = re.findall(r"[-+]?\d*\.?\d+", s)
    return [float(n) for n in nums]


def parse_cn_value(x):
    if pd.isna(x):
        return None
    try:
        v = float(x)
        if abs(v - round(v)) < 1e-9:
            return int(round(v))
        return v
    except Exception:
        return None


def is_integer_like(x, tol=1e-8):
    if pd.isna(x):
        return False
    try:
        x = float(x)
        return abs(x - round(x)) < tol
    except Exception:
        return False


def charge_to_label(element: str, x):
    try:
        x = int(round(float(x)))
        return f"{element}{x}+"
    except Exception:
        return f"{element}_unknown"


def classify_savg_threshold(x, threshold=1.0):
    if pd.isna(x):
        return "unknown"
    return "s_avg > 1" if float(x) > threshold else "s_avg ≤ 1"


def sanitize_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^\w\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def parse_subplot_groups(group_string: Optional[str]) -> List[List[str]]:
    """
    输入示例:
        "Co,Cr,Fe;Hf,Ir,Mn;Mo,Nb,Re"
    输出:
        [["Co","Cr","Fe"], ["Hf","Ir","Mn"], ["Mo","Nb","Re"]]
    """
    if not group_string:
        return []

    groups = []
    for group in str(group_string).split(";"):
        items = [x.strip() for x in group.split(",") if str(x).strip()]
        if items:
            groups.append(items)
    return groups


# =========================================================
# PART 1 — INPUT NORMALIZATION
# =========================================================
def detect_r0_column(df: pd.DataFrame) -> str:
    candidates = []
    for c in df.columns:
        s = str(c).strip()
        if s in ["R0", "R_0", "R$_0$"]:
            candidates.append(c)
        elif "R$_0$" in s:
            candidates.append(c)
    if not candidates:
        raise ValueError("R0 column not found.")
    return candidates[0]


def detect_b_column(df: pd.DataFrame) -> str:
    if "B" in df.columns:
        return "B"
    for c in df.columns:
        if str(c).strip() == "B":
            return c
    raise ValueError("B column not found.")


def detect_coord_column(df: pd.DataFrame, element: str) -> str:
    exact = f"{element}O_coordn"
    if exact in df.columns:
        return exact

    candidates = [c for c in df.columns if str(c).endswith("O_coordn")]
    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(f"coordination column not found for {element}.")


def detect_chemenv_column(df: pd.DataFrame, element: str) -> Optional[str]:
    exact = f"{element}_chemenv"
    if exact in df.columns:
        return exact

    candidates = [c for c in df.columns if str(c).endswith("_chemenv")]
    if len(candidates) == 1:
        return candidates[0]

    return None


def normalize_element_table(df: pd.DataFrame, element: str) -> pd.DataFrame:
    r0_col = detect_r0_column(df)
    b_col = detect_b_column(df)
    cn_col = detect_coord_column(df, element)
    chemenv_col = detect_chemenv_column(df, element)

    keep_cols = ["material_id", "chem_formula", r0_col, b_col, cn_col]

    charge_cols = [
        f"mean_{element}_charge",
        f"min_{element}_charge",
        f"max_{element}_charge",
    ]
    for c in charge_cols:
        if c in df.columns:
            keep_cols.append(c)

    if chemenv_col is not None:
        keep_cols.append(chemenv_col)

    optional_cols = [
        f"{element}_wyckoff",
        f"num_{element}_chemenvs",
        f"num_{element}_wyckoffs",
    ]
    for c in optional_cols:
        if c in df.columns:
            keep_cols.append(c)

    d = df[keep_cols].copy()

    rename_dict = {
        r0_col: "R0",
        b_col: "B",
        cn_col: "CN_raw",
    }

    if f"mean_{element}_charge" in d.columns:
        rename_dict[f"mean_{element}_charge"] = "mean_charge"
    if f"min_{element}_charge" in d.columns:
        rename_dict[f"min_{element}_charge"] = "min_charge"
    if f"max_{element}_charge" in d.columns:
        rename_dict[f"max_{element}_charge"] = "max_charge"

    d = d.rename(columns=rename_dict)

    if chemenv_col is not None and chemenv_col in d.columns:
        d = d.rename(columns={chemenv_col: "chemenv"})
    else:
        d["chemenv"] = None

    d["R0"] = pd.to_numeric(d["R0"], errors="coerce")
    d["B"] = pd.to_numeric(d["B"], errors="coerce")

    for c in ["mean_charge", "min_charge", "max_charge"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d["CN_list"] = d["CN_raw"].apply(parse_list_cell)
    d = d.explode("CN_list").rename(columns={"CN_list": "CN"})
    d["CN"] = pd.to_numeric(d["CN"], errors="coerce")

    d = d.dropna(subset=["material_id", "R0", "B", "CN"]).copy()
    d["element"] = element

    if "min_charge" in d.columns and "max_charge" in d.columns:
        d["charge_range"] = d["max_charge"] - d["min_charge"]
        d["is_single_charge"] = d["min_charge"] == d["max_charge"]
    else:
        d["charge_range"] = pd.NA
        d["is_single_charge"] = pd.NA

    return d


def load_element_table(cfg: AnalysisConfig) -> pd.DataFrame:
    if not cfg.input_csv.exists():
        raise FileNotFoundError(f"File not found: {cfg.input_csv}")
    df = pd.read_csv(cfg.input_csv)
    return normalize_element_table(df, cfg.element)


def save_input_tables(d: pd.DataFrame, cfg: AnalysisConfig):
    d.to_csv(cfg.normalized_detail_out, index=False)
    print(f"[SAVE] normalized detailed rows -> {cfg.normalized_detail_out}")

    agg_dict = {
        "n_rows": ("material_id", "size"),
        "CN_values": ("CN", lambda x: sorted(set(
            int(v) if float(v).is_integer() else float(v) for v in x
        ))),
        "R0_mean": ("R0", "mean"),
        "B_mean": ("B", "mean"),
        "chemenv_examples": ("chemenv", lambda x: "; ".join(sorted(set(map(str, x.dropna()))))[:300]),
    }

    if "mean_charge" in d.columns:
        agg_dict["mean_charge_mean"] = ("mean_charge", "mean")
    if "min_charge" in d.columns:
        agg_dict["min_charge_min"] = ("min_charge", "min")
    if "max_charge" in d.columns:
        agg_dict["max_charge_max"] = ("max_charge", "max")

    summary = (
        d.groupby(["material_id", "chem_formula"], as_index=False)
         .agg(**agg_dict)
         .sort_values(["n_rows", "material_id"], ascending=[False, True])
    )
    summary.to_csv(cfg.material_summary_out, index=False)
    print(f"[SAVE] materials summary -> {cfg.material_summary_out}")

    cn_agg = {
        "n_rows": ("CN", "size"),
        "n_unique_materials": ("material_id", "nunique"),
        "R0_mean": ("R0", "mean"),
        "B_mean": ("B", "mean"),
    }
    if "mean_charge" in d.columns:
        cn_agg["mean_charge_mean"] = ("mean_charge", "mean")

    cn_summary = (
        d.groupby("CN", as_index=False)
         .agg(**cn_agg)
         .sort_values("CN")
    )
    cn_summary.to_csv(cfg.cn_input_summary_out, index=False)
    print(f"[SAVE] CN input summary -> {cfg.cn_input_summary_out}")


# =========================================================
# PART 2 — STRUCTURE / LOCAL DESCRIPTORS
# =========================================================
def load_structure(cif_path: Path):
    return Structure.from_file(str(cif_path))


def angle_deg(structure, i, center, k):
    v1 = structure[i].coords - structure[center].coords
    v2 = structure[k].coords - structure[center].coords

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return np.nan

    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def get_candidate_element_sites(structure, element: str):
    return [i for i, site in enumerate(structure) if site.specie.symbol == element]


def get_oxygen_neighbors_by_cutoff(structure, center_idx, cutoff=3.2):
    center = structure[center_idx]
    neighs = structure.get_neighbors(center, cutoff)

    oxy = []
    for nn in neighs:
        if hasattr(nn, "specie"):
            symbol = nn.specie.symbol
        elif hasattr(nn, "species_string"):
            symbol = nn.species_string
        else:
            continue

        if symbol != "O":
            continue

        if hasattr(nn, "index"):
            idx = int(nn.index)
        else:
            idx = None
            for i, site in enumerate(structure):
                if np.allclose(site.coords, nn.coords, atol=1e-6):
                    idx = i
                    break
            if idx is None:
                continue

        if hasattr(nn, "nn_distance"):
            dist = float(nn.nn_distance)
        else:
            dist = float(structure.get_distance(center_idx, idx))

        oxy.append((idx, dist))

    oxy = sorted(oxy, key=lambda x: x[1])
    return oxy


def count_bridging_oxygens(structure, oxygen_indices, center_idx, metal_search_cutoff: float):
    count = 0

    metal_indices = [
        i for i, site in enumerate(structure)
        if site.specie.symbol != "O"
    ]

    for o_idx in oxygen_indices:
        bridged = False

        for m_idx in metal_indices:
            if m_idx == center_idx:
                continue

            d = structure.get_distance(o_idx, m_idx)
            if d <= metal_search_cutoff:
                bridged = True
                break

        if bridged:
            count += 1

    return count


def compute_site_descriptors(structure, center_idx, cn_expected: int, cfg: AnalysisConfig):
    oxy = get_oxygen_neighbors_by_cutoff(structure, center_idx, cutoff=cfg.nn_cutoff)

    if oxy is None or len(oxy) == 0:
        return None

    if len(oxy) != cn_expected:
        return None

    o_indices = [x[0] for x in oxy]
    distances = np.array([x[1] for x in oxy], dtype=float)

    if distances.size == 0:
        return None

    d_min = float(np.min(distances))
    d_max = float(np.max(distances))
    d_mean = float(np.mean(distances))
    d_std = float(np.std(distances))
    d_range = float(d_max - d_min)

    angles = []
    for i, j in combinations(o_indices, 2):
        ang = angle_deg(structure, i, center_idx, j)
        if np.isfinite(ang):
            angles.append(ang)

    if len(angles) > 0:
        angle_mean = float(np.mean(angles))
        angle_std = float(np.std(angles))
        angle_min = float(np.min(angles))
        angle_max = float(np.max(angles))
    else:
        angle_mean = np.nan
        angle_std = np.nan
        angle_min = np.nan
        angle_max = np.nan

    bridging_O_count = count_bridging_oxygens(
        structure, o_indices, center_idx=center_idx, metal_search_cutoff=cfg.metal_search_cutoff
    )
    bridging_O_fraction = float(bridging_O_count / cn_expected) if cn_expected > 0 else np.nan

    return {
        "site_index": center_idx,
        "cn_found": len(oxy),
        "d_min": d_min,
        "d_max": d_max,
        "d_mean": d_mean,
        "d_std": d_std,
        "d_range": d_range,
        "angle_mean": angle_mean,
        "angle_std": angle_std,
        "angle_min": angle_min,
        "angle_max": angle_max,
        "bridging_O_count": bridging_O_count,
        "bridging_O_fraction": bridging_O_fraction,
    }


def build_detailed_descriptor_table(df_input: pd.DataFrame, cfg: AnalysisConfig):
    results = []
    structure_cache = {}

    for _, row in df_input.iterrows():
        material_id = str(row["material_id"])
        cn = parse_cn_value(row["CN"])

        if cn is None:
            continue
        if not isinstance(cn, int):
            continue
        if cn <= 0:
            continue

        cif_path = cfg.struct_dir / f"{material_id}.cif"
        if not cif_path.exists():
            print(f"[SKIP] missing cif: {material_id}")
            continue

        if material_id not in structure_cache:
            try:
                structure_cache[material_id] = load_structure(cif_path)
            except Exception as e:
                print(f"[SKIP] failed to load {material_id}: {e}")
                continue

        structure = structure_cache[material_id]
        candidate_sites = get_candidate_element_sites(structure, cfg.element)

        matched = 0
        for site_idx in candidate_sites:
            desc = compute_site_descriptors(
                structure=structure,
                center_idx=site_idx,
                cn_expected=cn,
                cfg=cfg,
            )
            if desc is None:
                continue

            matched += 1
            out = row.to_dict()
            out.update(desc)
            out["n_candidate_sites_same_cn"] = matched
            out["geometry_type"] = row.get("chemenv", pd.NA)
            results.append(out)

        if matched == 0:
            print(f"[NO MATCH] material_id={material_id}, CN={cn}")
            out = row.to_dict()
            out["site_index"] = np.nan
            out["cn_found"] = np.nan
            out["d_min"] = np.nan
            out["d_max"] = np.nan
            out["d_mean"] = np.nan
            out["d_std"] = np.nan
            out["d_range"] = np.nan
            out["angle_mean"] = np.nan
            out["angle_std"] = np.nan
            out["angle_min"] = np.nan
            out["angle_max"] = np.nan
            out["bridging_O_count"] = np.nan
            out["bridging_O_fraction"] = np.nan
            out["n_candidate_sites_same_cn"] = 0
            out["geometry_type"] = row.get("chemenv", pd.NA)
            results.append(out)

    out_df = pd.DataFrame(results)
    out_df.to_csv(cfg.detail_out, index=False)
    print(f"[SAVE] detailed descriptor table -> {cfg.detail_out.resolve()}")
    print(f"[INFO] total rows in output = {len(out_df)}")
    return out_df


# =========================================================
# PART 3 — SUMMARY TABLES
# =========================================================
def save_analysis_summaries(detail_df: pd.DataFrame, d_input: pd.DataFrame, cfg: AnalysisConfig):
    if detail_df.empty:
        print("[WARN] no rows in detailed table, summaries skipped.")
        return

    cn_summary = (
        detail_df.groupby("CN")
        .agg(
            count=("CN", "size"),
            n_materials=("material_id", "nunique"),
            d_min_mean=("d_min", "mean"),
            d_mean_mean=("d_mean", "mean"),
            d_std_mean=("d_std", "mean"),
            bridging_O_fraction_mean=("bridging_O_fraction", "mean"),
        )
        .reset_index()
        .sort_values("CN")
    )
    cn_summary.to_csv(cfg.cn_summary_out, index=False)

    cn_geom_summary = (
        detail_df.groupby(["CN", "geometry_type"])
        .agg(
            count=("geometry_type", "size"),
            n_materials=("material_id", "nunique"),
            d_min_mean=("d_min", "mean"),
            d_mean_mean=("d_mean", "mean"),
            d_std_mean=("d_std", "mean"),
            bridging_O_fraction_mean=("bridging_O_fraction", "mean"),
        )
        .reset_index()
        .sort_values(["CN", "count"], ascending=[True, False])
    )

    cn_totals = cn_geom_summary.groupby("CN")["count"].transform("sum")
    cn_geom_summary["fraction_within_cn"] = cn_geom_summary["count"] / cn_totals
    cn_geom_summary.to_csv(cfg.cn_geom_summary_out, index=False)

    cn_dominant = (
        cn_geom_summary.sort_values(["CN", "count"], ascending=[True, False])
        .groupby("CN", as_index=False)
        .first()
        .rename(columns={
            "geometry_type": "dominant_geometry",
            "count": "dominant_count",
            "fraction_within_cn": "dominant_fraction"
        })
    )
    cn_dominant.to_csv(cfg.cn_dominant_out, index=False)

    if "mean_charge" in detail_df.columns:
        cn_charge_summary = (
            detail_df.groupby(["CN", "mean_charge"])
            .agg(
                count=("mean_charge", "size"),
                n_materials=("material_id", "nunique"),
                d_min_mean=("d_min", "mean"),
                d_mean_mean=("d_mean", "mean"),
                d_std_mean=("d_std", "mean"),
            )
            .reset_index()
            .sort_values(["CN", "count"], ascending=[True, False])
        )

        cn_charge_totals = cn_charge_summary.groupby("CN")["count"].transform("sum")
        cn_charge_summary["fraction_within_cn"] = cn_charge_summary["count"] / cn_charge_totals
        cn_charge_summary.to_csv(cfg.cn_charge_summary_out, index=False)

        cn_geom_charge_summary = (
            detail_df.groupby(["CN", "geometry_type", "mean_charge"])
            .agg(
                count=("mean_charge", "size"),
                n_materials=("material_id", "nunique"),
                d_min_mean=("d_min", "mean"),
                d_mean_mean=("d_mean", "mean"),
                d_std_mean=("d_std", "mean"),
                bridging_O_fraction_mean=("bridging_O_fraction", "mean"),
            )
            .reset_index()
            .sort_values(["CN", "geometry_type", "count"], ascending=[True, True, False])
        )

        cn_geom_charge_totals = cn_geom_charge_summary.groupby(["CN", "geometry_type"])["count"].transform("sum")
        cn_geom_charge_summary["fraction_within_cn_geometry"] = (
            cn_geom_charge_summary["count"] / cn_geom_charge_totals
        )
        cn_geom_charge_summary.to_csv(cfg.cn_geom_charge_summary_out, index=False)

    if "mean_charge" in d_input.columns:
        d_int = d_input.copy()
        d_int["mean_charge"] = pd.to_numeric(d_int["mean_charge"], errors="coerce")
        d_int = d_int[d_int["mean_charge"].notna()].copy()
        d_int = d_int[d_int["mean_charge"].apply(is_integer_like)].copy()

        if not d_int.empty:
            d_int["formal_charge"] = d_int["mean_charge"].round().astype(int)

            cn_counts = (
                d_int.groupby("CN", as_index=False)
                .agg(
                    n_samples=("CN", "size"),
                    n_materials=("material_id", "nunique")
                )
                .sort_values("CN")
            )

            cn_charge_frac = (
                d_int.groupby(["CN", "formal_charge"])
                .agg(count=("formal_charge", "size"))
                .reset_index()
                .sort_values(["CN", "formal_charge"])
            )

            cn_totals = cn_charge_frac.groupby("CN")["count"].transform("sum")
            cn_charge_frac["fraction_within_cn"] = cn_charge_frac["count"] / cn_totals

            cn_charge_frac["weighted_charge_contribution"] = (
                cn_charge_frac["formal_charge"] * cn_charge_frac["fraction_within_cn"]
            )

            cn_savg_summary = (
                cn_charge_frac.groupby("CN", as_index=False)
                .agg(
                    avg_formal_charge=("weighted_charge_contribution", "sum")
                )
                .sort_values("CN")
            )

            cn_savg_summary["s_avg_approx"] = (
                cn_savg_summary["avg_formal_charge"] / cn_savg_summary["CN"]
            )

            cn_savg_summary["savg_class"] = cn_savg_summary["s_avg_approx"].apply(
                lambda x: classify_savg_threshold(x, threshold=1.0)
            )

            charge_dist = (
                cn_charge_frac.groupby("CN")
                .apply(
                    lambda g: "; ".join(
                        f"{int(row['formal_charge'])}+:{row['fraction_within_cn']:.4f}"
                        for _, row in g.iterrows()
                    )
                )
                .reset_index(name="charge_fraction_breakdown")
            )

            cn_savg_summary = cn_savg_summary.merge(cn_counts, on="CN", how="left")
            cn_savg_summary = cn_savg_summary.merge(charge_dist, on="CN", how="left")

            cn_savg_summary = cn_savg_summary[
                [
                    "CN",
                    "n_samples",
                    "n_materials",
                    "avg_formal_charge",
                    "s_avg_approx",
                    "savg_class",
                    "charge_fraction_breakdown",
                ]
            ]

            cn_charge_frac.to_csv(cfg.cn_charge_fraction_detail_out, index=False)
            cn_savg_summary.to_csv(cfg.cn_savg_out, index=False)
            cn_savg_summary.to_csv(cfg.cn_savg_flagged_table_out, index=False)

            print(f"[SAVE] CN charge-fraction details -> {cfg.cn_charge_fraction_detail_out}")
            print(f"[SAVE] CN approximate s_avg summary -> {cfg.cn_savg_out}")
            print(f"[SAVE] CN approximate s_avg summary with flag -> {cfg.cn_savg_flagged_table_out}")
        else:
            print("[WARN] No integer mean_charge rows found in original exploded input. s_avg summary not generated.")

    print(f"[SAVE] CN summary -> {cfg.cn_summary_out}")
    print(f"[SAVE] CN+geometry summary -> {cfg.cn_geom_summary_out}")
    print(f"[SAVE] dominant geometry summary -> {cfg.cn_dominant_out}")
    if "mean_charge" in detail_df.columns:
        print(f"[SAVE] CN+mean_charge summary -> {cfg.cn_charge_summary_out}")
        print(f"[SAVE] CN+geometry+mean_charge summary -> {cfg.cn_geom_charge_summary_out}")


# =========================================================
# PART 4 — PLOTTING
# =========================================================
def make_plots(detail_df: pd.DataFrame, cfg: AnalysisConfig):
    df = detail_df.copy()
    df = df[df["site_index"].notna()].copy()

    if df.empty:
        print("[WARN] no matched rows, plotting skipped.")
        return

    if "mean_charge" not in df.columns:
        raise ValueError(
            "Missing required column: 'mean_charge'. "
            "Please make sure the input normalization step kept charge columns."
        )

    df["CN"] = pd.to_numeric(df["CN"], errors="coerce")
    df["d_std"] = pd.to_numeric(df["d_std"], errors="coerce")
    df["d_min"] = pd.to_numeric(df["d_min"], errors="coerce")
    df["bridging_O_fraction"] = pd.to_numeric(df["bridging_O_fraction"], errors="coerce")
    df["mean_charge"] = pd.to_numeric(df["mean_charge"], errors="coerce")
    df = df[df["CN"].notna()].copy()

    print(f"[INFO] total matched rows = {len(df)}")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="CN", y="d_std", ax=ax)
    sns.stripplot(data=df, x="CN", y="d_std", color="black", size=1, alpha=0.2, ax=ax)
    ax.set_title(f"Bond length std vs CN ({cfg.element}–O)")
    ax.set_xlabel("Coordination Number (CN)")
    ax.set_ylabel("Bond length std (Å)")
    save_plot(fig, cfg.plot_dir / f"{cfg.element}_d_std_vs_CN_all.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="CN", y="d_min", ax=ax)
    ax.set_title(f"Minimum bond length vs CN ({cfg.element}–O)")
    ax.set_xlabel("Coordination Number (CN)")
    ax.set_ylabel("Minimum bond length (Å)")
    save_plot(fig, cfg.plot_dir / f"{cfg.element}_d_min_vs_CN_all.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="CN", y="bridging_O_fraction", ax=ax)
    ax.set_title(f"Bridging oxygen fraction vs CN ({cfg.element}–O)")
    ax.set_xlabel("Coordination Number (CN)")
    ax.set_ylabel("Bridging O fraction")
    save_plot(fig, cfg.plot_dir / f"{cfg.element}_bridging_vs_CN_all.png")
    plt.close(fig)

    df_int = df[df["mean_charge"].apply(is_integer_like)].copy()

    if df_int.empty:
        print("[WARN] No rows remain after filtering integer mean_charge.")
        return

    df_int["formal_charge"] = df_int["mean_charge"].round().astype(int)
    df_int["formal_charge_label"] = df_int["formal_charge"].apply(lambda x: charge_to_label(cfg.element, x))

    charge_order_df = (
        df_int[["formal_charge", "formal_charge_label"]]
        .drop_duplicates()
        .sort_values("formal_charge")
    )
    hue_order = charge_order_df["formal_charge_label"].tolist()

    print(f"[INFO] rows with integer mean_charge = {len(df_int)}")
    print(f"[INFO] integer mean charges found = {charge_order_df['formal_charge'].tolist()}")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df_int,
        x="CN",
        y="d_std",
        hue="formal_charge_label",
        hue_order=hue_order,
        ax=ax
    )
    sns.stripplot(
        data=df_int,
        x="CN",
        y="d_std",
        hue="formal_charge_label",
        hue_order=hue_order,
        dodge=True,
        size=1,
        alpha=0.2,
        ax=ax
    )
    handles, labels = ax.get_legend_handles_labels()
    n = len(hue_order)
    ax.legend(handles[:n], labels[:n], title="Formal charge", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_title(f"Bond length std vs CN ({cfg.element}–O)")
    ax.set_xlabel("Coordination Number (CN)")
    ax.set_ylabel("Bond length std (Å)")
    save_plot(fig, cfg.plot_dir / f"{cfg.element}_d_std_vs_CN_integer_mean_charge_hue.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df_int,
        x="CN",
        y="d_min",
        hue="formal_charge_label",
        hue_order=hue_order,
        ax=ax
    )
    ax.legend(title="Formal charge", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_title(f"Minimum bond length vs CN ({cfg.element}–O)")
    ax.set_xlabel("Coordination Number (CN)")
    ax.set_ylabel("Minimum bond length (Å)")
    save_plot(fig, cfg.plot_dir / f"{cfg.element}_d_min_vs_CN_integer_mean_charge_hue.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df_int,
        x="CN",
        y="bridging_O_fraction",
        hue="formal_charge_label",
        hue_order=hue_order,
        ax=ax
    )
    ax.legend(title="Formal charge", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_title(f"Bridging oxygen fraction vs CN ({cfg.element}–O)")
    ax.set_xlabel("Coordination Number (CN)")
    ax.set_ylabel("Bridging O fraction")
    save_plot(fig, cfg.plot_dir / f"{cfg.element}_bridging_vs_CN_integer_mean_charge_hue.png")
    plt.close(fig)

    print("\nAll plots saved successfully.")


def plot_single_element_savg(df_plot: pd.DataFrame, element: str, cfg: AnalysisConfig):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        df_plot["CN"],
        df_plot["s_avg_approx"],
        linestyle="-",
        linewidth=2,
        color="gray",
        alpha=0.8,
        zorder=1,
        label="Trend"
    )

    df_gt = df_plot[df_plot["s_avg_approx"] > 1].copy()
    df_le = df_plot[df_plot["s_avg_approx"] <= 1].copy()

    if not df_gt.empty:
        ax.scatter(
            df_gt["CN"],
            df_gt["s_avg_approx"],
            s=180,
            color="crimson",
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
            label="s_avg > 1"
        )

    if not df_le.empty:
        ax.scatter(
            df_le["CN"],
            df_le["s_avg_approx"],
            s=180,
            color="royalblue",
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
            label="s_avg ≤ 1"
        )

    ax.axhline(
        y=1.0,
        linestyle="--",
        linewidth=1.5,
        color="black",
        alpha=0.8,
        label="s_avg = 1"
    )

    for _, row in df_plot.iterrows():
        ax.text(
            row["CN"],
            row["s_avg_approx"],
            f"n={int(row['n_samples'])}",
            fontsize=10,
            ha="left",
            va="bottom"
        )

    ax.set_title(f"{element}: s_avg vs CN (only CN with n_samples > {cfg.min_cn_count_for_savg})")
    ax.set_xlabel("Coordination Number (CN)")
    ax.set_ylabel("s_avg_approx")
    ax.legend()

    save_plot(fig, cfg.cn_savg_colored_plot_out)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(data=df_plot, x="CN", y="s_avg_approx", marker="o", ax=ax)

    for _, row in df_plot.iterrows():
        ax.text(
            row["CN"],
            row["s_avg_approx"],
            f"n={int(row['n_samples'])}",
            fontsize=10,
            ha="left",
            va="bottom"
        )

    ax.axhline(y=1.0, linestyle="--", linewidth=1.2, color="black", alpha=0.7)
    ax.set_title(f"{element}: s_avg vs CN (only CN with n_samples > {cfg.min_cn_count_for_savg})")
    ax.set_xlabel("Coordination Number (CN)")
    ax.set_ylabel("s_avg_approx")

    save_plot(fig, cfg.cn_savg_plot_out)
    plt.close(fig)


def plot_savg_from_summary(cfg: AnalysisConfig):
    if not cfg.cn_savg_out.exists():
        print(f"[WARN] Missing file: {cfg.cn_savg_out}")
        return

    df = pd.read_csv(cfg.cn_savg_out)

    if df.empty:
        print("[WARN] CN_SAVG_OUT is empty, skip plotting s_avg.")
        return

    df["CN"] = pd.to_numeric(df["CN"], errors="coerce")
    df["n_samples"] = pd.to_numeric(df["n_samples"], errors="coerce")
    df["s_avg_approx"] = pd.to_numeric(df["s_avg_approx"], errors="coerce")

    df_plot = df[df["n_samples"] > cfg.min_cn_count_for_savg].copy()
    df_plot = df_plot.dropna(subset=["CN", "s_avg_approx"])

    if df_plot.empty:
        print(f"[WARN] No CN groups with n_samples > {cfg.min_cn_count_for_savg}, skip s_avg plot.")
        return

    df_plot = df_plot.sort_values("CN").copy()
    df_plot["savg_class"] = df_plot["s_avg_approx"].apply(lambda x: classify_savg_threshold(x, threshold=1.0))

    df_plot.to_csv(cfg.cn_savg_flagged_table_out, index=False)
    print(f"[SAVE] plotted s_avg table with flag -> {cfg.cn_savg_flagged_table_out}")

    plot_single_element_savg(df_plot, cfg.element, cfg)


def collect_all_elements_savg_tables(
    cfg: AnalysisConfig,
    selected_elements: Optional[List[str]] = None
) -> pd.DataFrame:
    summary_files = list(cfg.out_root.glob("*/tables/*_cn_savg_summary.csv"))

    if not summary_files:
        return pd.DataFrame()

    selected_set = None
    if selected_elements:
        selected_set = {str(x).strip() for x in selected_elements if str(x).strip()}

    frames = []
    for f in summary_files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue

            element = f.name.replace("_cn_savg_summary.csv", "")

            if selected_set is not None and element not in selected_set:
                continue

            df["element"] = element

            df["CN"] = pd.to_numeric(df["CN"], errors="coerce")
            df["n_samples"] = pd.to_numeric(df["n_samples"], errors="coerce")
            df["s_avg_approx"] = pd.to_numeric(df["s_avg_approx"], errors="coerce")

            df = df.dropna(subset=["CN", "n_samples", "s_avg_approx"]).copy()
            df = df[df["n_samples"] > cfg.min_cn_count_for_savg].copy()
            df["savg_class"] = df["s_avg_approx"].apply(
                lambda x: classify_savg_threshold(x, threshold=1.0)
            )

            if not df.empty:
                frames.append(df)

        except Exception as e:
            print(f"[WARN] failed to read {f}: {e}")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["element", "CN"]).reset_index(drop=True)
    return out


def plot_combined_savg_subplots(
    cfg: AnalysisConfig,
    selected_elements: Optional[List[str]] = None,
    output_name: Optional[str] = None,
    ncols: int = 3,
):
    """
    如果 selected_elements=None，则画所有元素。
    如果 selected_elements=['Co','Cr','Fe']，则只画这几个元素。
    """
    all_df = collect_all_elements_savg_tables(cfg, selected_elements=selected_elements)

    if all_df.empty:
        print(f"[WARN] No s_avg rows found for selected elements: {selected_elements}")
        return

    elements = sorted(all_df["element"].unique().tolist())
    n = len(elements)

    if ncols < 1:
        ncols = 1
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    if selected_elements is None:
        all_df.to_csv(cfg.combined_savg_subplot_table_out, index=False)
        print(f"[SAVE] combined subplot s_avg table -> {cfg.combined_savg_subplot_table_out}")

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(8 * ncols, 5 * nrows),
        squeeze=False
    )

    axes_flat = axes.flatten()

    for ax_idx, element in enumerate(elements):
        ax = axes_flat[ax_idx]
        df_plot = all_df[all_df["element"] == element].copy().sort_values("CN")

        ax.plot(
            df_plot["CN"],
            df_plot["s_avg_approx"],
            linestyle="-",
            linewidth=2,
            color="gray",
            alpha=0.8,
            zorder=1
        )

        df_gt = df_plot[df_plot["s_avg_approx"] > 1].copy()
        df_le = df_plot[df_plot["s_avg_approx"] <= 1].copy()

        if not df_gt.empty:
            ax.scatter(
                df_gt["CN"],
                df_gt["s_avg_approx"],
                s=140,
                color="crimson",
                edgecolor="black",
                linewidth=0.7,
                zorder=3,
                label="s_avg > 1"
            )

        if not df_le.empty:
            ax.scatter(
                df_le["CN"],
                df_le["s_avg_approx"],
                s=140,
                color="royalblue",
                edgecolor="black",
                linewidth=0.7,
                zorder=3,
                label="s_avg ≤ 1"
            )

        for _, row in df_plot.iterrows():
            ax.text(
                row["CN"],
                row["s_avg_approx"],
                f"n={int(row['n_samples'])}",
                fontsize=9,
                ha="left",
                va="bottom"
            )

        ax.axhline(
            y=1.0,
            linestyle="--",
            linewidth=1.2,
            color="black",
            alpha=0.8
        )

        ax.set_title(f"{element}")
        ax.set_xlabel("Coordination Number (CN)")
        ax.set_ylabel("s_avg_approx")

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            seen = set()
            uniq_h = []
            uniq_l = []
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen.add(l)
                    uniq_h.append(h)
                    uniq_l.append(l)
            ax.legend(uniq_h, uniq_l, loc="best", fontsize=10)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    if selected_elements:
        title_text = ", ".join(elements)
        fig.suptitle(
            f"s_avg vs CN subplots: {title_text}",
            fontsize=18,
            y=0.99
        )
    else:
        fig.suptitle(
            f"All elements: s_avg vs CN subplots (only CN with n_samples > {cfg.min_cn_count_for_savg})",
            fontsize=18,
            y=0.99
        )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if output_name is None:
        if selected_elements:
            group_name = "_".join(elements)
            output_name = f"savg_subplots_{group_name}.png"
        else:
            output_name = f"all_elements_savg_subplots_min_count_gt_{cfg.min_cn_count_for_savg}.png"

    out_path = cfg.out_root / output_name
    save_plot(fig, out_path)
    plt.close(fig)

    print(f"[SAVE] grouped subplot figure -> {out_path}")


def plot_multiple_element_groups(
    cfg: AnalysisConfig,
    element_groups: List[List[str]],
    ncols: int = 3,
):
    if not element_groups:
        print("[WARN] No element groups provided.")
        return

    for i, group in enumerate(element_groups, start=1):
        group_clean = [str(x).strip() for x in group if str(x).strip()]
        if not group_clean:
            continue

        group_name = "_".join(group_clean)
        output_name = f"group_{i}_{sanitize_name(group_name)}.png"

        print(f"[INFO] plotting group {i}: {group_clean}")
        plot_combined_savg_subplots(
            cfg,
            selected_elements=group_clean,
            output_name=output_name,
            ncols=min(ncols, max(1, len(group_clean))),
        )


# =========================================================
# MAIN
# =========================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze local environments from cached CIF structures."
    )
    parser.add_argument("--element", required=True, help="Target element, e.g. V, Ti, Fe")
    parser.add_argument("--data-dir", default="data_all_R0Bs", help="Directory containing df_consist_{ELEMENT}.csv")
    parser.add_argument("--out-root", default="out/12", help="Output root directory")
    parser.add_argument("--nn-cutoff", type=float, default=3.2, help="O neighbor cutoff")
    parser.add_argument("--metal-search-cutoff", type=float, default=2.6, help="Cutoff for bridging O check")
    parser.add_argument("--min-cn-count-for-savg", type=int, default=20, help="Minimum sample count for s_avg plot")

    # 新增：只画拼图，不做当前元素的单元素分析
    parser.add_argument(
        "--plot-only-combined",
        action="store_true",
        help="Only generate combined/grouped subplot figures from existing summary CSV files"
    )

    # 新增：单元素分析时跳过总拼图，适合 batch 内部调用
    parser.add_argument(
        "--skip-combined-subplots",
        action="store_true",
        help="Skip combined/grouped subplot plotting in this run"
    )

    # 新增：只画指定元素的一张 subplot 图
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

    # 新增：一次画多组图
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

    args = parser.parse_args()

    cfg = AnalysisConfig(
        element=args.element,
        data_dir=Path(args.data_dir),
        out_root=Path(args.out_root),
        nn_cutoff=args.nn_cutoff,
        metal_search_cutoff=args.metal_search_cutoff,
        min_cn_count_for_savg=args.min_cn_count_for_savg,
    )

    ensure_dirs(cfg)

    # -------------------------------------------------
    # 模式 1：正常做当前元素分析
    # -------------------------------------------------
    if not args.plot_only_combined:
        print(f"[INFO] loading data for element = {cfg.element}")
        d = load_element_table(cfg)

        if d.empty:
            print("[WARN] no valid rows found.")
            return

        save_input_tables(d, cfg)

        detail_df = build_detailed_descriptor_table(d, cfg)
        save_analysis_summaries(detail_df, d, cfg)
        make_plots(detail_df, cfg)

        # 单元素图
        plot_savg_from_summary(cfg)

    # -------------------------------------------------
    # 模式 2：生成总拼图 / 分组图
    # -------------------------------------------------
    if not args.skip_combined_subplots:
        element_groups = parse_subplot_groups(args.subplot_groups)

        if element_groups:
            plot_multiple_element_groups(
                cfg,
                element_groups=element_groups,
                ncols=args.combined_ncols,
            )
        elif args.subplot_elements:
            plot_combined_savg_subplots(
                cfg,
                selected_elements=args.subplot_elements,
                output_name=args.subplot_output_name,
                ncols=args.combined_ncols,
            )
        else:
            plot_combined_savg_subplots(
                cfg,
                selected_elements=None,
                output_name=args.subplot_output_name,
                ncols=args.combined_ncols,
            )

    print("\n[DONE] Local environment analysis finished successfully.")


if __name__ == "__main__":
    main()