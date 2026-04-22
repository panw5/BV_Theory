import os
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =========================
# 1. Basic settings
# =========================
element_groups = {
    "main_group": ["C", "N", "Si", "P", "S", "Ge", "As", "Se", "Te"],
    "positive_slope_group": ["V", "Cr", "Mo", "Re", "W"],
}

target_cn_map = {
    "C": [2, 3],
    "N": [3],
    "Si": [4],
    "Ge": [4],
    "P": [2, 3, 4],
    "S": [3, 4],
    "As": [3, 4],
    "Se": [3, 4],
    "Te": [3, 4],
    "V": [3, 4],
    "Cr": [4],
    "Mo": [4, 5],
    "Re": [4, 6],
    "W": [4, 5],
}

base_dir = os.path.join(".", "data_all_R0Bs")
output_dir = os.path.join(".", "out", "12-04")
os.makedirs(output_dir, exist_ok=True)

drop_no_data = True
top_n_geometries = 8

save_expanded_debug_csv = False
save_pivot_csv = False

use_geometry_cn_filter = False
annotate_dominant_geometry = True


geometry_expected_cn = {
    "Single neighbor": 1,
    "Linear": 2,
    "Angular": 2,
    "Trigonal plane": 3,
    "Trigonal non-coplanar": 3,
    "Tetrahedron": 4,
    "Square plane": 4,
    "Square non-coplanar": 4,
    "See-saw": 4,
    "Square pyramid": 5,
    "Trigonal bipyramid": 5,
    "Pentagonal plane": 5,
    "Pentagonal pyramid": 6,
    "Octahedron": 6,
    "Trigonal prism": 6,
    "Pentagonal bipyramid": 7,
    "Hexagonal bipyramid": 8,
}


# =========================
# 2. Utility functions
# =========================
def normalize_geometry_name(g: str) -> str:
    g = str(g).strip()
    replacements = {
        "no_data": "no_data",
    }
    return replacements.get(g, g)


def geometry_matches_cn(geometry: str, cn: int) -> bool:
    if geometry not in geometry_expected_cn:
        return True
    return geometry_expected_cn[geometry] == cn


def load_and_expand_element(elem: str, debug_subdir: str) -> pd.DataFrame:
    csv_path = os.path.join(base_dir, f"df_consist_{elem}.csv")

    if not os.path.exists(csv_path):
        print(f"[Warning] File not found: {csv_path}")
        return pd.DataFrame()

    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path)

    cn_col = f"mean_coordn_{elem}-O"
    geom_col = f"{elem}_chemenv"

    needed_cols = ["material_id", "chem_formula", cn_col, geom_col]
    missing = [c for c in [cn_col, geom_col] if c not in df.columns]
    if missing:
        print(f"[Warning] Missing columns in {csv_path}: {missing}")
        return pd.DataFrame()

    existing_cols = [c for c in needed_cols if c in df.columns]
    df = df[existing_cols].copy()

    df[cn_col] = pd.to_numeric(df[cn_col], errors="coerce")
    df[geom_col] = df[geom_col].astype(str).str.strip()
    df = df.dropna(subset=[cn_col, geom_col])

    expanded_rows = []

    for _, row in df.iterrows():
        cn_mean = row[cn_col]
        geom_text = str(row[geom_col]).strip()

        if geom_text == "" or geom_text.lower() == "nan":
            continue

        cn = int(round(cn_mean))

        geom_list = [normalize_geometry_name(g) for g in geom_text.split(",") if g.strip()]

        if drop_no_data:
            geom_list = [g for g in geom_list if g.lower() != "no_data"]

        if len(geom_list) == 0:
            continue

        split_weight = 1.0 / len(geom_list)

        for geom in geom_list:
            if use_geometry_cn_filter and not geometry_matches_cn(geom, cn):
                continue

            expanded_rows.append({
                "element": elem,
                "material_id": row["material_id"] if "material_id" in row else None,
                "chem_formula": row["chem_formula"] if "chem_formula" in row else None,
                "CN_mean": cn_mean,
                "CN": cn,
                "geometry": geom,
                "weight": split_weight,
                "raw_geometry_text": geom_text,
            })

    df_expanded = pd.DataFrame(expanded_rows)

    if not df_expanded.empty and save_expanded_debug_csv:
        out_debug = os.path.join(debug_subdir, f"{elem}_expanded_geometry_debug.csv")
        df_expanded.to_csv(out_debug, index=False)
        print(f"Saved debug expanded rows: {out_debug}")

    return df_expanded


def summarize_geometry(df_all: pd.DataFrame) -> pd.DataFrame:
    df_sum = (
        df_all.groupby(["element", "CN", "geometry"], as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "count"})
    )

    df_sum["fraction"] = (
        df_sum.groupby(["element", "CN"])["count"]
        .transform(lambda x: x / x.sum())
    )

    return df_sum


def reduce_to_top_geometries(df_sum: pd.DataFrame, elements: list[str]) -> pd.DataFrame:
    filtered_list = []

    for elem in elements:
        df_elem = df_sum[df_sum["element"] == elem].copy()
        if df_elem.empty:
            continue

        geom_rank = (
            df_elem.groupby("geometry", as_index=False)["count"]
            .sum()
            .sort_values("count", ascending=False)
        )

        keep_geoms = geom_rank["geometry"].head(top_n_geometries).tolist()

        df_elem["geometry"] = df_elem["geometry"].apply(
            lambda g: g if g in keep_geoms else "Others"
        )

        df_elem = (
            df_elem.groupby(["element", "CN", "geometry"], as_index=False)["count"]
            .sum()
        )

        df_elem["fraction"] = (
            df_elem.groupby(["element", "CN"])["count"]
            .transform(lambda x: x / x.sum())
        )

        filtered_list.append(df_elem)

    if len(filtered_list) == 0:
        return pd.DataFrame()

    return pd.concat(filtered_list, ignore_index=True)


def annotate_dominant(ax, pivot: pd.DataFrame):
    for i, cn in enumerate(pivot.index):
        row = pivot.loc[cn]
        dominant_fraction = row.max()

        if dominant_fraction <= 0:
            continue

        label = f"{dominant_fraction:.2f}"

        ax.text(
            i,
            1.02,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0
        )


def build_geometry_colors(all_geoms: list[str]) -> dict:
    color_pool = []
    for cmap_name in ["tab20", "tab20b", "tab20c", "Set3", "Paired", "Accent", "Dark2"]:
        cmap = plt.get_cmap(cmap_name)
        if hasattr(cmap, "colors"):
            color_pool.extend(list(cmap.colors))
        else:
            color_pool.extend([cmap(i / 20) for i in range(20)])

    if len(all_geoms) > len(color_pool):
        raise ValueError(
            f"Not enough unique colors: {len(all_geoms)} geometries found, "
            f"but only {len(color_pool)} colors available."
        )

    geometry_colors = {g: color_pool[i] for i, g in enumerate(all_geoms)}
    geometry_colors["Others"] = "#cccccc"
    return geometry_colors


def export_target_geometry_summary(df_plot: pd.DataFrame, elements: list[str], group_output_dir: str, group_name: str):
    rows = []

    for elem in elements:
        df_elem = df_plot[df_plot["element"] == elem].copy()
        if df_elem.empty or elem not in target_cn_map:
            continue

        pivot = df_elem.pivot_table(
            index="CN",
            columns="geometry",
            values="fraction",
            aggfunc="sum",
            fill_value=0
        ).sort_index()

        for cn in target_cn_map[elem]:
            if cn not in pivot.index:
                continue

            row = pivot.loc[cn]
            dominant_geometry = row.idxmax()
            dominant_fraction = row.max()

            rows.append({
                "element": elem,
                "CN": cn,
                "dominant_geometry": dominant_geometry,
                "fraction": dominant_fraction
            })

    if not rows:
        print(f"[Warning] No target dominant geometry rows found for group: {group_name}")
        return

    df_out = pd.DataFrame(rows).sort_values(["element", "CN"]).reset_index(drop=True)

    csv_path = os.path.join(group_output_dir, f"target_dominant_geometry_{group_name}.csv")
    txt_path = os.path.join(group_output_dir, f"target_dominant_geometry_{group_name}.txt")

    df_out.to_csv(csv_path, index=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        for _, r in df_out.iterrows():
            f.write(f"{r['element']}  CN={r['CN']}: {r['dominant_geometry']} ({r['fraction']:.2f})\n")

    print(f"Saved: {csv_path}")
    print(f"Saved: {txt_path}")


def plot_one_group(group_name: str, elements: list[str]):
    print(f"\n========== Processing group: {group_name} ==========")

    group_output_dir = os.path.join(output_dir, group_name)
    os.makedirs(group_output_dir, exist_ok=True)

    all_rows = []

    for elem in elements:
        df_expanded = load_and_expand_element(elem, group_output_dir)
        if not df_expanded.empty:
            all_rows.append(df_expanded)

    if len(all_rows) == 0:
        print(f"[Warning] No valid data loaded for group: {group_name}")
        return

    df_all = pd.concat(all_rows, ignore_index=True)

    all_geoms = sorted(df_all["geometry"].dropna().unique().tolist())

    print("\nAll geometries found:")
    for g in all_geoms:
        print(g)

    geometry_colors = build_geometry_colors(all_geoms)

    df_sum = summarize_geometry(df_all)
    df_plot = reduce_to_top_geometries(df_sum, elements)

    if df_plot.empty:
        print(f"[Warning] No valid summarized data to plot for group: {group_name}")
        return

    # ===== export target dominant geometry summary =====
    export_target_geometry_summary(df_plot, elements, group_output_dir, group_name)

    n = len(elements)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 4.8 * nrows))
    axes = axes.flatten()

    all_geometries_used = []

    for i, elem in enumerate(elements):
        ax = axes[i]
        df_elem = df_plot[df_plot["element"] == elem].copy()

        if df_elem.empty:
            ax.set_title(f"{elem} (no data)")
            ax.axis("off")
            continue

        pivot = df_elem.pivot_table(
            index="CN",
            columns="geometry",
            values="fraction",
            aggfunc="sum",
            fill_value=0
        ).sort_index()

        if save_pivot_csv:
            pivot_path = os.path.join(group_output_dir, f"{elem}_geometry_pivot.csv")
            pivot.to_csv(pivot_path)
            print(f"Saved pivot table: {pivot_path}")

        preferred_order = [g for g in all_geoms if g in pivot.columns]
        remaining = [g for g in pivot.columns if g not in preferred_order]
        ordered_cols = preferred_order + sorted(remaining)
        pivot = pivot[ordered_cols]

        colors = [geometry_colors.get(g, "#999999") for g in pivot.columns]

        pivot.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            width=0.8,
            color=colors
        )

        ax.set_title(elem, fontsize=14)
        ax.set_xlabel("Coordination Number (CN)")
        ax.set_ylabel("Fraction")
        ax.set_ylim(0, 1.10 if annotate_dominant_geometry else 1.0)

        if annotate_dominant_geometry:
            annotate_dominant(ax, pivot)

        if ax.legend_ is not None:
            ax.legend_.remove()

        for g in pivot.columns:
            if g not in all_geometries_used:
                all_geometries_used.append(g)

    for j in range(len(elements), len(axes)):
        fig.delaxes(axes[j])

    legend_handles = [
        Patch(facecolor=geometry_colors.get(g, "#999999"), edgecolor="none", label=g)
        for g in all_geometries_used
    ]

    fig.legend(
        handles=legend_handles,
        labels=all_geometries_used,
        loc="center right",
        bbox_to_anchor=(1.03, 0.5),
        frameon=True
    )

    fig.suptitle(
        f"Geometry distribution across coordination numbers for {', '.join(elements)}",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 0.88, 0.95])

    output_png = os.path.join(group_output_dir, f"geometry_distribution_{group_name}.png")
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {output_png}")


# =========================
#  Run all groups
# =========================
for group_name, elems in element_groups.items():
    plot_one_group(group_name, elems)