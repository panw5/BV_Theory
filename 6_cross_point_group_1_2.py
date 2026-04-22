import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Settings
# ============================================================
INPUT_CSV = Path("out/1_fit_R0B_group_1_2/tables/beta_beta0_all_methods.csv")
OUT_DIR = Path("out/6_cross_point_group_1_2")

METHOD_TO_USE = "RANSAC"   # choose from: OLS, Huber, RANSAC

GROUP1 = ["Li", "Na", "K", "Rb", "Cs", "Fr"]
GROUP2 = ["Be", "Mg", "Ca", "Sr", "Ba", "Ra"]

# Manual axis limits for extended-line plots
GROUP1_XLIM = (-1.2, 0.1)
GROUP1_YLIM = (0.0, 3.0)

GROUP2_XLIM = (-2.2, 0.1)
GROUP2_YLIM = (0.0, 3.5)

# Marker map for CN
MARKERS = ["o", "s", "^", "D", "P", "X", "v", ">", "<", "*", "h", "8"]


# ============================================================
# Utilities
# ============================================================
def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Cannot find input file: {INPUT_CSV.resolve()}")

    df = pd.read_csv(INPUT_CSV)

    required = {"element", "CN", "method", "beta", "beta0"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["method"] == METHOD_TO_USE].copy()
    if df.empty:
        raise ValueError(f"No rows found for method={METHOD_TO_USE}")

    df["beta"] = pd.to_numeric(df["beta"], errors="coerce")
    df["beta0"] = pd.to_numeric(df["beta0"], errors="coerce")
    df["CN"] = pd.to_numeric(df["CN"], errors="coerce")

    df = df.dropna(subset=["element", "beta", "beta0", "CN"]).copy()
    return df


def build_cn_marker_map(cn_values):
    cn_sorted = sorted(cn_values)
    return {cn: MARKERS[i % len(MARKERS)] for i, cn in enumerate(cn_sorted)}


def fit_element_line(g):
    """
    Fit beta0 = c1 * beta + c2 for one element.
    Return c1, c2 or (None, None) if fitting is not possible.
    """
    x = g["beta"].to_numpy(dtype=float)
    y = g["beta0"].to_numpy(dtype=float)

    if len(x) < 2:
        return None, None
    if np.std(x) < 1e-12:
        return None, None

    c1, c2 = np.polyfit(x, y, 1)
    return float(c1), float(c2)


def annotate_cn(ax, x, y, cn, i=0, dx=0.015, dy=0.03):
    offsets = [
        (+dx, +dy, "left", "bottom"),
        (-dx, +dy, "right", "bottom"),
        (+dx, -dy, "left", "top"),
        (-dx, -dy, "right", "top"),
        (0.0, +dy, "center", "bottom"),
        (0.0, -dy, "center", "top"),
    ]
    ox, oy, ha, va = offsets[i % len(offsets)]
    label = str(int(cn)) if abs(cn - int(cn)) < 1e-9 else f"{cn:g}"
    ax.text(x + ox, y + oy, label, fontsize=9, ha=ha, va=va)


def plot_group_extended_lines(df, elements, title, out_path, xlim, ylim):
    sub = df[df["element"].isin(elements)].copy()
    if sub.empty:
        print(f"[WARN] No data for {title}")
        return

    all_cn = sorted(sub["CN"].unique())
    cn_to_marker = build_cn_marker_map(all_cn)

    elements_present = [e for e in elements if e in sub["element"].unique()]
    color_map = {e: plt.cm.tab10(i % 10) for i, e in enumerate(elements_present)}

    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=200)

    fit_rows = []

    for element in elements_present:
        g = sub[sub["element"] == element].copy().sort_values("CN")
        color = color_map[element]

        c1, c2 = fit_element_line(g)
        if c1 is not None:
            x_line = np.linspace(xlim[0], xlim[1], 300)
            y_line = c1 * x_line + c2

            ax.plot(
                x_line, y_line,
                linestyle="--",
                linewidth=1.8,
                color=color,
                alpha=0.95,
                label=element
            )

            fit_rows.append({
                "element": element,
                "c1": c1,
                "c2": c2,
                "n_points": len(g)
            })

        for i, (_, r) in enumerate(g.iterrows()):
            cn = float(r["CN"])
            ax.scatter(
                r["beta"], r["beta0"],
                s=60,
                marker=cn_to_marker[cn],
                facecolors=color,
                edgecolors="black",
                linewidths=0.6,
                alpha=0.95
            )
            annotate_cn(ax, r["beta"], r["beta0"], cn, i=i)

    ax.set_title(title, fontsize=15)
    ax.set_xlabel(r"$\beta$", fontsize=15)
    ax.set_ylabel(r"$\beta_0$ (Å)", fontsize=15)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=11)

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)

    # CN legend
    cn_handles = []
    cn_labels = []
    for cn in all_cn:
        cn_handles.append(
            plt.Line2D(
                [0], [0],
                marker=cn_to_marker[cn],
                linestyle="",
                markerfacecolor="gray",
                markeredgecolor="black",
                markersize=9
            )
        )
        cn_labels.append(str(int(cn)) if abs(cn - int(cn)) < 1e-9 else f"{cn:g}")

    leg1 = ax.legend(
        cn_handles, cn_labels,
        title="C.N.",
        frameon=False,
        fontsize=10,
        title_fontsize=11,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0)
    )
    ax.add_artist(leg1)

    # Element legend
    element_handles = [
        plt.Line2D([0], [0], linestyle="--", color=color_map[e], linewidth=2)
        for e in elements_present
    ]
    ax.legend(
        element_handles, elements_present,
        title="Element",
        frameon=False,
        fontsize=10,
        title_fontsize=11,
        loc="upper left",
        bbox_to_anchor=(1.16, 1.0)
    )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_path}")

    fit_df = pd.DataFrame(fit_rows)
    fit_csv = out_path.with_suffix(".csv")
    fit_df.to_csv(fit_csv, index=False)
    print(f"[SAVE] {fit_csv}")


# ============================================================
# Main
# ============================================================
def main():
    ensure_dirs()
    df = load_data()

    print(f"[INFO] Method used: {METHOD_TO_USE}")
    print(f"[INFO] Total rows: {len(df)}")

    plot_group_extended_lines(
        df=df,
        elements=GROUP1,
        title=f"Group 1: extended fitted lines in β₀ vs β ({METHOD_TO_USE})",
        out_path=OUT_DIR / f"GROUP1_extended_lines_{METHOD_TO_USE}.png",
        xlim=GROUP1_XLIM,
        ylim=GROUP1_YLIM
    )

    plot_group_extended_lines(
        df=df,
        elements=GROUP2,
        title=f"Group 2: extended fitted lines in β₀ vs β ({METHOD_TO_USE})",
        out_path=OUT_DIR / f"GROUP2_extended_lines_{METHOD_TO_USE}.png",
        xlim=GROUP2_XLIM,
        ylim=GROUP2_YLIM
    )

    print(f"[DONE] All outputs written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()