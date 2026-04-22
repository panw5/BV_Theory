import re
import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor


# ===================== Settings =====================
DATA_DIR = Path("data_group_13_14")
OUT_DIR = Path("out/3_fit_R0B_group_13_14")

DROP_CN_ZERO = True
KEEP_CN_LIST = None

MIN_POINTS_PER_CN = 10

MIN_CN_FITS_PER_ELEMENT_GROUPPLOT = 3
ALLOW_SINGLE_CN_IN_GROUPPLOT = True

TEXT_OFFSET_FRAC = (0.012, 0.012)

CN_CMAP = plt.cm.tab20
ELEMENT_CMAP = plt.cm.tab20

GROUP13 = {"Al", "Ga", "In", "Tl"}
GROUP14 = {"Ge", "Sn", "Pb"}

METHODS = ["OLS", "Huber", "RANSAC"]


# ---------------- Output folders ----------------
DIR_TABLES = OUT_DIR / "tables"

DIR_BETA_GROUPED = OUT_DIR / "beta0_vs_beta" / "grouped"
DIR_BETA_PER_ELEMENT = OUT_DIR / "beta0_vs_beta" / "per_element"

DIR_R0B_PANELS = OUT_DIR / "B_vs_R0_panels"
DIR_R0B_CN_GRID = OUT_DIR / "B_vs_R0_by_CN_grid"


def ensure_output_dirs():
    for d in [
        OUT_DIR,
        DIR_TABLES,
        DIR_BETA_GROUPED,
        DIR_BETA_PER_ELEMENT,
        DIR_R0B_PANELS,
        DIR_R0B_CN_GRID,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def parse_list_cell(x):
    """Parse a cell that may contain list-like content into a list[float]."""
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


def find_columns_basic(df: pd.DataFrame):
    """Detect R0, B, and *O_coordn columns; infer element symbol."""
    r0_candidates = [c for c in df.columns if "R$_0$" in c or c.strip() in ["R0", "R_0", "R$_0$"]]
    if not r0_candidates:
        raise ValueError("R0 column not found (expected 'R$_0$' or 'R0').")
    r0_col = r0_candidates[0]

    if "B" not in df.columns:
        raise ValueError("'B' column not found.")
    b_col = "B"

    o_candidates = [c for c in df.columns if c.endswith("O_coordn")]
    if not o_candidates:
        raise ValueError("*O_coordn column not found.")
    o_col = o_candidates[0]

    m = re.match(r"([A-Z][a-z]?)O_coordn$", o_col)
    element = m.group(1) if m else "M"

    return element, r0_col, b_col, o_col


def find_columns_with_charge(df: pd.DataFrame):
    """Detect R0, B, *O_coordn, and mean_*_charge columns; infer element symbol."""
    element, r0_col, b_col, o_col = find_columns_basic(df)

    charge_candidates = [c for c in df.columns if re.fullmatch(r"mean_.*_charge", c)]
    if not charge_candidates:
        raise ValueError("mean_*_charge column not found.")
    charge_col = charge_candidates[0]

    return element, r0_col, b_col, o_col, charge_col


def apply_cn_filters(d_cn: pd.DataFrame):
    """Apply CN filters configured in settings."""
    if DROP_CN_ZERO:
        d_cn = d_cn[d_cn["CN"] != 0]
    if KEEP_CN_LIST is not None:
        keep = set(float(v) for v in KEEP_CN_LIST)
        d_cn = d_cn[d_cn["CN"].isin(keep)]
    return d_cn


def clean_xy(x, y):
    """Drop non-finite values and return contiguous float arrays."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def is_degenerate_x(x, eps=1e-12):
    """True if x has near-zero variance."""
    x = np.asarray(x, dtype=float).ravel()
    if len(x) < 2:
        return True
    return float(np.nanstd(x)) < eps


def fit_line_ols(x, y):
    """OLS via numpy polyfit: y = beta*x + beta0."""
    if len(x) < 2:
        return None, None
    beta, beta0 = np.polyfit(x, y, 1)
    return float(beta), float(beta0)


def fit_line_huber(x, y):
    """Huber regression."""
    if len(x) < 2:
        return None, None
    X = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)

    model = HuberRegressor(epsilon=1.35, max_iter=500, alpha=0.0)
    model.fit(X, y)
    return float(model.coef_[0]), float(model.intercept_)


def fit_line_ransac(x, y):
    """RANSAC regression."""
    if len(x) < 2:
        return None, None
    X = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)

    base = LinearRegression()
    min_samples = max(2, int(0.5 * len(x)))
    model = RANSACRegressor(
        estimator=base,
        min_samples=min_samples,
        residual_threshold=None,
        max_trials=200,
        random_state=0
    )
    model.fit(X, y)
    return float(model.estimator_.coef_[0]), float(model.estimator_.intercept_)


def fit_by_method(method, x, y):
    """Return (beta, beta0) or (None, None) on failure."""
    x, y = clean_xy(x, y)

    if len(x) < 2:
        return None, None
    if is_degenerate_x(x):
        return None, None

    method_u = method.strip().upper()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)

            if method_u == "OLS":
                return fit_line_ols(x, y)
            if method_u == "HUBER":
                return fit_line_huber(x, y)
            if method_u == "RANSAC":
                return fit_line_ransac(x, y)

        raise ValueError(f"Unknown method: {method}")
    except Exception:
        return None, None


def build_cn_style_maps(all_cn_values):
    """Create consistent CN marker and color-index maps."""
    markers = ["o", "s", "^", "D", "P", "X", "v", ">", "<", "*", "h", "8"]
    cn_sorted = sorted(all_cn_values)
    cn_to_marker = {cn: markers[i % len(markers)] for i, cn in enumerate(cn_sorted)}
    cn_to_cidx = {cn: i for i, cn in enumerate(cn_sorted)}
    return cn_to_marker, cn_to_cidx


def cn_color(cn, cn_to_cidx):
    """Map CN to an RGBA color."""
    return CN_CMAP(cn_to_cidx[cn] % CN_CMAP.N)


def annotate_cn(ax, x, y, cn, xlim, ylim, i=0):
    """Annotate CN near a point with cycling offsets."""
    dx0 = (xlim[1] - xlim[0]) * TEXT_OFFSET_FRAC[0]
    dy0 = (ylim[1] - ylim[0]) * TEXT_OFFSET_FRAC[1]
    offsets = [
        (+dx0, +dy0, "left", "bottom"),
        (-dx0, +dy0, "right", "bottom"),
        (+dx0, -dy0, "left", "top"),
        (-dx0, -dy0, "right", "top"),
        (0.0, +dy0, "center", "bottom"),
        (0.0, -dy0, "center", "top"),
        (+dx0, 0.0, "left", "center"),
        (-dx0, 0.0, "right", "center"),
    ]
    dx, dy, ha, va = offsets[i % len(offsets)]
    cn_label = str(int(cn)) if abs(cn - int(cn)) < 1e-9 else f"{cn:g}"
    ax.text(x + dx, y + dy, cn_label, fontsize=9, ha=ha, va=va)


def _add_two_column_legends(ax, element_handles, element_labels, cn_handles, cn_labels):
    """Place CN legend and element legend as two separate columns."""
    cn_legend = ax.legend(
        cn_handles,
        cn_labels,
        title="C.N.",
        frameon=False,
        fontsize=10,
        title_fontsize=11,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0
    )
    ax.add_artist(cn_legend)

    ax.legend(
        element_handles,
        element_labels,
        title="Element",
        frameon=False,
        fontsize=10,
        title_fontsize=11,
        loc="upper left",
        bbox_to_anchor=(1.20, 1.0),
        borderaxespad=0.0
    )


def _plot_group(beta_df: pd.DataFrame, elements, out_path: Path, title: str, method: str):
    """Grouped beta0-vs-beta plot for selected elements."""
    requested = sorted(set(elements))
    beta_plot = beta_df[(beta_df["element"].isin(requested)) & (beta_df["method"] == method)].copy()

    if beta_plot.empty:
        print(f"[WARN] {title} ({method}): no data to plot.")
        return

    cn_fit_counts = beta_plot.groupby("element")["CN"].nunique().to_dict()

    keep_elements = []
    dropped = []
    for e in requested:
        nfits = int(cn_fit_counts.get(e, 0))
        if ALLOW_SINGLE_CN_IN_GROUPPLOT:
            if nfits >= 1:
                keep_elements.append(e)
            else:
                dropped.append((e, nfits))
        else:
            if nfits >= MIN_CN_FITS_PER_ELEMENT_GROUPPLOT:
                keep_elements.append(e)
            else:
                dropped.append((e, nfits))

    for e, nfits in dropped:
        print(f"[WARN] {title} ({method}): {e} not shown (CN fits = {nfits}).")

    beta_plot = beta_plot[beta_plot["element"].isin(keep_elements)].copy()
    if beta_plot.empty:
        print(f"[WARN] {title} ({method}): all elements filtered out.")
        return

    all_cn = sorted(beta_plot["CN"].unique())
    cn_to_marker, _ = build_cn_style_maps(all_cn)

    elements_sorted = sorted(beta_plot["element"].unique())
    element_to_idx = {e: i for i, e in enumerate(elements_sorted)}

    fig, ax = plt.subplots(figsize=(10.8, 6.5), dpi=200)

    x_min, x_max = beta_plot["beta"].min(), beta_plot["beta"].max()
    y_min, y_max = beta_plot["beta0"].min(), beta_plot["beta0"].max()
    padx = 0.06 * (x_max - x_min) if x_max > x_min else 0.5
    pady = 0.06 * (y_max - y_min) if y_max > y_min else 0.5
    xlim = (x_min - padx, x_max + padx)
    ylim = (y_min - pady, y_max + pady)

    element_handles, element_labels = [], []
    for element, g in beta_plot.groupby("element"):
        g = g.sort_values("CN")
        color = ELEMENT_CMAP(element_to_idx[element] % ELEMENT_CMAP.N)

        h, = ax.plot(
            g["beta"],
            g["beta0"],
            linestyle="--",
            linewidth=1.8,
            color=color,
            alpha=0.95
        )
        element_handles.append(h)
        element_labels.append(element)

        for i, (_, r) in enumerate(g.iterrows()):
            cn = float(r["CN"])
            ax.scatter(
                r["beta"],
                r["beta0"],
                s=55,
                marker=cn_to_marker[cn],
                facecolors=color,
                edgecolors="black",
                linewidths=0.6,
                alpha=0.95
            )
            annotate_cn(ax, r["beta"], r["beta0"], cn, xlim, ylim, i=i)

    ax.set_title(f"{title} ({method})", fontsize=14)
    ax.set_xlabel(r"$\beta$", fontsize=14)
    ax.set_ylabel(r"$\beta_0$ (Å)", fontsize=14)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    cn_handles, cn_labels = [], []
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

    _add_two_column_legends(ax, element_handles, element_labels, cn_handles, cn_labels)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_one_element_beta0_vs_beta(beta_df: pd.DataFrame, element: str, out_path: Path, method: str):
    """Per-element beta0-vs-beta plot."""
    g = beta_df[(beta_df["element"] == element) & (beta_df["method"] == method)].sort_values("CN")
    if g.empty:
        return

    cn_vals = sorted(g["CN"].unique())
    cn_to_marker, cn_to_cidx = build_cn_style_maps(cn_vals)

    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=200)

    x_min, x_max = g["beta"].min(), g["beta"].max()
    y_min, y_max = g["beta0"].min(), g["beta0"].max()
    padx = 0.08 * (x_max - x_min) if x_max > x_min else 0.5
    pady = 0.08 * (y_max - y_min) if y_max > y_min else 0.5
    xlim = (x_min - padx, x_max + padx)
    ylim = (y_min - pady, y_max + pady)

    ax.plot(g["beta"], g["beta0"], linestyle="--", color="black", linewidth=2.0)

    for i, (_, r) in enumerate(g.iterrows()):
        cn = float(r["CN"])
        ax.scatter(
            r["beta"],
            r["beta0"],
            s=70,
            marker=cn_to_marker[cn],
            facecolors=cn_color(cn, cn_to_cidx),
            edgecolors="black",
            linewidths=0.7,
            alpha=0.95
        )
        annotate_cn(ax, r["beta"], r["beta0"], cn, xlim, ylim, i=i)

    ax.set_title(f"{element} ({method})", fontsize=18, fontweight="bold")
    ax.set_xlabel(r"$\beta$", fontsize=14)
    ax.set_ylabel(r"$\beta_0$ (Å)", fontsize=14)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    proxies, labels = [], []
    for cn in cn_vals:
        proxies.append(
            plt.Line2D(
                [0], [0],
                marker=cn_to_marker[cn],
                linestyle="",
                markeredgecolor="black",
                markerfacecolor=cn_color(cn, cn_to_cidx),
                markersize=9
            )
        )
        labels.append(str(int(cn)) if abs(cn - int(cn)) < 1e-9 else f"{cn:g}")

    ax.legend(proxies, labels, title="C.N.", frameon=False, fontsize=10, title_fontsize=11, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _line_on_axis(ax, x, beta, beta0, color="black"):
    """Draw y = beta*x + beta0 on given axes."""
    xg = np.linspace(np.nanmin(x), np.nanmax(x), 120)
    yg = beta * xg + beta0
    ax.plot(xg, yg, linestyle="--", linewidth=1.8, color=color)


def plot_element_R0_B_Z_CN(csv_path: Path, method: str):
    """Per-element 1x2 plot: left by Z, right by CN."""
    df = pd.read_csv(csv_path)
    element, r0_col, b_col, o_col, charge_col = find_columns_with_charge(df)

    d = df[[r0_col, b_col, o_col, charge_col]].copy()
    d = d.rename(columns={r0_col: "R0", b_col: "B", o_col: "CN_raw", charge_col: "Z"})
    d["R0"] = pd.to_numeric(d["R0"], errors="coerce")
    d["B"] = pd.to_numeric(d["B"], errors="coerce")
    d["Z"] = pd.to_numeric(d["Z"], errors="coerce")
    d = d.dropna(subset=["R0", "B"])

    d["CN_list"] = d["CN_raw"].apply(parse_list_cell)
    d_cn = d.explode("CN_list").rename(columns={"CN_list": "CN"})
    d_cn["CN"] = pd.to_numeric(d_cn["CN"], errors="coerce")
    d_cn = d_cn.dropna(subset=["CN"])
    d_cn = apply_cn_filters(d_cn)

    d_z = d.dropna(subset=["Z"]).copy()
    d_z["Z_cat"] = d_z["Z"].round(1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), dpi=200)

    z_vals = sorted(d_z["Z_cat"].unique())
    for z in z_vals:
        sub = d_z[d_z["Z_cat"] == z]
        ax1.scatter(sub["R0"], sub["B"], s=35, edgecolors="black", linewidths=0.6, alpha=0.85, label=f"{z:.1f}")
        x, y = clean_xy(sub["R0"].to_numpy(), sub["B"].to_numpy())
        beta, beta0 = fit_by_method(method, x, y)
        if beta is not None:
            _line_on_axis(ax1, x, beta, beta0, color="black")

    ax1.set_xlabel(r"R$_0$ (Å)", fontsize=14)
    ax1.set_ylabel(r"B (Å)", fontsize=14)
    ax1.legend(title="Z", frameon=False, loc="upper right")
    ax1.text(0.04, 0.08, element, transform=ax1.transAxes, fontsize=20, fontweight="bold")

    cn_vals = sorted(d_cn["CN"].unique())
    cn_to_marker, cn_to_cidx = build_cn_style_maps(cn_vals)

    for cn in cn_vals:
        sub = d_cn[d_cn["CN"] == cn]
        ax2.scatter(
            sub["R0"], sub["B"],
            s=40,
            marker=cn_to_marker[cn],
            facecolors=cn_color(cn, cn_to_cidx),
            edgecolors="black",
            linewidths=0.6,
            alpha=0.85,
            label=str(int(cn)) if abs(cn - int(cn)) < 1e-9 else f"{cn:g}"
        )
        x, y = clean_xy(sub["R0"].to_numpy(), sub["B"].to_numpy())
        beta, beta0 = fit_by_method(method, x, y)
        if beta is not None:
            _line_on_axis(ax2, x, beta, beta0, color="black")

    ax2.set_xlabel(r"R$_0$ (Å)", fontsize=14)
    ax2.set_ylabel(r"B (Å)", fontsize=14)
    ax2.legend(title="C.N.", frameon=False, loc="upper right")
    ax2.text(0.04, 0.08, element, transform=ax2.transAxes, fontsize=20, fontweight="bold")

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        ax.tick_params(labelsize=11)

    fig.tight_layout(w_pad=2.0)

    out_path = DIR_R0B_PANELS / f"{element}_R0_B_Z_CN_{method}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_element_R0_B_by_CN_grid(csv_path: Path, method: str, skipped_rows: list[dict]):
    """CN-split grid: one subplot per CN, with beta and beta0 annotation."""
    df = pd.read_csv(csv_path)
    element, r0_col, b_col, o_col = find_columns_basic(df)

    d = df[[r0_col, b_col, o_col]].copy()
    d = d.rename(columns={r0_col: "R0", b_col: "B", o_col: "CN_raw"})
    d["R0"] = pd.to_numeric(d["R0"], errors="coerce")
    d["B"] = pd.to_numeric(d["B"], errors="coerce")
    d = d.dropna(subset=["R0", "B"])

    d["CN_list"] = d["CN_raw"].apply(parse_list_cell)
    d_cn = d.explode("CN_list").rename(columns={"CN_list": "CN"}).dropna(subset=["CN"]).copy()
    d_cn["CN"] = pd.to_numeric(d_cn["CN"], errors="coerce")
    d_cn = d_cn.dropna(subset=["CN"])
    d_cn = apply_cn_filters(d_cn)

    if d_cn.empty:
        return None

    cn_vals = sorted(d_cn["CN"].unique())
    cn_vals = [cn for cn in cn_vals if len(d_cn[d_cn["CN"] == cn]) >= MIN_POINTS_PER_CN]
    if not cn_vals:
        return None

    n_cols = 4
    n_rows = int(np.ceil(len(cn_vals) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows), dpi=200)
    axes = np.array(axes).reshape(-1)

    x_global = d_cn["R0"].to_numpy()
    x_global, _ = clean_xy(x_global, x_global)
    xg = np.linspace(np.nanmin(x_global), np.nanmax(x_global), 120)

    fig.suptitle(f"{element}: B vs R$_0$ split by C.N. ({method})", fontsize=14, y=0.995)

    for i_ax, cn in enumerate(cn_vals):
        ax = axes[i_ax]
        sub = d_cn[d_cn["CN"] == cn]
        x = sub["R0"].to_numpy()
        y = sub["B"].to_numpy()
        x, y = clean_xy(x, y)

        ax.scatter(x, y, s=12, edgecolors="black", linewidths=0.3, alpha=0.85)

        beta, beta0 = fit_by_method(method, x, y)
        if beta is None:
            skipped_rows.append({
                "file": csv_path.name,
                "element": element,
                "CN": float(cn),
                "method": method,
                "n_points": int(len(x)),
                "reason": "fit_failed_or_degenerate"
            })
        else:
            yg = beta * xg + beta0
            ax.plot(xg, yg, linestyle="--", linewidth=1.2, color="tab:blue")

            ax.text(
                0.98, 0.92,
                f"C.N.={int(cn) if abs(cn-int(cn))<1e-9 else cn:g}\n"
                f"$\\beta$={beta:.2f} Å$^{{-1}}$\n"
                f"$\\beta_0$={beta0:.2f} Å",
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=8
            )

        if i_ax % n_cols == 0:
            ax.set_ylabel("B (Å)", fontsize=10)
        ax.set_xlabel(r"R$_0$ (Å)", fontsize=10)
        ax.tick_params(labelsize=8)

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    for j in range(len(cn_vals), len(axes)):
        axes[j].axis("off")

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    out_path = DIR_R0B_CN_GRID / f"{element}_R0_B_by_CN_grid_{method}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def extract_beta_pairs_from_csv(csv_path: Path, skipped_rows: list[dict]) -> pd.DataFrame:
    """Compute CN-resolved fits (B vs R0) for all methods and return tidy rows."""
    df = pd.read_csv(csv_path)
    element, r0_col, b_col, o_col = find_columns_basic(df)

    d = df[[r0_col, b_col, o_col]].copy()
    d = d.rename(columns={r0_col: "R0", b_col: "B", o_col: "CN_raw"})
    d["R0"] = pd.to_numeric(d["R0"], errors="coerce")
    d["B"] = pd.to_numeric(d["B"], errors="coerce")
    d = d.dropna(subset=["R0", "B"])

    d["CN_list"] = d["CN_raw"].apply(parse_list_cell)
    d_cn = d.explode("CN_list").rename(columns={"CN_list": "CN"}).dropna(subset=["CN"]).copy()
    d_cn["CN"] = pd.to_numeric(d_cn["CN"], errors="coerce")
    d_cn = d_cn.dropna(subset=["CN"])
    d_cn = apply_cn_filters(d_cn)

    if d_cn.empty:
        return pd.DataFrame(columns=["element", "CN", "method", "beta", "beta0", "n_points"])

    rows = []

    for cn in sorted(d_cn["CN"].unique()):
        sub = d_cn[d_cn["CN"] == cn]
        x = sub["R0"].to_numpy()
        y = sub["B"].to_numpy()
        x, y = clean_xy(x, y)
        n = int(len(x))

        if n < MIN_POINTS_PER_CN:
            for m in METHODS:
                skipped_rows.append({
                    "file": csv_path.name,
                    "element": element,
                    "CN": float(cn),
                    "method": m,
                    "n_points": n,
                    "reason": f"n_points<{MIN_POINTS_PER_CN}"
                })
            continue

        if is_degenerate_x(x):
            for m in METHODS:
                skipped_rows.append({
                    "file": csv_path.name,
                    "element": element,
                    "CN": float(cn),
                    "method": m,
                    "n_points": n,
                    "reason": "degenerate_x"
                })
            continue

        for m in METHODS:
            beta, beta0 = fit_by_method(m, x, y)
            if beta is None:
                skipped_rows.append({
                    "file": csv_path.name,
                    "element": element,
                    "CN": float(cn),
                    "method": m,
                    "n_points": n,
                    "reason": "fit_failed"
                })
                continue
            rows.append({
                "element": element,
                "CN": float(cn),
                "method": m,
                "beta": float(beta),
                "beta0": float(beta0),
                "n_points": n
            })

    return pd.DataFrame(rows)


def main():
    ensure_output_dirs()

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in: {DATA_DIR.resolve()}")

    skipped_rows = []

    frames = []
    for f in csv_files:
        try:
            df_beta = extract_beta_pairs_from_csv(f, skipped_rows)
            frames.append(df_beta)
            print(f"[OK] {f.name}: CN fits (rows) = {len(df_beta)}")
        except Exception as e:
            print(f"[FAIL] {f.name}: {e}")

    beta_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if beta_df.empty:
        raise RuntimeError("No fitted beta/beta0 rows were produced. Check filters and MIN_POINTS_PER_CN.")

    out_beta_csv = DIR_TABLES / "beta_beta0_all_methods.csv"
    beta_df.sort_values(["method", "element", "CN"]).to_csv(out_beta_csv, index=False)
    print(f"[SAVE] {out_beta_csv}")

    out_skip_csv = DIR_TABLES / "skipped_fits_log.csv"
    pd.DataFrame(skipped_rows).to_csv(out_skip_csv, index=False)
    print(f"[SAVE] {out_skip_csv}")

    all_elements = sorted(beta_df["element"].unique())
    group13 = sorted([e for e in all_elements if e in GROUP13])
    group14 = sorted([e for e in all_elements if e in GROUP14])

    for method in METHODS:
        out_all = DIR_BETA_GROUPED / f"ALL_ELEMENTS_beta0_vs_beta_by_CN_{method}.png"
        _plot_group(beta_df, all_elements, out_all, "All elements", method)
        print(f"[SAVE] {out_all}")

        if group13:
            out_g13 = DIR_BETA_GROUPED / f"GROUP13_beta0_vs_beta_by_CN_{method}.png"
            _plot_group(beta_df, group13, out_g13, "Group 13", method)
            print(f"[SAVE] {out_g13}")

        if group14:
            out_g14 = DIR_BETA_GROUPED / f"GROUP14_beta0_vs_beta_by_CN_{method}.png"
            _plot_group(beta_df, group14, out_g14, "Group 14", method)
            print(f"[SAVE] {out_g14}")

    for f in csv_files:
        try:
            df = pd.read_csv(f)
            element, *_ = find_columns_basic(df)
        except Exception:
            continue

        for method in METHODS:
            try:
                out1 = plot_element_R0_B_Z_CN(f, method)
                print(f"[SAVE] {out1}")
            except Exception as e:
                print(f"[WARN] {f.name} ({method}) R0_B_Z_CN: {e}")

            try:
                out2 = plot_element_R0_B_by_CN_grid(f, method, skipped_rows)
                if out2 is not None:
                    print(f"[SAVE] {out2}")
            except Exception as e:
                print(f"[WARN] {f.name} ({method}) CN_grid: {e}")

            try:
                out3 = DIR_BETA_PER_ELEMENT / f"{element}_beta0_vs_beta_by_CN_{method}.png"
                plot_one_element_beta0_vs_beta(beta_df, element, out3, method)
                if out3.exists():
                    print(f"[SAVE] {out3}")
            except Exception as e:
                print(f"[WARN] {element} ({method}) beta0_vs_beta: {e}")

    pd.DataFrame(skipped_rows).to_csv(out_skip_csv, index=False)

    print(f"[DONE] All outputs written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()