import re
import ast
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor


# ===================== Settings =====================
DATA_DIR = Path("data_all_R0Bs")
OUT_DIR = Path("out/10_transition_metals_least_spread_all_methods")

TRANSITION_METALS = [
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Ac", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
]

# CN filtering
DROP_CN_ZERO = True
KEEP_CN_LIST = None          # e.g. [2,3,4,5,6,7,8]
MIN_POINTS_PER_CN = 5

# Output methods
METHODS = ["OLS", "Huber", "RANSAC"]

# Least-spread weights:
#   "equal" -> each CN line same weight
#   "n"     -> weighted by sample count of each CN
LSP_WEIGHT_MODE = "equal"

# Plot style
FIG_DPI = 260
POINTS_BG = False            # whether to show raw scatter in background
POINT_ALPHA = 0.18
POINT_SIZE = 7

LINE_WIDTH = 1.8
LSP_MARKER_SIZE = 70

# text annotation offset
TEXT_OFFSET_FRAC = (0.03, 0.06)

# Dynamic subplot layout
MAX_COLS = 8   # max number of subplots per row


# ===================== IO helpers =====================
def ensure_output_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_list_cell(x):
    """Parse a cell that may contain list-like content into list[float]."""
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


def detect_r0_column(df: pd.DataFrame):
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


def detect_b_column(df: pd.DataFrame):
    if "B" in df.columns:
        return "B"
    for c in df.columns:
        if str(c).strip() == "B":
            return c
    raise ValueError("B column not found.")


def detect_coord_column(df: pd.DataFrame, element: str):
    exact = f"{element}O_coordn"
    if exact in df.columns:
        return exact
    candidates = [c for c in df.columns if str(c).endswith("O_coordn")]
    if candidates:
        return candidates[0]
    raise ValueError(f"coordination column not found for {element}.")


def load_element_points(data_dir: Path, element: str):
    """
    Return exploded CN point table:
      columns = [element, R0, B, CN]
    """
    file_path = data_dir / f"df_consist_{element}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    r0_col = detect_r0_column(df)
    b_col = detect_b_column(df)
    coord_col = detect_coord_column(df, element)

    d = df[[r0_col, b_col, coord_col]].copy()
    d = d.rename(columns={
        r0_col: "R0",
        b_col: "B",
        coord_col: "CN_raw"
    })

    d["R0"] = pd.to_numeric(d["R0"], errors="coerce")
    d["B"] = pd.to_numeric(d["B"], errors="coerce")
    d = d.dropna(subset=["R0", "B"]).copy()

    d["CN_list"] = d["CN_raw"].apply(parse_list_cell)
    d = d.explode("CN_list").rename(columns={"CN_list": "CN"})
    d["CN"] = pd.to_numeric(d["CN"], errors="coerce")
    d = d.dropna(subset=["CN"]).copy()

    if DROP_CN_ZERO:
        d = d[d["CN"] != 0]

    if KEEP_CN_LIST is not None:
        keep = set(float(v) for v in KEEP_CN_LIST)
        d = d[d["CN"].isin(keep)]

    d["element"] = element
    return d[["element", "R0", "B", "CN"]].copy()


# ===================== Fit helpers =====================
def clean_xy(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def is_degenerate_x(x, eps=1e-12):
    x = np.asarray(x, dtype=float).ravel()
    if len(x) < 2:
        return True
    return float(np.nanstd(x)) < eps


def fit_line_ols(x, y):
    if len(x) < 2:
        return None, None
    beta, beta0 = np.polyfit(x, y, 1)
    if not np.isfinite(beta) or not np.isfinite(beta0):
        return None, None
    return float(beta), float(beta0)


def fit_line_huber(x, y):
    """
    Standardize x before Huber fitting to reduce numerical warnings.
    """
    if len(x) < 2:
        return None, None

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if len(x) < 2:
        return None, None

    x_mean = np.mean(x)
    x_std = np.std(x)

    if x_std < 1e-10:
        return None, None

    x_scaled = (x - x_mean) / x_std
    X = x_scaled.reshape(-1, 1)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)

            model = HuberRegressor(epsilon=1.35, max_iter=1000, alpha=0.0)
            model.fit(X, y)

        a = float(model.coef_[0])
        b = float(model.intercept_)

        beta = a / x_std
        beta0 = b - a * x_mean / x_std

        if not np.isfinite(beta) or not np.isfinite(beta0):
            return None, None

        return float(beta), float(beta0)

    except Exception:
        return None, None


def fit_line_ransac(x, y):
    if len(x) < 2:
        return None, None

    X = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)

    try:
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

        beta = float(model.estimator_.coef_[0])
        beta0 = float(model.estimator_.intercept_)

        if not np.isfinite(beta) or not np.isfinite(beta0):
            return None, None

        return beta, beta0

    except Exception:
        return None, None


def fit_by_method(method, x, y):
    x, y = clean_xy(x, y)

    if len(x) < 2 or is_degenerate_x(x):
        return None, None

    method_u = method.strip().upper()
    if method_u == "OLS":
        return fit_line_ols(x, y)
    elif method_u == "HUBER":
        return fit_line_huber(x, y)
    elif method_u == "RANSAC":
        return fit_line_ransac(x, y)
    else:
        raise ValueError(f"Unknown method: {method}")


def fit_cn_lines(points_df: pd.DataFrame, method: str):
    """
    Fit one line per CN:
      B = beta * R0 + beta0
    Return DataFrame with columns:
      element, CN, n_points, beta, beta0, method
    """
    rows = []
    element = points_df["element"].iloc[0]

    for cn in sorted(points_df["CN"].unique()):
        sub = points_df[points_df["CN"] == cn]
        x = sub["R0"].to_numpy()
        y = sub["B"].to_numpy()
        x, y = clean_xy(x, y)
        n = len(x)

        if n < MIN_POINTS_PER_CN:
            continue
        if is_degenerate_x(x):
            continue

        beta, beta0 = fit_by_method(method, x, y)
        if beta is None:
            continue

        rows.append({
            "element": element,
            "CN": float(cn),
            "n_points": int(n),
            "beta": float(beta),
            "beta0": float(beta0),
            "method": method
        })

    return pd.DataFrame(rows)


# ===================== Least spread point =====================
def compute_least_spread_point(lines_df: pd.DataFrame, weight_mode: str = "equal"):
    """
    Least spread point for lines:
        B_i(R0) = beta_i * R0 + beta0_i

    spread(x) = weighted variance of {beta_i*x + beta0_i}

    Analytic minimizer:
        x* = -Cov_w(beta, beta0) / Var_w(beta)
    """
    if lines_df is None or len(lines_df) < 2:
        return None

    m = lines_df["beta"].to_numpy(dtype=float)
    b = lines_df["beta0"].to_numpy(dtype=float)

    if weight_mode == "equal":
        w = np.ones(len(lines_df), dtype=float)
    elif weight_mode == "n":
        w = lines_df["n_points"].to_numpy(dtype=float)
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    w = w / np.sum(w)

    mu_m = np.sum(w * m)
    mu_b = np.sum(w * b)

    dm = m - mu_m
    db = b - mu_b

    var_m = np.sum(w * dm * dm)
    cov_mb = np.sum(w * dm * db)

    if abs(var_m) < 1e-14:
        return None

    x_star = -cov_mb / var_m
    y_vals = m * x_star + b
    y_star = np.sum(w * y_vals)

    spread_var = np.sum(w * (y_vals - y_star) ** 2)
    spread_std = np.sqrt(spread_var)

    return {
        "R0_star": float(x_star),
        "B_star": float(y_star),
        "spread_var_star": float(spread_var),
        "spread_std_star": float(spread_std),
        "n_lines": int(len(lines_df))
    }


# ===================== Plot helpers =====================
def build_style_maps(cn_values):
    styles = ["-", "--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
    cn_sorted = sorted(cn_values)
    return {cn: styles[i % len(styles)] for i, cn in enumerate(cn_sorted)}


def element_color(element):
    cmap = {
        "Sc": "#4B3F99", "Ti": "#5C86C5", "V": "#67C5F1", "Cr": "#58B8A5", "Mn": "#2C9348",
        "Fe": "#8E5EA2", "Co": "#C97B63", "Ni": "#D4A017", "Cu": "#B85C8A", "Zn": "#6B8E23",
        "Y": "#4C78A8", "Zr": "#F58518", "Nb": "#E45756", "Mo": "#72B7B2", "Tc": "#54A24B",
        "Ru": "#EECA3B", "Rh": "#B279A2", "Pd": "#FF9DA6", "Ag": "#9D755D", "Cd": "#BAB0AC",
        "La": "#1F77B4", "Hf": "#AEC7E8", "Ta": "#FF7F0E", "W": "#FFBB78", "Re": "#2CA02C",
        "Os": "#98DF8A", "Ir": "#D62728", "Pt": "#FF9896", "Au": "#9467BD", "Hg": "#C5B0D5",
        "Ac": "#8C564B", "Rf": "#C49C94", "Db": "#E377C2", "Sg": "#F7B6D2", "Bh": "#7F7F7F",
        "Hs": "#C7C7C7", "Mt": "#BCBD22", "Ds": "#DBDB8D", "Rg": "#17BECF", "Cn": "#9EDAE5",
    }
    return cmap.get(element, "tab:blue")


def format_cn(cn):
    if abs(cn - int(cn)) < 1e-9:
        return f"{int(cn)}"
    return f"{cn:g}"


def line_y(beta, beta0, x):
    return beta * x + beta0


def annotate_lsp(ax, x_star, y_star, xlim, ylim, color):
    dx = (xlim[1] - xlim[0]) * TEXT_OFFSET_FRAC[0]
    dy = (ylim[1] - ylim[0]) * TEXT_OFFSET_FRAC[1]
    ax.annotate(
        f"({x_star:.2f}, {y_star:.2f})",
        (x_star, y_star),
        xytext=(x_star + dx, y_star + dy),
        textcoords="data",
        fontsize=8,
        color=color
    )


def plot_transition_metals_dynamic(method, all_points_dict, all_lines_df, all_lsp_df, out_fig):
    """
    Only plot transition metals that were successfully loaded.
    Missing elements are skipped entirely, without empty subplots.
    """
    valid_elements = [el for el in TRANSITION_METALS if el in all_points_dict]

    if len(valid_elements) == 0:
        print(f"[WARN] No valid elements to plot for method {method}")
        return

    n = len(valid_elements)
    ncols = min(MAX_COLS, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.2 * ncols, 3.0 * nrows),
        dpi=FIG_DPI
    )

    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    elif nrows == 1 or ncols == 1:
        axes = np.array(axes).reshape(-1)
    else:
        axes = np.array(axes).flatten()

    for idx, element in enumerate(valid_elements):
        ax = axes[idx]

        points_df = all_points_dict[element]
        lines_df = all_lines_df[
            (all_lines_df["element"] == element) &
            (all_lines_df["method"] == method)
        ].copy()

        color = element_color(element)

        if lines_df.empty:
            ax.axis("off")
            continue

        x_data = pd.to_numeric(points_df["R0"], errors="coerce").to_numpy(dtype=float)
        y_data = pd.to_numeric(points_df["B"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x_data) & np.isfinite(y_data)
        x_data = x_data[mask]
        y_data = y_data[mask]

        if len(x_data) == 0 or len(y_data) == 0:
            ax.axis("off")
            continue

        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)

        x_pad = 0.10 * (x_max - x_min) if x_max > x_min else 0.5
        y_pad = 0.10 * (y_max - y_min) if y_max > y_min else 0.5

        xlim = (x_min - x_pad, x_max + x_pad)
        ylim = (y_min - y_pad, y_max + y_pad)
        x_plot = np.linspace(xlim[0], xlim[1], 200)

        if POINTS_BG:
            for cn in sorted(lines_df["CN"].unique()):
                dd = points_df[points_df["CN"] == cn]
                ax.scatter(dd["R0"], dd["B"], s=POINT_SIZE, color=color, alpha=POINT_ALPHA)

        linestyles = build_style_maps(lines_df["CN"].tolist())

        legend_handles = []
        legend_labels = []

        for _, row in lines_df.sort_values("CN").iterrows():
            cn = float(row["CN"])
            beta = float(row["beta"])
            beta0 = float(row["beta0"])
            n_points = int(row["n_points"])

            y_plot = line_y(beta, beta0, x_plot)
            h, = ax.plot(
                x_plot, y_plot,
                linestyle=linestyles[cn],
                color=color,
                linewidth=LINE_WIDTH,
                alpha=0.95
            )
            legend_handles.append(h)
            legend_labels.append(f"CN{format_cn(cn)} (n={n_points}, β={beta:.2f})")

        lsp_row = all_lsp_df[
            (all_lsp_df["element"] == element) &
            (all_lsp_df["method"] == method)
        ]

        if not lsp_row.empty:
            x_star = float(lsp_row["R0_star"].iloc[0])
            y_star = float(lsp_row["B_star"].iloc[0])

            h_lsp = ax.scatter(
                [x_star], [y_star],
                s=LSP_MARKER_SIZE,
                facecolors="white",
                edgecolors=color,
                linewidths=1.8,
                zorder=6
            )
            annotate_lsp(ax, x_star, y_star, xlim, ylim, color)
            legend_handles.append(h_lsp)
            legend_labels.append("least-spread point")

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"{element}-O", fontsize=12)
        ax.set_xlabel(r"R$_0$ (Å)", fontsize=10)
        ax.set_ylabel(r"B (Å)", fontsize=10)

        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.22)

        for spine in ax.spines.values():
            spine.set_linewidth(0.9)

        ax.legend(
            legend_handles, legend_labels,
            frameon=False, fontsize=6.5,
            loc="upper right"
        )

    # remove unused axes
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Transition metals: least-spread point with {method} fitting", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)


# ===================== Main =====================
def main():
    ensure_output_dir()

    all_points_dict = {}
    for element in TRANSITION_METALS:
        try:
            points_df = load_element_points(DATA_DIR, element)
            all_points_dict[element] = points_df
            print(f"[OK] loaded {element}")
        except Exception as e:
            print(f"[SKIP] {element}: {e}")

    if not all_points_dict:
        raise RuntimeError("No element data loaded.")

    all_line_frames = []
    all_lsp_rows = []

    for method in METHODS:
        print(f"\n===== METHOD: {method} =====")
        for element in TRANSITION_METALS:
            if element not in all_points_dict:
                continue

            try:
                points_df = all_points_dict[element]
                lines_df = fit_cn_lines(points_df, method=method)

                if lines_df.empty:
                    print(f"[WARN] {element} ({method}): no valid CN lines fitted.")
                    continue

                all_line_frames.append(lines_df)

                lsp = compute_least_spread_point(lines_df, weight_mode=LSP_WEIGHT_MODE)
                if lsp is None:
                    print(f"[WARN] {element} ({method}): slopes nearly parallel, no stable least-spread point.")
                else:
                    all_lsp_rows.append({
                        "element": element,
                        "method": method,
                        **lsp
                    })

                print(f"[OK] {element} ({method}): fitted CN lines = {len(lines_df)}")

            except Exception as e:
                print(f"[FAIL] {element} ({method}): {e}")

    if not all_line_frames:
        raise RuntimeError("No valid fitted lines were produced.")

    all_lines_df = pd.concat(all_line_frames, ignore_index=True)
    all_lsp_df = pd.DataFrame(all_lsp_rows)

    out_lines = OUT_DIR / "transition_metals_XO_fitted_lines_all_methods.csv"
    out_lsp = OUT_DIR / "transition_metals_XO_least_spread_points_all_methods.csv"

    all_lines_df.to_csv(out_lines, index=False)
    all_lsp_df.to_csv(out_lsp, index=False)

    # One figure for each method
    for method in METHODS:
        out_fig = OUT_DIR / f"transition_metals_XO_least_spread_{method}.png"
        plot_transition_metals_dynamic(method, all_points_dict, all_lines_df, all_lsp_df, out_fig)
        print(f"[SAVE] {out_fig.resolve()}")

    print(f"[SAVE] {out_lines.resolve()}")
    print(f"[SAVE] {out_lsp.resolve()}")


if __name__ == "__main__":
    main()