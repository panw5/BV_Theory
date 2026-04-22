import re
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===================== Settings =====================
DATA_DIR = Path("data_all_R0Bs")
OUT_DIR = Path("out/8_paper_style_compare")
OUT_FIG = OUT_DIR / "paper_style_6elements_compare.png"

# Six elements shown in the paper figure and their order
ELEMENT_ORDER = ["Ti", "V", "Cr", "Mn", "Fe", "Bi"]

# Coordination numbers and charges shown in the paper figure
# If inconsistencies are found later, only update this block
PLOT_CONFIG = {
    "Ti": {
        "coordination": [4, 5, 6],
        "charge": [3, 4],
    },
    "V": {
        "coordination": [4, 5, 6],
        "charge": [2, 3, 4, 5],
    },
    "Cr": {
        "coordination": [4, 6],
        "charge": [2, 3, 4, 5, 6],
    },
    "Mn": {
        "coordination": [4, 5, 6],
        "charge": [2, 3, 4],
    },
    "Fe": {
        "coordination": [4, 5, 6],
        "charge": [2, 3, 4],
    },
    "Bi": {
        "coordination": [5, 6, 7, 8],
        "charge": [3, 5],
    },
}

# Scatter style
POINT_SIZE = 10
POINT_ALPHA = 0.75

# Colormaps
COORD_CMAP = plt.cm.Set2
CHARGE_CMAP = plt.cm.Dark2

# Figure settings
FIGSIZE = (16, 12)
DPI = 300


# ===================== Helpers =====================
def ensure_output_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


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
    # Prefer ElementO_coordn
    pattern = re.compile(rf"^{element}O_coordn$")
    for c in df.columns:
        if pattern.match(str(c)):
            return c

    # Fallback: any column ending with *O_coordn
    candidates = [c for c in df.columns if str(c).endswith("O_coordn")]
    if candidates:
        return candidates[0]

    raise ValueError(f"coordination column not found for {element}.")


def detect_charge_column(df: pd.DataFrame, element: str):
    # Prefer mean_Element_charge
    exact = f"mean_{element}_charge"
    if exact in df.columns:
        return exact

    # Fallback: any column matching mean_*_charge
    candidates = [c for c in df.columns if re.fullmatch(r"mean_.*_charge", str(c))]
    if candidates:
        return candidates[0]

    raise ValueError(f"charge column not found for {element}.")


def load_element_data(data_dir: Path, element: str):
    """
    Read one element file and return a tidy DataFrame with columns:
      element, R0, B, CN, charge
    """
    file_path = data_dir / f"df_consist_{element}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    r0_col = detect_r0_column(df)
    b_col = detect_b_column(df)
    coord_col = detect_coord_column(df, element)
    charge_col = detect_charge_column(df, element)

    d = df[[r0_col, b_col, coord_col, charge_col]].copy()
    d = d.rename(columns={
        r0_col: "R0",
        b_col: "B",
        coord_col: "CN_raw",
        charge_col: "charge"
    })

    d["R0"] = pd.to_numeric(d["R0"], errors="coerce")
    d["B"] = pd.to_numeric(d["B"], errors="coerce")
    d["charge"] = pd.to_numeric(d["charge"], errors="coerce")
    d = d.dropna(subset=["R0", "B"])

    # Expand coordination numbers
    d["CN_list"] = d["CN_raw"].apply(parse_list_cell)
    d = d.explode("CN_list").rename(columns={"CN_list": "CN"})
    d["CN"] = pd.to_numeric(d["CN"], errors="coerce")
    d = d.dropna(subset=["CN"])

    d["element"] = element
    return d[["element", "R0", "B", "CN", "charge"]].copy()


def make_color_map(values, cmap):
    values = list(values)
    values_sorted = sorted(values)
    n = max(len(values_sorted), 1)
    color_map = {}
    for i, v in enumerate(values_sorted):
        if n == 1:
            color_map[v] = cmap(0.5)
        else:
            color_map[v] = cmap(i / (n - 1))
    return color_map


def format_val(v):
    if abs(v - int(v)) < 1e-9:
        return f"{float(v):.1f}"
    return f"{v:g}"


def plot_one_panel_by_coord(ax, d: pd.DataFrame, element: str, coord_list):
    sub = d[d["CN"].isin(coord_list)].copy()
    coord_present = sorted(sub["CN"].dropna().unique())

    colors = make_color_map(coord_present, COORD_CMAP)

    for cn in coord_present:
        dd = sub[sub["CN"] == cn]
        ax.scatter(
            dd["R0"], dd["B"],
            s=POINT_SIZE,
            color=colors[cn],
            alpha=POINT_ALPHA,
            label=format_val(cn)
        )

    ax.set_xlabel(r"R$_0$ (Å)", fontsize=14)
    ax.set_ylabel(r"B (Å)", fontsize=14)
    ax.legend(title="Coordination", frameon=False, fontsize=11, title_fontsize=12, loc="upper right")
    ax.text(0.05, 0.06, element, transform=ax.transAxes, fontsize=20, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.tick_params(labelsize=11)


def plot_one_panel_by_charge(ax, d: pd.DataFrame, element: str, charge_list):
    sub = d[d["charge"].isin(charge_list)].copy()
    charge_present = sorted(sub["charge"].dropna().unique())

    colors = make_color_map(charge_present, CHARGE_CMAP)

    for ch in charge_present:
        dd = sub[sub["charge"] == ch]
        ax.scatter(
            dd["R0"], dd["B"],
            s=POINT_SIZE,
            color=colors[ch],
            alpha=POINT_ALPHA,
            label=format_val(ch)
        )

    ax.set_xlabel(r"R$_0$ (Å)", fontsize=14)
    ax.set_ylabel(r"B (Å)", fontsize=14)
    ax.legend(title="Charge", frameon=False, fontsize=11, title_fontsize=12, loc="upper right")
    ax.text(0.05, 0.06, element, transform=ax.transAxes, fontsize=20, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.tick_params(labelsize=11)


def main():
    ensure_output_dir()

    # Load all six elements
    all_data = {}
    for element in ELEMENT_ORDER:
        try:
            all_data[element] = load_element_data(DATA_DIR, element)
            print(f"[OK] loaded {element}")
        except Exception as e:
            print(f"[FAIL] {element}: {e}")
            raise

    # 3 rows × 4 columns: two panels per element
    fig, axes = plt.subplots(3, 4, figsize=FIGSIZE, dpi=DPI)
    axes = np.array(axes)

    # Layout mapping
    layout = [
        ("Ti", (0, 0), (0, 1)),
        ("V",  (0, 2), (0, 3)),
        ("Cr", (1, 0), (1, 1)),
        ("Mn", (1, 2), (1, 3)),
        ("Fe", (2, 0), (2, 1)),
        ("Bi", (2, 2), (2, 3)),
    ]

    for element, pos_coord, pos_charge in layout:
        d = all_data[element]
        cfg = PLOT_CONFIG[element]

        ax_coord = axes[pos_coord]
        ax_charge = axes[pos_charge]

        plot_one_panel_by_coord(ax_coord, d, element, cfg["coordination"])
        plot_one_panel_by_charge(ax_charge, d, element, cfg["charge"])

    plt.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVE] {OUT_FIG.resolve()}")


if __name__ == "__main__":
    main()