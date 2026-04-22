from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path("data_all_R0Bs")
OUTPUT_DIR = Path("out/13_cn_charge_stats")
OUTPUT_DIR.mkdir(exist_ok=True)

ELEMENTS = ["Cr", "V"]


def clean_charge_value(x):
    x = round(float(x), 3)
    if abs(x - round(x)) < 1e-8:
        return int(round(x))
    return x


def process_element(element: str):
    file_path = DATA_DIR / f"df_consist_{element}.csv"

    if not file_path.exists():
        print(f"[Skip] File not found: {file_path}")
        return None

    print(f"Processing {file_path} ...")

    df = pd.read_csv(file_path)

    cn_col = f"mean_coordn_{element}-O"
    charge_col = f"mean_{element}_charge"

    if cn_col not in df.columns or charge_col not in df.columns:
        print(f"[Skip] Required columns not found in {file_path.name}")
        print(f"Need columns: {cn_col}, {charge_col}")
        print("Actual columns:")
        print(df.columns.tolist())
        return None

    work = df[[cn_col, charge_col]].copy()
    work = work.dropna()

    work[cn_col] = pd.to_numeric(work[cn_col], errors="coerce")
    work[charge_col] = pd.to_numeric(work[charge_col], errors="coerce")
    work = work.dropna()

    work["CN"] = work[cn_col].round().astype(int)
    work["charge"] = work[charge_col].apply(clean_charge_value)

    summary = (
        work.groupby(["CN", "charge"])
        .size()
        .reset_index(name="count")
        .sort_values(["CN", "charge"])
    )

    summary["cn_total"] = summary.groupby("CN")["count"].transform("sum")
    summary["fraction"] = summary["count"] / summary["cn_total"]
    summary["element"] = element

    summary_file = OUTPUT_DIR / f"{element}_cn_mean_charge_summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"Saved: {summary_file}")

    pivot_count = summary.pivot(index="CN", columns="charge", values="count").fillna(0)
    pivot_frac = summary.pivot(index="CN", columns="charge", values="fraction").fillna(0)

    pivot_count_file = OUTPUT_DIR / f"{element}_cn_charge_count_pivot.csv"
    pivot_frac_file = OUTPUT_DIR / f"{element}_cn_charge_fraction_pivot.csv"

    pivot_count.to_csv(pivot_count_file)
    pivot_frac.to_csv(pivot_frac_file)

    print(f"Saved: {pivot_count_file}")
    print(f"Saved: {pivot_frac_file}")

    make_plots(element, pivot_frac)

    return summary


def make_plots(element: str, pivot_frac: pd.DataFrame):
    """
    Draw:
    1. grouped bar chart
    2. stacked bar chart
    3. heatmap
    On the stacked bar chart, annotate only the dominant charge for each CN.
    """
    if pivot_frac.empty:
        print(f"[Skip] No plotting data for {element}")
        return

    pivot_frac = pivot_frac.sort_index()

    # ---------- Plot 1: grouped bar chart ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_frac.plot(kind="bar", ax=ax)

    ax.set_title(f"{element}: charge fraction under each CN")
    ax.set_xlabel("CN")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Charge", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    grouped_png = OUTPUT_DIR / f"{element}_cn_charge_grouped_bar.png"
    plt.savefig(grouped_png, dpi=300)
    plt.close()
    print(f"Saved: {grouped_png}")

    # ---------- Plot 2: stacked bar chart + dominant charge annotation ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_frac.plot(kind="bar", stacked=True, ax=ax)

    ax.set_title(f"{element}: stacked charge fraction under each CN")
    ax.set_xlabel("CN")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.08)
    ax.legend(title="Charge", bbox_to_anchor=(1.02, 1), loc="upper left")

    # 对每个 CN 标出占比最大的 charge
    for i, cn in enumerate(pivot_frac.index):
        row = pivot_frac.loc[cn]

        dominant_charge = row.idxmax()
        dominant_fraction = row.max()

        # 标签内容：charge + fraction
        label = f"{dominant_charge}\n({dominant_fraction:.2f})"

        # 放在柱子顶部稍上方
        ax.text(
            i,                       # x
            dominant_fraction + 0.03,  # y
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    plt.tight_layout()
    stacked_png = OUTPUT_DIR / f"{element}_cn_charge_stacked_bar.png"
    plt.savefig(stacked_png, dpi=300)
    plt.close()
    print(f"Saved: {stacked_png}")

    # ---------- Plot 3: heatmap ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot_frac.values, aspect="auto")

    ax.set_xticks(range(len(pivot_frac.columns)))
    ax.set_xticklabels(pivot_frac.columns)
    ax.set_yticks(range(len(pivot_frac.index)))
    ax.set_yticklabels(pivot_frac.index)

    ax.set_xlabel("Charge")
    ax.set_ylabel("CN")
    ax.set_title(f"{element}: fraction heatmap")

    for i in range(len(pivot_frac.index)):
        for j in range(len(pivot_frac.columns)):
            val = pivot_frac.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Fraction")

    plt.tight_layout()
    heatmap_png = OUTPUT_DIR / f"{element}_cn_charge_heatmap.png"
    plt.savefig(heatmap_png, dpi=300)
    plt.close()
    print(f"Saved: {heatmap_png}")


def main():
    all_results = []

    for element in ELEMENTS:
        result = process_element(element)
        if result is not None and not result.empty:
            all_results.append(result)

    if all_results:
        merged = pd.concat(all_results, ignore_index=True)
        merged_file = OUTPUT_DIR / "all_elements_cn_mean_charge_summary.csv"
        merged.to_csv(merged_file, index=False)
        print(f"Saved: {merged_file}")

        print("\nPreview:")
        print(merged.head(30))
    else:
        print("No valid results generated.")


if __name__ == "__main__":
    main()