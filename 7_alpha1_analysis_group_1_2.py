import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================
# Settings
# ============================================================
INPUT_CSV = Path("out/1_fit_R0B_group_1_2/tables/beta_beta0_all_methods.csv")
OUT_DIR = Path("out/7_alpha1_analysis_group12")

METHOD_TO_USE = "Huber"   # choose from: OLS, Huber, RANSAC
MIN_ROWS_PER_ELEMENT = 3

# Constrained model:
# beta0 = (beta + BETA_SHIFT) * C1 + C2_CONST
BETA_SHIFT = 0.1
C2_CONST = 0.5

GROUP1 = {"Li", "Na", "K", "Rb", "Cs", "Fr"}
GROUP2 = {"Be", "Mg", "Ca", "Sr", "Ba", "Ra"}

Z_MAP = {
    "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Fr": 1,
    "Be": 2, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Ra": 2
}

# ============================================================
# Shannon effective ionic radii (Å), keyed by (element, CN)
# ============================================================
IONIC_RADIUS_MAP = {
    "Li": {4: 0.59, 6: 0.76, 8: 0.92},
    "Na": {4: 0.99, 5: 1.00, 6: 1.02, 7: 1.12, 8: 1.18, 9: 1.24, 12: 1.39},
    "K":  {4: 1.37, 6: 1.38, 7: 1.46, 8: 1.51, 9: 1.55, 10: 1.59, 12: 1.64},
    "Rb": {6: 1.52, 7: 1.56, 8: 1.61, 9: 1.63, 10: 1.66, 11: 1.69, 12: 1.72},
    "Cs": {6: 1.67, 8: 1.74, 9: 1.78, 10: 1.81, 11: 1.85, 12: 1.88},

    "Be": {3: 0.16, 4: 0.27, 6: 0.45},
    "Mg": {4: 0.57, 5: 0.66, 6: 0.72, 8: 0.89},
    "Ca": {6: 1.00, 7: 1.06, 8: 1.12, 9: 1.18, 10: 1.23, 12: 1.34},
    "Sr": {6: 1.18, 7: 1.21, 8: 1.26, 9: 1.31, 10: 1.36, 12: 1.44},
    "Ba": {6: 1.35, 7: 1.38, 8: 1.42, 9: 1.47, 10: 1.52, 11: 1.57, 12: 1.61},
}


# ============================================================
# Utilities
# ============================================================
def ensure_dirs():
    (OUT_DIR / "plots_pred_vs_true").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "plots_residuals").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "plots_radius").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "plots_c1").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)


def get_radius_with_nearest_cn(element: str, cn_value: float):
    if element not in IONIC_RADIUS_MAP:
        return np.nan, None, "element_not_found"

    cn_dict = IONIC_RADIUS_MAP[element]
    cn_int = int(round(float(cn_value)))

    if cn_int in cn_dict:
        return cn_dict[cn_int], cn_int, "exact"

    available = sorted(cn_dict.keys())
    nearest = min(available, key=lambda x: abs(x - cn_int))
    return cn_dict[nearest], nearest, "nearest"


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

    df["Z"] = df["element"].map(Z_MAP)
    df["group"] = np.where(df["element"].isin(GROUP1), "Group1",
                   np.where(df["element"].isin(GROUP2), "Group2", "Other"))

    df["beta"] = pd.to_numeric(df["beta"], errors="coerce")
    df["beta0"] = pd.to_numeric(df["beta0"], errors="coerce")
    df["CN"] = pd.to_numeric(df["CN"], errors="coerce")
    df["Z"] = pd.to_numeric(df["Z"], errors="coerce")

    df = df.dropna(subset=["element", "beta", "beta0", "CN", "Z"]).copy()

    df["Z_over_CN"] = df["Z"] / df["CN"]
    df["Z_times_CN"] = df["Z"] * df["CN"]

    radius_rows = df.apply(
        lambda r: get_radius_with_nearest_cn(r["element"], r["CN"]),
        axis=1
    )
    df["ionic_radius"] = [x[0] for x in radius_rows]
    df["radius_cn_used"] = [x[1] for x in radius_rows]
    df["radius_match_type"] = [x[2] for x in radius_rows]

    df = df.dropna(subset=["ionic_radius"]).copy()
    return df


def build_c1_target(df, beta_shift=BETA_SHIFT, c2_const=C2_CONST):
    """
    From:
        beta0 = (beta + beta_shift) * C1 + c2_const
    derive:
        C1 = (beta0 - c2_const) / (beta + beta_shift)
    """
    df = df.copy()
    denom = df["beta"] + beta_shift
    df = df[np.abs(denom) > 1e-12].copy()
    df["c1_target"] = (df["beta0"] - c2_const) / denom
    return df


def fit_c1_model_and_reconstruct_beta0(
    df, feature_cols, beta_shift=BETA_SHIFT, c2_const=C2_CONST, target_col="c1_target"
):
    """
    Step 1: fit C1 = f(features)
    Step 2: reconstruct beta0_pred = (beta + beta_shift)*C1_pred + c2_const
    Step 3: evaluate against true beta0
    """
    X = df[feature_cols].to_numpy()
    y_c1 = df[target_col].to_numpy()

    model = LinearRegression()
    model.fit(X, y_c1)

    pred_c1 = model.predict(X)
    pred_beta0 = (df["beta"].to_numpy() + beta_shift) * pred_c1 + c2_const
    true_beta0 = df["beta0"].to_numpy()
    resid_beta0 = true_beta0 - pred_beta0
    resid_c1 = y_c1 - pred_c1

    result = {
        "features": " + ".join(feature_cols),
        "n": len(df),
        "R2_c1": float(r2_score(y_c1, pred_c1)),
        "MAE_c1": float(mean_absolute_error(y_c1, pred_c1)),
        "RMSE_c1": float(np.sqrt(mean_squared_error(y_c1, pred_c1))),
        "R2_beta0": float(r2_score(true_beta0, pred_beta0)),
        "MAE_beta0": float(mean_absolute_error(true_beta0, pred_beta0)),
        "RMSE_beta0": float(np.sqrt(mean_squared_error(true_beta0, pred_beta0))),
        "intercept_c1": float(model.intercept_),
    }

    for name, coef in zip(feature_cols, model.coef_):
        result[f"coef_{name}"] = float(coef)

    return result, model, pred_c1, pred_beta0, resid_c1, resid_beta0


def save_c1_model_summary(results, filename):
    out = pd.DataFrame(results).sort_values(
        ["R2_beta0", "RMSE_beta0"], ascending=[False, True]
    )
    out_path = OUT_DIR / "tables" / filename
    out.to_csv(out_path, index=False)
    print(f"[SAVE] {out_path}")
    return out


def plot_pred_vs_true(y_true, y_pred, title, xlabel, ylabel, out_path):
    plt.figure(figsize=(5.5, 5.2), dpi=180)
    plt.scatter(y_true, y_pred, s=60, edgecolors="black", alpha=0.85)

    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    pad = 0.05 * (mx - mn) if mx > mn else 0.1

    plt.plot([mn - pad, mx + pad], [mn - pad, mx + pad], "k--", linewidth=1.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_path}")


def plot_residual_vs_variable(df, resid, xcol, title, ylabel, out_path):
    plt.figure(figsize=(5.5, 4.2), dpi=180)
    plt.scatter(df[xcol], resid, s=60, edgecolors="black", alpha=0.85)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.2)
    plt.xlabel(xcol)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_path}")


def plot_residual_by_element(df, resid, title, ylabel, out_path):
    tmp = df.copy()
    tmp["resid"] = resid

    elements = sorted(tmp["element"].unique())
    data = [tmp.loc[tmp["element"] == el, "resid"].values for el in elements]

    plt.figure(figsize=(max(7, 0.8 * len(elements)), 4.8), dpi=180)
    plt.boxplot(data, tick_labels=elements)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.2)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_path}")


def plot_c1_vs_variable(df, xcol, title, out_path):
    plt.figure(figsize=(5.5, 4.4), dpi=180)
    plt.scatter(df[xcol], df["c1_target"], s=70, edgecolors="black", alpha=0.9)

    for _, r in df.iterrows():
        plt.text(r[xcol], r["c1_target"], str(r["element"]), fontsize=8)

    plt.xlabel(xcol)
    plt.ylabel("C1 = (beta0 - 0.5)/(beta + 0.1)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_path}")


def plot_with_labels(df, xcol, ycol, title, out_path):
    plt.figure(figsize=(5.5, 4.6), dpi=180)
    plt.scatter(df[xcol], df[ycol], s=70, edgecolors="black", alpha=0.9)
    for _, r in df.iterrows():
        plt.text(r[xcol], r[ycol], str(r["name"]), fontsize=9, ha="left", va="bottom")
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_path}")


def fit_c1_for_subset(df_subset):
    if len(df_subset) < 2:
        return None

    y = df_subset["c1_target"].to_numpy()
    mean_c1 = float(np.mean(y))

    return {
        "c1_mean": mean_c1,
        "c1_std": float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
        "n": int(len(df_subset))
    }


def summarize_c1_by_global_group_element(df):
    rows = []

    out = fit_c1_for_subset(df)
    if out is not None:
        rows.append({"level": "global", "name": "all", **out})

    for gname in ["Group1", "Group2"]:
        sub = df[df["group"] == gname].copy()
        if len(sub) >= 2:
            out = fit_c1_for_subset(sub)
            if out is not None:
                rows.append({"level": "group", "name": gname, **out})

    for el, sub in df.groupby("element"):
        if len(sub) < MIN_ROWS_PER_ELEMENT:
            continue
        out = fit_c1_for_subset(sub)
        if out is not None:
            rows.append({"level": "element", "name": el, **out})

    out_df = pd.DataFrame(rows)
    out_path = OUT_DIR / "tables" / "c1_global_group_element_summary.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[SAVE] {out_path}")
    return out_df


def plot_all_c1_by_element(df, out_dir):
    if df.empty:
        print("[WARN] plot_all_c1_by_element: input DataFrame is empty.")
        return

    desired_order = ["Li", "Na", "K", "Rb", "Cs", "Fr",
                     "Be", "Mg", "Ca", "Sr", "Ba", "Ra"]
    present_order = [el for el in desired_order if el in df["name"].values]
    df = df.set_index("name").loc[present_order].reset_index()

    elements = df["name"].tolist()
    x = np.arange(len(elements))

    plt.figure(figsize=(9.0, 5.2), dpi=180)
    plt.plot(x, df["mean_c1"], marker="o", linewidth=1.8, label="mean C1")
    plt.xticks(x, elements)
    plt.ylabel("Value")
    plt.title("Element-wise mean C1")
    plt.legend(frameon=False)
    plt.tight_layout()

    out1 = out_dir / "all_mean_c1_by_element_line.png"
    plt.savefig(out1, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out1}")

    plt.figure(figsize=(9.2, 5.2), dpi=180)
    plt.bar(x, df["mean_c1"], width=0.55, label="mean C1")
    plt.xticks(x, elements)
    plt.ylabel("Value")
    plt.title("Element-wise mean C1")
    plt.legend(frameon=False)
    plt.tight_layout()

    out2 = out_dir / "all_mean_c1_by_element_bar.png"
    plt.savefig(out2, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out2}")


# ============================================================
# Main
# ============================================================
def main():
    ensure_dirs()
    df = load_data()
    df = build_c1_target(df, beta_shift=BETA_SHIFT, c2_const=C2_CONST)

    print(f"[INFO] Method used: {METHOD_TO_USE}")
    print(f"[INFO] Total rows after C1 construction: {len(df)}")
    print(f"[INFO] Elements: {sorted(df['element'].unique())}")

    out_radius_assignment = OUT_DIR / "tables" / "radius_assignment_table.csv"
    df[[
        "element", "CN", "Z", "beta", "beta0",
        "ionic_radius", "radius_cn_used", "radius_match_type", "c1_target"
    ]].sort_values(["element", "CN"]).to_csv(out_radius_assignment, index=False)
    print(f"[SAVE] {out_radius_assignment}")

    out_c1_target = OUT_DIR / "tables" / "derived_c1_target_table.csv"
    df[[
        "element", "CN", "beta", "beta0", "ionic_radius", "Z", "c1_target"
    ]].sort_values(["element", "CN"]).to_csv(out_c1_target, index=False)
    print(f"[SAVE] {out_c1_target}")

    # --------------------------------------------------------
    # Constrained C1 models
    # beta0 = (beta + 0.1)*C1 + 0.5
    # --------------------------------------------------------
    c1_model_specs = [
        ("C1_M1_Z_only", ["Z"]),
        ("C1_M2_CN_only", ["CN"]),
        ("C1_M3_radius_only", ["ionic_radius"]),   # only r
        ("C1_M4_Z_CN", ["Z", "CN"]),
        ("C1_M5_Z_radius", ["Z", "ionic_radius"]),
        ("C1_M6_CN_radius", ["CN", "ionic_radius"]),
        ("C1_M7_Z_CN_radius", ["Z", "CN", "ionic_radius"]),
        ("C1_M8_Z_over_CN", ["Z_over_CN"]),
    ]

    c1_results = []
    c1_pred_store = {}

    for model_name, feats in c1_model_specs:
        res, model, pred_c1, pred_beta0, resid_c1, resid_beta0 = fit_c1_model_and_reconstruct_beta0(
            df, feats, beta_shift=BETA_SHIFT, c2_const=C2_CONST
        )
        res["model_name"] = model_name
        c1_results.append(res)

        tmp = df.copy()
        tmp["pred_c1"] = pred_c1
        tmp["pred_beta0"] = pred_beta0
        tmp["resid_c1"] = resid_c1
        tmp["resid_beta0"] = resid_beta0
        c1_pred_store[model_name] = tmp

        title_beta0 = f"{model_name}\nβ0: R2={res['R2_beta0']:.4f}, RMSE={res['RMSE_beta0']:.4f}"
        out1 = OUT_DIR / "plots_pred_vs_true" / f"{model_name}_beta0_pred_vs_true.png"
        plot_pred_vs_true(
            df["beta0"].to_numpy(),
            pred_beta0,
            title_beta0,
            xlabel="True beta0",
            ylabel="Predicted beta0",
            out_path=out1
        )

        title_c1 = f"{model_name}\nC1: R2={res['R2_c1']:.4f}, RMSE={res['RMSE_c1']:.4f}"
        out2 = OUT_DIR / "plots_pred_vs_true" / f"{model_name}_c1_pred_vs_true.png"
        plot_pred_vs_true(
            df["c1_target"].to_numpy(),
            pred_c1,
            title_c1,
            xlabel="True C1",
            ylabel="Predicted C1",
            out_path=out2
        )

    summary_c1 = save_c1_model_summary(c1_results, "c1_model_comparison.csv")

    # --------------------------------------------------------
    # Residual analysis: use radius-only as a simple baseline
    # --------------------------------------------------------
    base_tmp = c1_pred_store["C1_M3_radius_only"]

    plot_residual_vs_variable(
        base_tmp, base_tmp["resid_beta0"].to_numpy(),
        "CN",
        "Radius-only model residual(beta0) vs CN",
        "Residual of beta0",
        OUT_DIR / "plots_residuals" / "radius_only_residual_beta0_vs_CN.png"
    )
    plot_residual_vs_variable(
        base_tmp, base_tmp["resid_beta0"].to_numpy(),
        "Z",
        "Radius-only model residual(beta0) vs Z",
        "Residual of beta0",
        OUT_DIR / "plots_residuals" / "radius_only_residual_beta0_vs_Z.png"
    )
    plot_residual_vs_variable(
        base_tmp, base_tmp["resid_beta0"].to_numpy(),
        "ionic_radius",
        "Radius-only model residual(beta0) vs ionic radius",
        "Residual of beta0",
        OUT_DIR / "plots_residuals" / "radius_only_residual_beta0_vs_radius.png"
    )
    plot_residual_by_element(
        base_tmp, base_tmp["resid_beta0"].to_numpy(),
        "Radius-only model residual(beta0) by element",
        "Residual of beta0",
        OUT_DIR / "plots_residuals" / "radius_only_residual_beta0_by_element.png"
    )

    # --------------------------------------------------------
    # Direct plots for understanding C1
    # --------------------------------------------------------
    plot_c1_vs_variable(
        df, "ionic_radius",
        "Derived C1 vs ionic radius",
        OUT_DIR / "plots_radius" / "derived_c1_vs_radius.png"
    )
    plot_c1_vs_variable(
        df, "CN",
        "Derived C1 vs CN",
        OUT_DIR / "plots_radius" / "derived_c1_vs_CN.png"
    )
    plot_c1_vs_variable(
        df, "Z",
        "Derived C1 vs Z",
        OUT_DIR / "plots_radius" / "derived_c1_vs_Z.png"
    )
    plot_c1_vs_variable(
        df, "Z_over_CN",
        "Derived C1 vs Z/CN",
        OUT_DIR / "plots_radius" / "derived_c1_vs_Z_over_CN.png"
    )

    # --------------------------------------------------------
    # Element-wise mean C1 and relation to radius
    # --------------------------------------------------------
    c1_summary = summarize_c1_by_global_group_element(df)

    element_rows = []
    for el, sub in df.groupby("element"):
        if len(sub) < MIN_ROWS_PER_ELEMENT:
            continue
        mean_radius = float(sub["ionic_radius"].mean())
        mean_c1 = float(sub["c1_target"].mean())
        std_c1 = float(sub["c1_target"].std(ddof=1)) if len(sub) > 1 else 0.0
        element_rows.append({
            "name": el,
            "mean_ionic_radius": mean_radius,
            "mean_c1": mean_c1,
            "std_c1": std_c1,
            "n": len(sub)
        })

    element_c1_df = pd.DataFrame(element_rows).sort_values("name")
    out_element_c1 = OUT_DIR / "tables" / "element_mean_c1_vs_radius.csv"
    element_c1_df.to_csv(out_element_c1, index=False)
    print(f"[SAVE] {out_element_c1}")

    if not element_c1_df.empty:
        plot_with_labels(
            element_c1_df, "mean_ionic_radius", "mean_c1",
            "Mean C1 vs mean ionic radius",
            OUT_DIR / "plots_radius" / "mean_c1_vs_mean_ionic_radius.png"
        )
        plot_all_c1_by_element(
            element_c1_df,
            OUT_DIR / "plots_c1"
        )

    # --------------------------------------------------------
    # Additional simple fit: C1 vs ionic radius
    # --------------------------------------------------------
    print("\n===== Fit relation: C1 = A + B*r =====")
    X_r = df["ionic_radius"].values.reshape(-1, 1)
    y_c1 = df["c1_target"].values

    model_c1_r = LinearRegression()
    model_c1_r.fit(X_r, y_c1)

    A = model_c1_r.intercept_
    B = model_c1_r.coef_[0]
    R2 = model_c1_r.score(X_r, y_c1)

    print(f"A (intercept) = {A:.6f}")
    print(f"B (slope)     = {B:.6f}")
    print(f"R^2           = {R2:.6f}")

    print("\nDerived relation:")
    print(f"C1 = {A:.4f} + ({B:.4f}) * r")

    print("\nEquivalent beta0 formula:")
    print(f"beta0 = (beta + {BETA_SHIFT:.4f}) * ({A:.4f} + {B:.4f} * r) + {C2_CONST:.4f}")

    plt.figure(figsize=(6, 5), dpi=200)
    plt.scatter(
        df["ionic_radius"],
        df["c1_target"],
        s=70,
        edgecolors="black",
        alpha=0.85
    )

    for _, r in df.iterrows():
        plt.text(
            r["ionic_radius"],
            r["c1_target"],
            r["element"],
            fontsize=9
        )

    x_line = np.linspace(df["ionic_radius"].min(), df["ionic_radius"].max(), 200)
    y_line = model_c1_r.predict(x_line.reshape(-1, 1))
    plt.plot(x_line, y_line, "r--", linewidth=2)

    plt.xlabel("ionic radius (Å)")
    plt.ylabel("C1")
    plt.title(f"C1 vs ionic radius (R² = {R2:.4f})")
    plt.tight_layout()

    out_path_c1r = OUT_DIR / "plots_radius" / "c1_vs_radius_linear_fit.png"
    plt.savefig(out_path_c1r, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_path_c1r}")

    # --------------------------------------------------------
    # Save enriched dataset
    # --------------------------------------------------------
    enriched = df.copy()

    best_model_name = summary_c1.iloc[0]["model_name"]
    best_tmp = c1_pred_store[best_model_name]

    enriched["pred_c1_best"] = best_tmp["pred_c1"].values
    enriched["resid_c1_best"] = best_tmp["resid_c1"].values
    enriched["pred_beta0_best"] = best_tmp["pred_beta0"].values
    enriched["resid_beta0_best"] = best_tmp["resid_beta0"].values

    out_enriched = OUT_DIR / "tables" / "analysis_dataset_with_constrained_c1_predictions.csv"
    enriched.to_csv(out_enriched, index=False)
    print(f"[SAVE] {out_enriched}")

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------
    print("\n===== C1-model comparison under constrained form =====")
    cols_show = [
        "model_name", "features", "n",
        "R2_c1", "RMSE_c1",
        "R2_beta0", "RMSE_beta0"
    ]
    print(summary_c1[cols_show].to_string(index=False))

    print("\n===== Meaning of C1 in the constrained model =====")
    print("Model assumed:")
    print(f"beta0 = (beta + {BETA_SHIFT})*C1 + {C2_CONST}")
    print("Equivalent form:")
    print(f"beta0 = C1*beta + ({BETA_SHIFT}*C1 + {C2_CONST})")
    print("Therefore, C1 is still the slope with respect to beta,")
    print("while the intercept is constrained to be:")
    print(f"C2 = {BETA_SHIFT}*C1 + {C2_CONST}")
    print("For each data point, the implied C1 is:")
    print(f"C1 = (beta0 - {C2_CONST}) / (beta + {BETA_SHIFT})")

    print("\n===== Global / Group / Element summary of C1 =====")
    if not c1_summary.empty:
        print(c1_summary.to_string(index=False))
    else:
        print("No C1 summary generated.")

    print(f"\n[BEST MODEL] {best_model_name}")
    print(f"[DONE] All outputs written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()