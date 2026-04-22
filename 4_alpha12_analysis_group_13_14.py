import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================
# Settings
# ============================================================
INPUT_CSV = Path("out/3_fit_R0B_group_13_14/tables/beta_beta0_all_methods.csv")
OUT_DIR = Path("out/4_alpha12_analysis_group_13_14/c1_c2_analysis_merged")

METHOD_TO_USE = "Huber"
MIN_ROWS_PER_ELEMENT = 3

GROUP13 = {"Al", "Ga", "In", "Tl"}
GROUP14 = {"Ge", "Sn", "Pb"}

Z_MAP = {
    "Al": 3, "Ga": 3, "In": 3, "Tl": 3,
    "Ge": 4, "Sn": 4, "Pb": 4
}

# Shannon-like effective ionic radii (Å)
# First-pass values for common oxidation states used in this analysis
# Nearest-CN fallback is used when exact CN is unavailable
IONIC_RADIUS_MAP = {
    "Al": {4: 0.39, 5: 0.48, 6: 0.535},
    "Ga": {4: 0.47, 5: 0.55, 6: 0.62},
    "In": {4: 0.62, 6: 0.80, 8: 0.92},
    "Tl": {6: 0.885, 8: 0.98},

    "Ge": {4: 0.53, 6: 0.67},
    "Sn": {4: 0.69, 5: 0.76, 6: 0.83, 7: 0.89, 8: 0.95},
    "Pb": {6: 0.775, 8: 0.94, 9: 0.98, 10: 1.03, 12: 1.19},
}

ELEMENT_ORDER = ["Al", "Ga", "In", "Tl", "Ge", "Sn", "Pb"]


# ============================================================
# Utilities
# ============================================================
def ensure_dirs():
    for subdir in [
        "plots_pred_vs_true",
        "plots_residuals",
        "plots_radius",
        "plots_c1_c2",
        "plots_models",
        "plots_ratio",
        "plots_universality",
        "tables",
    ]:
        (OUT_DIR / subdir).mkdir(parents=True, exist_ok=True)


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


def load_base_data():
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
    df["group"] = np.where(df["element"].isin(GROUP13), "Group13",
                   np.where(df["element"].isin(GROUP14), "Group14", "Other"))

    df["beta"] = pd.to_numeric(df["beta"], errors="coerce")
    df["beta0"] = pd.to_numeric(df["beta0"], errors="coerce")
    df["CN"] = pd.to_numeric(df["CN"], errors="coerce")
    df["Z"] = pd.to_numeric(df["Z"], errors="coerce")

    df = df.dropna(subset=["element", "beta", "beta0", "CN", "Z"]).copy()

    df["Z_over_CN"] = df["Z"] / df["CN"]
    df["Z_times_CN"] = df["Z"] * df["CN"]
    df["beta_x_Z"] = df["beta"] * df["Z"]
    df["beta_x_CN"] = df["beta"] * df["CN"]

    return df


def load_radius_data():
    df = load_base_data().copy()

    radius_rows = df.apply(
        lambda r: get_radius_with_nearest_cn(r["element"], r["CN"]),
        axis=1
    )
    df["ionic_radius"] = [x[0] for x in radius_rows]
    df["radius_cn_used"] = [x[1] for x in radius_rows]
    df["radius_match_type"] = [x[2] for x in radius_rows]

    df = df.dropna(subset=["ionic_radius"]).copy()
    df = df[df["beta"] != 0].copy()

    df["beta0_over_beta"] = df["beta0"] / df["beta"]
    df["beta_x_radius"] = df["beta"] * df["ionic_radius"]
    return df


def fit_linear_model(df, feature_cols, target_col="beta0"):
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()

    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)
    resid = y - pred

    result = {
        "target": target_col,
        "features": " + ".join(feature_cols),
        "n": len(df),
        "R2": float(r2_score(y, pred)),
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "intercept": float(model.intercept_),
    }

    rss = float(np.sum((y - pred) ** 2))
    k = len(feature_cols) + 1
    n = len(y)
    if n > 0 and rss > 0:
        result["RSS"] = rss
        result["AIC"] = float(n * np.log(rss / n) + 2 * k)
        result["BIC"] = float(n * np.log(rss / n) + k * np.log(n))
    else:
        result["RSS"] = np.nan
        result["AIC"] = np.nan
        result["BIC"] = np.nan

    for name, coef in zip(feature_cols, model.coef_):
        result[f"coef_{name}"] = float(coef)

    return result, model, pred, resid


def save_model_summary(results, filename):
    out = pd.DataFrame(results).sort_values(["R2", "RMSE"], ascending=[False, True])
    out_path = OUT_DIR / "tables" / filename
    out.to_csv(out_path, index=False)
    print(f"[SAVE] {out_path}")
    return out


def save_table(df, filename):
    out_path = OUT_DIR / "tables" / filename
    df.to_csv(out_path, index=False)
    print(f"[SAVE] {out_path}")
    return out_path


def plot_pred_vs_true(df_or_ytrue, pred, title, out_path, xlabel="True beta0", ylabel="Predicted beta0"):
    if isinstance(df_or_ytrue, pd.DataFrame):
        y_true = df_or_ytrue["beta0"].to_numpy()
    else:
        y_true = np.asarray(df_or_ytrue)

    plt.figure(figsize=(5.5, 5.2), dpi=180)
    plt.scatter(y_true, pred, s=60, edgecolors="black", alpha=0.85)

    mn = min(y_true.min(), pred.min())
    mx = max(y_true.max(), pred.max())
    pad = 0.05 * (mx - mn) if mx > mn else 0.1

    plt.plot([mn - pad, mx + pad], [mn - pad, mx + pad], "k--", linewidth=1.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_path}")


def plot_residual_vs_variable(df, resid, xcol, title, out_path):
    plt.figure(figsize=(5.5, 4.2), dpi=180)
    plt.scatter(df[xcol], resid, s=60, edgecolors="black", alpha=0.85)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.2)
    plt.xlabel(xcol)
    plt.ylabel("Residual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_path}")


def plot_residual_by_element(df, resid, title, out_path):
    tmp = df.copy()
    tmp["resid"] = resid

    elements = sorted(tmp["element"].unique())
    data = [tmp.loc[tmp["element"] == el, "resid"].values for el in elements]

    plt.figure(figsize=(max(7, 0.8 * len(elements)), 4.8), dpi=180)
    plt.boxplot(data, tick_labels=elements)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.2)
    plt.ylabel("Residual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_path}")


def fit_beta0_vs_beta_for_subset(df_subset):
    if len(df_subset) < 2:
        return None

    x = df_subset[["beta"]].to_numpy()
    y = df_subset["beta0"].to_numpy()

    model = LinearRegression()
    model.fit(x, y)
    pred = model.predict(x)

    return {
        "c1": float(model.coef_[0]),
        "c2": float(model.intercept_),
        "R2": float(r2_score(y, pred)),
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "n": int(len(df_subset))
    }


def run_global_group_element_fits(df, filename):
    rows = []

    out = fit_beta0_vs_beta_for_subset(df)
    if out is not None:
        rows.append({"level": "global", "name": "all", **out})

    for gname in ["Group13", "Group14"]:
        sub = df[df["group"] == gname].copy()
        if len(sub) >= 2:
            out = fit_beta0_vs_beta_for_subset(sub)
            if out is not None:
                rows.append({"level": "group", "name": gname, **out})

    for el, sub in df.groupby("element"):
        if len(sub) < MIN_ROWS_PER_ELEMENT:
            continue
        out = fit_beta0_vs_beta_for_subset(sub)
        if out is not None:
            rows.append({"level": "element", "name": el, **out})

    out_df = pd.DataFrame(rows)
    out_path = OUT_DIR / "tables" / filename
    out_df.to_csv(out_path, index=False)
    print(f"[SAVE] {out_path}")
    return out_df


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


def plot_all_c1_c2_by_element(df, out_dir):
    if df.empty:
        return

    present_order = [el for el in ELEMENT_ORDER if el in df["name"].values]
    df = df.set_index("name").loc[present_order].reset_index()

    elements = df["name"].tolist()
    x = np.arange(len(elements))

    plt.figure(figsize=(9.0, 5.2), dpi=180)
    plt.plot(x, df["c1"], marker="o", linewidth=1.8, label="c1")
    plt.plot(x, df["c2"], marker="s", linewidth=1.8, label="c2")
    plt.xticks(x, elements)
    plt.ylabel("Value")
    plt.title("Element-wise c1 and c2")
    plt.legend(frameon=False)
    plt.tight_layout()
    out1 = out_dir / "all_c1_c2_by_element_line.png"
    plt.savefig(out1, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out1}")

    width = 0.38
    plt.figure(figsize=(9.4, 5.4), dpi=180)
    plt.bar(x - width / 2, df["c1"], width=width, label="c1")
    plt.bar(x + width / 2, df["c2"], width=width, label="c2")
    plt.xticks(x, elements)
    plt.ylabel("Value")
    plt.title("Element-wise c1 and c2")
    plt.legend(frameon=False)
    plt.tight_layout()
    out2 = out_dir / "all_c1_c2_by_element_bar.png"
    plt.savefig(out2, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out2}")


def plot_ratio_vs_radius(df, model, out_path):
    plt.figure(figsize=(6, 5), dpi=200)
    plt.scatter(df["ionic_radius"], df["beta0_over_beta"], s=70, edgecolors="black", alpha=0.85)
    for _, r in df.iterrows():
        plt.text(r["ionic_radius"], r["beta0_over_beta"], r["element"], fontsize=9)
    x_line = np.linspace(df["ionic_radius"].min(), df["ionic_radius"].max(), 200)
    y_line = model.predict(x_line.reshape(-1, 1))
    r2_ratio = model.score(df[["ionic_radius"]].to_numpy(), df["beta0_over_beta"].to_numpy())
    plt.plot(x_line, y_line, "r--", linewidth=2)
    plt.xlabel("ionic radius (Å)")
    plt.ylabel("beta0 / beta")
    plt.title(f"beta0 / beta vs ionic radius (R² = {r2_ratio:.4f})")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_path}")


def fit_ratio_vs_radius_for_subset(df_subset):
    if len(df_subset) < 2:
        return None, None
    X = df_subset[["ionic_radius"]].to_numpy()
    y = df_subset["beta0_over_beta"].to_numpy()
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)
    result = {
        "intercept_A": float(model.intercept_),
        "slope_B": float(model.coef_[0]),
        "n": len(df_subset),
        "R2": float(r2_score(y, pred)),
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred)))
    }
    return result, model


def ordered_element_rows(df):
    rows = []
    for el in ELEMENT_ORDER:
        sub = df[df["element"] == el].copy()
        if len(sub) < MIN_ROWS_PER_ELEMENT:
            continue
        fit_out, _ = fit_ratio_vs_radius_for_subset(sub)
        if fit_out is not None:
            rows.append({"level": "element", "name": el, **fit_out})
    return pd.DataFrame(rows)


def plot_groupwise_ratio_fits(df, out_path):
    plt.figure(figsize=(6.3, 5.1), dpi=200)
    for gname, sub in df.groupby("group"):
        if len(sub) < 2:
            continue
        x = sub["ionic_radius"].to_numpy().reshape(-1, 1)
        y = sub["beta0_over_beta"].to_numpy()
        plt.scatter(sub["ionic_radius"], sub["beta0_over_beta"], s=70, alpha=0.85, edgecolors="black", label=gname)
        model = LinearRegression()
        model.fit(x, y)
        x_line = np.linspace(sub["ionic_radius"].min(), sub["ionic_radius"].max(), 100)
        y_line = model.predict(x_line.reshape(-1, 1))
        plt.plot(x_line, y_line, linewidth=1.8)
    plt.xlabel("ionic radius (Å)")
    plt.ylabel("beta0 / beta")
    plt.title("Group-wise beta0 / beta vs ionic radius")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_path}")


def plot_elementwise_slopes(df_fit, out_path):
    if df_fit.empty:
        return
    order = [el for el in ELEMENT_ORDER if el in df_fit["name"].values]
    df_plot = df_fit.set_index("name").loc[order].reset_index()
    plt.figure(figsize=(9.0, 4.8), dpi=180)
    plt.bar(df_plot["name"], df_plot["slope_B"])
    plt.ylabel("Slope B")
    plt.title("Element-wise slope in beta0 / beta = A + B r")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_path}")


def plot_elementwise_intercepts(df_fit, out_path):
    if df_fit.empty:
        return
    order = [el for el in ELEMENT_ORDER if el in df_fit["name"].values]
    df_plot = df_fit.set_index("name").loc[order].reset_index()
    plt.figure(figsize=(9.0, 4.8), dpi=180)
    plt.bar(df_plot["name"], df_plot["intercept_A"])
    plt.ylabel("Intercept A")
    plt.title("Element-wise intercept in beta0 / beta = A + B r")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_path}")


# ============================================================
# Main
# ============================================================
def main():
    ensure_dirs()

    # --------------------------------------------------------
    # Section 1: no-radius analysis
    # --------------------------------------------------------
    df_no_radius = load_base_data()
    print(f"[INFO] No-radius dataset rows: {len(df_no_radius)}")

    model_specs_round1 = [
        ("M1_beta_only", ["beta"]),
        ("M2_beta_Z", ["beta", "Z"]),
        ("M3_beta_CN", ["beta", "CN"]),
        ("M4_beta_Z_CN", ["beta", "Z", "CN"]),
        ("M5_beta_Z_over_CN", ["beta", "Z_over_CN"]),
        ("M6_beta_Z_CN_ZoverCN", ["beta", "Z", "CN", "Z_over_CN"]),
    ]

    results_round1 = []
    pred_store_round1 = {}
    for model_name, feats in model_specs_round1:
        res, model, pred, resid = fit_linear_model(df_no_radius, feats)
        res["model_name"] = model_name
        results_round1.append(res)
        tmp = df_no_radius.copy()
        tmp["pred"] = pred
        tmp["resid"] = resid
        pred_store_round1[model_name] = tmp
        plot_pred_vs_true(df_no_radius, pred, f"{model_name}\nR2={res['R2']:.4f}, RMSE={res['RMSE']:.4f}", OUT_DIR / "plots_pred_vs_true" / f"{model_name}_pred_vs_true.png")
    summary_round1 = save_model_summary(results_round1, "model_comparison_round1.csv")

    interaction_name = "M7_interaction_beta_betaZ_betaCN_Z_CN"
    interaction_features = ["beta", "beta_x_Z", "beta_x_CN", "Z", "CN"]
    res_int, model_int, pred_int, resid_int = fit_linear_model(df_no_radius, interaction_features)
    res_int["model_name"] = interaction_name
    summary_interaction = pd.DataFrame([res_int])
    save_table(summary_interaction, "interaction_model_summary.csv")
    plot_pred_vs_true(df_no_radius, pred_int, f"{interaction_name}\nR2={res_int['R2']:.4f}, RMSE={res_int['RMSE']:.4f}", OUT_DIR / "plots_pred_vs_true" / f"{interaction_name}_pred_vs_true.png")

    base_tmp = pred_store_round1["M1_beta_only"]
    plot_residual_vs_variable(base_tmp, base_tmp["resid"].to_numpy(), "CN", "Baseline residual vs CN", OUT_DIR / "plots_residuals" / "baseline_residual_vs_CN.png")
    plot_residual_vs_variable(base_tmp, base_tmp["resid"].to_numpy(), "Z", "Baseline residual vs Z", OUT_DIR / "plots_residuals" / "baseline_residual_vs_Z.png")
    plot_residual_by_element(base_tmp, base_tmp["resid"].to_numpy(), "Baseline residual by element", OUT_DIR / "plots_residuals" / "baseline_residual_by_element.png")

    tmp_int = df_no_radius.copy()
    tmp_int["pred"] = pred_int
    tmp_int["resid"] = resid_int
    plot_residual_vs_variable(tmp_int, tmp_int["resid"].to_numpy(), "CN", "Interaction model residual vs CN", OUT_DIR / "plots_residuals" / "interaction_residual_vs_CN.png")
    plot_residual_vs_variable(tmp_int, tmp_int["resid"].to_numpy(), "Z", "Interaction model residual vs Z", OUT_DIR / "plots_residuals" / "interaction_residual_vs_Z.png")
    plot_residual_by_element(tmp_int, tmp_int["resid"].to_numpy(), "Interaction model residual by element", OUT_DIR / "plots_residuals" / "interaction_residual_by_element.png")

    fit_table_no_radius = run_global_group_element_fits(df_no_radius, "beta0_vs_beta_global_group_element_fits_no_radius.csv")

    enriched_no_radius = df_no_radius.copy()
    enriched_no_radius["pred_baseline"] = base_tmp["pred"].values
    enriched_no_radius["resid_baseline"] = base_tmp["resid"].values
    enriched_no_radius["pred_interaction"] = pred_int
    enriched_no_radius["resid_interaction"] = resid_int
    save_table(enriched_no_radius, "analysis_dataset_with_predictions_no_radius.csv")

    # --------------------------------------------------------
    # Section 2: radius-extended analysis
    # --------------------------------------------------------
    df_radius = load_radius_data()
    print(f"[INFO] Radius dataset rows: {len(df_radius)}")

    save_table(
        df_radius[[
            "element", "CN", "Z", "beta", "beta0",
            "ionic_radius", "radius_cn_used", "radius_match_type"
        ]].sort_values(["element", "CN"]),
        "radius_assignment_table.csv"
    )

    model_specs_radius = [
        ("M1_beta_only", ["beta"]),
        ("M2_beta_Z", ["beta", "Z"]),
        ("M3_beta_CN", ["beta", "CN"]),
        ("M4_beta_Z_CN", ["beta", "Z", "CN"]),
        ("M5_beta_Z_over_CN", ["beta", "Z_over_CN"]),
        ("M6_beta_Z_CN_ZoverCN", ["beta", "Z", "CN", "Z_over_CN"]),
        ("M8_beta_radius", ["beta", "ionic_radius"]),
        ("M9_beta_Z_CN_radius", ["beta", "Z", "CN", "ionic_radius"]),
        ("M10_beta_Z_CN_ZoverCN_radius", ["beta", "Z", "CN", "Z_over_CN", "ionic_radius"]),
    ]

    results_radius = []
    pred_store_radius = {}
    for model_name, feats in model_specs_radius:
        res, model, pred, resid = fit_linear_model(df_radius, feats)
        res["model_name"] = model_name
        results_radius.append(res)
        tmp = df_radius.copy()
        tmp["pred"] = pred
        tmp["resid"] = resid
        pred_store_radius[model_name] = tmp
        plot_pred_vs_true(df_radius, pred, f"{model_name}\nR2={res['R2']:.4f}, RMSE={res['RMSE']:.4f}", OUT_DIR / "plots_pred_vs_true" / f"{model_name}_pred_vs_true.png")
    summary_radius = save_model_summary(results_radius, "model_comparison_with_radius.csv")

    interaction_name_radius = "M11_interaction_beta_betaZ_betaCN_betaR_Z_CN_R"
    interaction_features_radius = ["beta", "beta_x_Z", "beta_x_CN", "beta_x_radius", "Z", "CN", "ionic_radius"]
    res_int_r, model_int_r, pred_int_r, resid_int_r = fit_linear_model(df_radius, interaction_features_radius)
    res_int_r["model_name"] = interaction_name_radius
    save_table(pd.DataFrame([res_int_r]), "interaction_model_with_radius_summary.csv")
    plot_pred_vs_true(df_radius, pred_int_r, f"{interaction_name_radius}\nR2={res_int_r['R2']:.4f}, RMSE={res_int_r['RMSE']:.4f}", OUT_DIR / "plots_pred_vs_true" / f"{interaction_name_radius}_pred_vs_true.png")

    base_tmp_r = pred_store_radius["M1_beta_only"]
    plot_residual_vs_variable(base_tmp_r, base_tmp_r["resid"].to_numpy(), "CN", "Baseline residual vs CN", OUT_DIR / "plots_residuals" / "baseline_residual_vs_CN_with_radius.png")
    plot_residual_vs_variable(base_tmp_r, base_tmp_r["resid"].to_numpy(), "Z", "Baseline residual vs Z", OUT_DIR / "plots_residuals" / "baseline_residual_vs_Z_with_radius.png")
    plot_residual_vs_variable(base_tmp_r, base_tmp_r["resid"].to_numpy(), "ionic_radius", "Baseline residual vs ionic radius", OUT_DIR / "plots_residuals" / "baseline_residual_vs_ionic_radius.png")
    plot_residual_by_element(base_tmp_r, base_tmp_r["resid"].to_numpy(), "Baseline residual by element", OUT_DIR / "plots_residuals" / "baseline_residual_by_element_with_radius.png")

    tmp_int_r = df_radius.copy()
    tmp_int_r["pred"] = pred_int_r
    tmp_int_r["resid"] = resid_int_r
    plot_residual_vs_variable(tmp_int_r, tmp_int_r["resid"].to_numpy(), "CN", "Interaction+radius residual vs CN", OUT_DIR / "plots_residuals" / "interaction_radius_residual_vs_CN.png")
    plot_residual_vs_variable(tmp_int_r, tmp_int_r["resid"].to_numpy(), "Z", "Interaction+radius residual vs Z", OUT_DIR / "plots_residuals" / "interaction_radius_residual_vs_Z.png")
    plot_residual_vs_variable(tmp_int_r, tmp_int_r["resid"].to_numpy(), "ionic_radius", "Interaction+radius residual vs ionic radius", OUT_DIR / "plots_residuals" / "interaction_radius_residual_vs_ionic_radius.png")
    plot_residual_by_element(tmp_int_r, tmp_int_r["resid"].to_numpy(), "Interaction+radius residual by element", OUT_DIR / "plots_residuals" / "interaction_radius_residual_by_element.png")

    fit_table_radius = run_global_group_element_fits(df_radius, "beta0_vs_beta_global_group_element_fits_with_radius.csv")

    element_rows = []
    for el, sub in df_radius.groupby("element"):
        if len(sub) < MIN_ROWS_PER_ELEMENT:
            continue
        fit_out = fit_beta0_vs_beta_for_subset(sub)
        if fit_out is None:
            continue
        mean_radius = float(sub["ionic_radius"].mean())
        element_rows.append({
            "name": el,
            "mean_ionic_radius": mean_radius,
            "c1": fit_out["c1"],
            "c2": fit_out["c2"],
            "R2": fit_out["R2"],
            "n": fit_out["n"]
        })
    element_radius_df = pd.DataFrame(element_rows).sort_values("name")
    save_table(element_radius_df, "element_c1_c2_vs_radius.csv")
    if not element_radius_df.empty:
        plot_with_labels(element_radius_df, "mean_ionic_radius", "c1", "c1 vs mean ionic radius", OUT_DIR / "plots_radius" / "c1_vs_mean_ionic_radius.png")
        plot_with_labels(element_radius_df, "mean_ionic_radius", "c2", "c2 vs mean ionic radius", OUT_DIR / "plots_radius" / "c2_vs_mean_ionic_radius.png")
        plot_all_c1_c2_by_element(element_radius_df, OUT_DIR / "plots_c1_c2")

    enriched_radius = df_radius.copy()
    enriched_radius["pred_baseline"] = base_tmp_r["pred"].values
    enriched_radius["resid_baseline"] = base_tmp_r["resid"].values
    enriched_radius["pred_interaction_radius"] = pred_int_r
    enriched_radius["resid_interaction_radius"] = resid_int_r
    save_table(enriched_radius, "analysis_dataset_with_radius_predictions.csv")

    # --------------------------------------------------------
    # Section 3: refined ratio and simplified model analysis
    # --------------------------------------------------------
    ratio_result, ratio_model, ratio_pred, ratio_resid = fit_linear_model(df_radius, ["ionic_radius"], target_col="beta0_over_beta")
    ratio_summary = pd.DataFrame([{"model_name": "ratio_radius_only", **ratio_result}])
    save_table(ratio_summary, "ratio_radius_model_summary.csv")

    df_ratio = df_radius.copy()
    df_ratio["ratio_pred"] = ratio_pred
    df_ratio["ratio_resid"] = ratio_resid
    save_table(df_ratio, "ratio_radius_analysis_dataset.csv")

    plot_ratio_vs_radius(df_ratio, ratio_model, OUT_DIR / "plots_ratio" / "beta0_over_beta_vs_radius.png")
    plot_pred_vs_true(df_ratio["beta0_over_beta"].to_numpy(), ratio_pred, f"beta0 / beta ~ ionic radius\nR2={ratio_result['R2']:.4f}, RMSE={ratio_result['RMSE']:.4f}", OUT_DIR / "plots_ratio" / "ratio_pred_vs_true.png", xlabel="True beta0 / beta", ylabel="Predicted beta0 / beta")

    plot_residual_vs_variable(df_ratio, ratio_resid, "Z", "Residual of beta0 / beta ~ r vs Z", OUT_DIR / "plots_residuals" / "ratio_residual_vs_Z.png")
    plot_residual_vs_variable(df_ratio, ratio_resid, "CN", "Residual of beta0 / beta ~ r vs CN", OUT_DIR / "plots_residuals" / "ratio_residual_vs_CN.png")
    plot_residual_vs_variable(df_ratio, ratio_resid, "Z_over_CN", "Residual of beta0 / beta ~ r vs Z/CN", OUT_DIR / "plots_residuals" / "ratio_residual_vs_Z_over_CN.png")

    residual_explainer_rows = []
    for name, feats in [
        ("residual_vs_Z", ["Z"]),
        ("residual_vs_CN", ["CN"]),
        ("residual_vs_Z_over_CN", ["Z_over_CN"]),
        ("residual_vs_Z_CN", ["Z", "CN"]),
        ("residual_vs_Z_CN_ZoverCN", ["Z", "CN", "Z_over_CN"]),
    ]:
        out, _, pred, _ = fit_linear_model(df_ratio, feats, target_col="ratio_resid")
        residual_explainer_rows.append({"model_name": name, **out})
        plot_pred_vs_true(df_ratio["ratio_resid"].to_numpy(), pred, f"{name}\nR2={out['R2']:.4f}, RMSE={out['RMSE']:.4f}", OUT_DIR / "plots_residuals" / f"{name}_pred_vs_true.png", xlabel="True residual", ylabel="Predicted residual")
    save_table(pd.DataFrame(residual_explainer_rows), "ratio_residual_explainer_models.csv")

    full_name = "M11_full_interaction_radius"
    full_features = ["beta", "beta_x_Z", "beta_x_CN", "beta_x_radius", "Z", "CN", "ionic_radius"]
    simple_name = "M12_simplified_radius_dominant"
    simple_features = ["beta", "beta_x_radius", "ionic_radius"]

    full_result, full_model, full_pred, full_resid = fit_linear_model(df_radius, full_features, target_col="beta0")
    simple_result, simple_model, simple_pred, simple_resid = fit_linear_model(df_radius, simple_features, target_col="beta0")
    comparison_df = pd.DataFrame([
        {"model_name": full_name, **full_result},
        {"model_name": simple_name, **simple_result},
    ])
    save_table(comparison_df, "full_vs_simplified_model_comparison.csv")

    plot_pred_vs_true(df_radius, full_pred, f"{full_name}\nR2={full_result['R2']:.4f}, RMSE={full_result['RMSE']:.4f}", OUT_DIR / "plots_models" / f"{full_name}_pred_vs_true.png")
    plot_pred_vs_true(df_radius, simple_pred, f"{simple_name}\nR2={simple_result['R2']:.4f}, RMSE={simple_result['RMSE']:.4f}", OUT_DIR / "plots_models" / f"{simple_name}_pred_vs_true.png")

    plot_residual_vs_variable(df_radius, full_resid, "ionic_radius", "Full-model residual vs ionic radius", OUT_DIR / "plots_models" / f"{full_name}_residual_vs_radius.png")
    plot_residual_vs_variable(df_radius, simple_resid, "ionic_radius", "Simplified-model residual vs ionic radius", OUT_DIR / "plots_models" / f"{simple_name}_residual_vs_radius.png")
    plot_residual_vs_variable(df_radius, full_resid, "Z", "Full-model residual vs Z", OUT_DIR / "plots_models" / f"{full_name}_residual_vs_Z.png")
    plot_residual_vs_variable(df_radius, simple_resid, "Z", "Simplified-model residual vs Z", OUT_DIR / "plots_models" / f"{simple_name}_residual_vs_Z.png")
    plot_residual_vs_variable(df_radius, full_resid, "CN", "Full-model residual vs CN", OUT_DIR / "plots_models" / f"{full_name}_residual_vs_CN.png")
    plot_residual_vs_variable(df_radius, simple_resid, "CN", "Simplified-model residual vs CN", OUT_DIR / "plots_models" / f"{simple_name}_residual_vs_CN.png")

    df_models = df_radius.copy()
    df_models["pred_full"] = full_pred
    df_models["resid_full"] = full_resid
    df_models["pred_simple"] = simple_pred
    df_models["resid_simple"] = simple_resid
    save_table(df_models, "full_vs_simplified_predictions.csv")

    full_coef = dict(zip(full_features, full_model.coef_))
    a = full_coef["beta"]
    b = full_coef["beta_x_Z"]
    c = full_coef["beta_x_CN"]
    d = full_coef["beta_x_radius"]
    e = full_coef["Z"]
    f = full_coef["CN"]
    g = full_coef["ionic_radius"]
    h = float(full_model.intercept_)

    simple_coef = dict(zip(simple_features, simple_model.coef_))
    a_s = simple_coef["beta"]
    d_s = simple_coef["beta_x_radius"]
    g_s = simple_coef["ionic_radius"]
    h_s = float(simple_model.intercept_)

    formula_rows = [
        {
            "model_name": full_name,
            "C1_formula": f"{a:.8f} + ({b:.8f})*Z + ({c:.8f})*CN + ({d:.8f})*r",
            "C2_formula": f"{h:.8f} + ({e:.8f})*Z + ({f:.8f})*CN + ({g:.8f})*r",
        },
        {
            "model_name": simple_name,
            "C1_formula": f"{a_s:.8f} + ({d_s:.8f})*r",
            "C2_formula": f"{h_s:.8f} + ({g_s:.8f})*r",
        }
    ]
    save_table(pd.DataFrame(formula_rows), "explicit_C1_C2_formulas.csv")

    universality_rows = []
    global_fit, _ = fit_ratio_vs_radius_for_subset(df_radius)
    if global_fit is not None:
        universality_rows.append({"level": "global", "name": "all", **global_fit})
    for gname in ["Group13", "Group14"]:
        sub = df_radius[df_radius["group"] == gname].copy()
        fit_out, _ = fit_ratio_vs_radius_for_subset(sub)
        if fit_out is not None:
            universality_rows.append({"level": "group", "name": gname, **fit_out})
    element_fit_df = ordered_element_rows(df_radius)
    if not element_fit_df.empty:
        universality_rows.extend(element_fit_df.to_dict("records"))
    universality_df = pd.DataFrame(universality_rows)
    save_table(universality_df, "ratio_vs_radius_universality_fits.csv")
    plot_groupwise_ratio_fits(df_radius, OUT_DIR / "plots_universality" / "groupwise_ratio_vs_radius.png")
    plot_elementwise_slopes(element_fit_df, OUT_DIR / "plots_universality" / "elementwise_ratio_slope_B.png")
    plot_elementwise_intercepts(element_fit_df, OUT_DIR / "plots_universality" / "elementwise_ratio_intercept_A.png")

    print("\n===== Round 1 model comparison =====")
    print(summary_round1[["model_name", "features", "n", "R2", "MAE", "RMSE"]].to_string(index=False))
    print("\n===== M7 interaction model =====")
    print(summary_interaction.to_string(index=False))
    print("\n===== Model comparison with radius =====")
    print(summary_radius[["model_name", "features", "n", "R2", "MAE", "RMSE"]].to_string(index=False))
    print("\n===== Full vs simplified =====")
    print(comparison_df[["model_name", "features", "n", "R2", "MAE", "RMSE", "AIC", "BIC"]].to_string(index=False))
    print(f"\n[DONE] All outputs written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()