"""
Orchestration Script for BEANWISE Phase 4 Artifact Generation.
Creates tables, figures, JSON metadata, and HTML reports for Feature Engineering and Selection.
"""

import sys
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    FIGURES_DIR,
    TABLES_DIR,
    REPORTS_DIR,
    PROCESSED_DATASET_DIR,
    TARGET_COLUMN,
    FEATURE_NAMES,
)
from src.features.feature_engineering import (
    add_engineered_features,
    get_engineered_feature_names,
    ENGINEERED_FEATURE_METADATA,
)
from src.features.feature_selection import (
    split_dataset,
    evaluate_feature_selection_methods,
)


def run_phase4_artifacts():
    print("Running Phase 4 Feature Engineering and Selection Pipeline...")
    phase4_rep_dir = REPORTS_DIR / "phase4"
    phase4_rep_dir.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_path = PROCESSED_DATASET_DIR / "cleaned_dry_beans.csv"
    if not cleaned_path.is_file():
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_path}")

    df_cleaned = pd.read_csv(cleaned_path)
    print(f"Loaded cleaned dataset: {len(df_cleaned)} rows, {len(df_cleaned.columns)} columns.")

    # 1. Feature Engineering
    print("1/7 Engineering domain features...")
    df_engineered = add_engineered_features(df_cleaned)
    eng_feature_names = get_engineered_feature_names()
    all_feature_names = FEATURE_NAMES + eng_feature_names

    # Save feature engineering dataset
    fe_analysis_path = PROCESSED_DATASET_DIR / "feature_engineering_analysis.csv"
    df_engineered.to_csv(fe_analysis_path, index=False)
    print(f"Engineered dataset saved: {len(df_engineered)} rows, {len(df_engineered.columns)} columns (6 new features).")

    # 2. Stratified Data Splitting (Leakage Prevention)
    print("2/7 Performing stratified train (70%) / val (15%) / test (15%) split...")
    train_df, val_df, test_df, split_meta = split_dataset(df_engineered, random_state=42)

    # Save split metadata to processed and reports
    with open(PROCESSED_DATASET_DIR / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)
    with open(phase4_rep_dir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)

    # Export split summary table
    split_summary_df = pd.DataFrame([
        {"Partition": "Training Set", "Observations": split_meta["train_size"], "Percentage": f"{split_meta['train_pct']}%", "Role": "Model fitting & feature selection"},
        {"Partition": "Validation Set", "Observations": split_meta["val_size"], "Percentage": f"{split_meta['val_pct']}%", "Role": "Hyperparameter tuning & model comparison"},
        {"Partition": "Test Set (Held-Out)", "Observations": split_meta["test_size"], "Percentage": f"{split_meta['test_pct']}%", "Role": "Untouched final model certification"},
    ])
    split_summary_df.to_csv(TABLES_DIR / "phase4_split_summary.csv", index=False)

    # 3. Feature Selection on Training Set ONLY
    print("3/7 Evaluating feature selection methods on training set...")
    X_train = train_df[all_feature_names]
    y_train = train_df[TARGET_COLUMN]

    sel_results = evaluate_feature_selection_methods(train_df, y_train, all_feature_names, random_state=42)
    rankings_df = sel_results["consolidated_rankings_df"]
    rankings_df.to_csv(TABLES_DIR / "phase4_feature_rankings.csv", index=False)

    # Mutual information table
    mi_df = pd.DataFrame({
        "Feature": sel_results["mutual_info_series"].index,
        "Mutual Information Score": sel_results["mutual_info_series"].values.round(4),
    })
    mi_df.to_csv(TABLES_DIR / "phase4_mutual_information.csv", index=False)

    # ANOVA table
    sel_results["anova_df"].to_csv(TABLES_DIR / "phase4_anova_scores.csv", index=False)

    # Feature statistics table
    feat_stats_rows = []
    for f in all_feature_names:
        s = X_train[f]
        feat_stats_rows.append({
            "Feature": f,
            "Mean": round(float(s.mean()), 4),
            "Std Dev": round(float(s.std(ddof=1)), 4),
            "Min": round(float(s.min()), 4),
            "Median": round(float(s.median()), 4),
            "Max": round(float(s.max()), 4),
            "Skewness": round(float(s.skew()), 4),
            "Type": "Engineered" if f in eng_feature_names else "Original",
        })
    pd.DataFrame(feat_stats_rows).to_csv(TABLES_DIR / "phase4_feature_statistics.csv", index=False)

    # Feature comparison table (Original vs Engineered properties)
    feat_comp_df = pd.DataFrame([
        {"Feature Set": "Original Baseline", "Total Features": len(FEATURE_NAMES), "Description": "16 standard geometric and morphological features from UCI source"},
        {"Feature Set": "Engineered Set", "Total Features": len(all_feature_names), "Description": "Original 16 + 6 domain ratios (Area/Perimeter, Axis Ratios, Compactness/Roundness)"},
        {"Feature Set": "Selected Top-10", "Total Features": 10, "Description": "Top 10 features identified by consolidated RFE and multi-criteria ranking"},
    ])
    feat_comp_df.to_csv(TABLES_DIR / "phase4_feature_comparison.csv", index=False)

    # 4. Correlation Analysis on Training Features
    print("4/7 Computing correlation on training features...")
    corr_mat = X_train.corr(method="pearson").round(3)
    corr_mat.to_csv(TABLES_DIR / "phase4_correlation_analysis.csv")

    # 5. PCA on Scaled Training Features
    print("5/7 Computing PCA on training set...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[FEATURE_NAMES])
    pca = PCA(n_components=10, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)

    pca_var_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(10)],
        "Explained Variance Ratio": pca.explained_variance_ratio_.round(4),
        "Cumulative Explained Variance": np.cumsum(pca.explained_variance_ratio_).round(4),
    })
    pca_var_df.to_csv(TABLES_DIR / "phase4_pca_variance.csv", index=False)

    # 6. Figures Generation
    print("6/7 Generating Phase 4 visualization figures...")

    # Figure 1: Feature Correlation Heatmap (All 22 features)
    fig1, ax1 = plt.subplots(figsize=(14, 12), dpi=300)
    sns.heatmap(corr_mat, cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 6.5}, ax=ax1)
    ax1.set_title("Training Set Correlation Matrix (16 Original + 6 Engineered Features)", fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase4_feature_correlation.png", bbox_inches="tight")
    plt.close()

    # Figure 2: Mutual Information Ranking
    fig2, ax2 = plt.subplots(figsize=(10, 6.5), dpi=300)
    top_mi = sel_results["mutual_info_series"].head(15)
    sns.barplot(x=top_mi.values, y=top_mi.index, ax=ax2, palette="Blues_r")
    ax2.set_title("Top 15 Features by Mutual Information with Bean Variety", fontweight="bold")
    ax2.set_xlabel("Mutual Information Score (Nats)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase4_mutual_information.png", bbox_inches="tight")
    plt.close()

    # Figure 3: ANOVA Ranking
    fig3, ax3 = plt.subplots(figsize=(10, 6.5), dpi=300)
    top_anova = sel_results["anova_df"].head(15)
    sns.barplot(data=top_anova, x="ANOVA F-Score", y="Feature", ax=ax3, palette="Greens_r")
    ax3.set_title("Top 15 Features by ANOVA F-Score", fontweight="bold")
    ax3.set_xlabel("ANOVA F-Score")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase4_anova_ranking.png", bbox_inches="tight")
    plt.close()

    # Figure 4: Random Forest Feature Importance
    fig4, ax4 = plt.subplots(figsize=(10, 6.5), dpi=300)
    top_rf = sel_results["rf_importance_series"].head(15)
    sns.barplot(x=top_rf.values, y=top_rf.index, ax=ax4, palette="Purples_r")
    ax4.set_title("Random Forest Gini Feature Importance (Training Set)", fontweight="bold")
    ax4.set_xlabel("Relative Feature Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase4_feature_importance.png", bbox_inches="tight")
    plt.close()

    # Figure 5: PCA Scree Plot
    fig5, ax5 = plt.subplots(figsize=(9, 4.5), dpi=300)
    x_comps = range(1, 11)
    ax5.bar(x_comps, pca.explained_variance_ratio_ * 100, color="#2b5c8f", alpha=0.7, label="Individual Component Variance (%)")
    ax5.plot(x_comps, np.cumsum(pca.explained_variance_ratio_) * 100, color="#d95f02", marker="o", linewidth=2, label="Cumulative Variance (%)")
    ax5.set_xticks(list(x_comps))
    ax5.set_xlabel("Principal Component")
    ax5.set_ylabel("Explained Variance (%)")
    ax5.set_title("PCA Scree Plot (Training Set Variance Decomposition)", fontweight="bold")
    ax5.legend(loc="center right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase4_pca_scree.png", bbox_inches="tight")
    plt.close()

    # Figure 6: PCA 2D Visualization
    fig6, ax6 = plt.subplots(figsize=(10, 7), dpi=300)
    pca_df = pd.DataFrame({
        "PC1": X_train_pca[:, 0],
        "PC2": X_train_pca[:, 1],
        "Class": y_train.values,
    })
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Class", palette="tab10", alpha=0.5, s=16, ax=ax6)
    ax6.set_title(f"Training Set 2D PCA (PC1: {pca.explained_variance_ratio_[0]*100:.1f}%, PC2: {pca.explained_variance_ratio_[1]*100:.1f}%)", fontweight="bold")
    ax6.set_xlabel("Principal Component 1")
    ax6.set_ylabel("Principal Component 2")
    ax6.legend(title="Cultivar", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase4_pca_visualization.png", bbox_inches="tight")
    plt.close()

    # Figure 7: Feature Comparison Ranking (Mean Rank Comparison)
    fig7, ax7 = plt.subplots(figsize=(11, 7), dpi=300)
    top_ranked = rankings_df.head(12)
    sns.barplot(data=top_ranked, x="Mean Rank", y="Feature", ax=ax7, palette="Blues_r")
    ax7.set_title("Top 12 Consolidated Feature Rankings (Multi-Criteria Average Rank)", fontweight="bold")
    ax7.set_xlabel("Average Rank (Lower indicates higher consistency across MI, ANOVA, and RF)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase4_feature_comparison.png", bbox_inches="tight")
    plt.close()

    # 7. Reports & Dictionaries Export
    print("7/7 Exporting Phase 4 reports and JSON metadata...")
    feature_dict = {
        "original_features": FEATURE_NAMES,
        "engineered_features": ENGINEERED_FEATURE_METADATA,
        "selected_top_10_features": rankings_df.head(10)["Feature"].tolist(),
    }
    with open(phase4_rep_dir / "feature_dictionary.json", "w", encoding="utf-8") as f:
        json.dump(feature_dict, f, indent=2)

    selection_summary = {
        "train_samples": len(X_train),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "total_candidate_features": len(all_feature_names),
        "top_features_by_mutual_info": sel_results["mutual_info_series"].head(5).to_dict(),
        "top_features_by_anova": sel_results["anova_df"].head(5)[["Feature", "ANOVA F-Score"]].to_dict(orient="records"),
        "top_features_by_rf": sel_results["rf_importance_series"].head(5).to_dict(),
        "pca_cumulative_variance_3_components": float(pca_var_df.iloc[2]["Cumulative Explained Variance"]),
        "selected_candidate_subset": rankings_df.head(10)["Feature"].tolist(),
    }
    with open(phase4_rep_dir / "feature_selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(selection_summary, f, indent=2)

    # HTML Reports
    html_fe = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>BEANWISE - Phase 4 Feature Engineering Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; line-height: 1.6; color: #24292e; background: #f6f8fa; margin: 0; padding: 0; }}
        .container {{ max-width: 1000px; margin: 40px auto; background: #ffffff; padding: 40px 50px; border: 1px solid #e1e4e8; border-radius: 8px; }}
        h1 {{ font-size: 24px; color: #0366d6; border-bottom: 2px solid #eaecef; padding-bottom: 10px; }}
        h2 {{ font-size: 19px; color: #1b1f23; border-bottom: 1px solid #eaecef; padding-bottom: 6px; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 18px 0; font-size: 13.5px; }}
        th, td {{ padding: 8px 12px; border: 1px solid #e1e4e8; text-align: left; }}
        th {{ background: #fafbfc; }}
        .badge {{ background: #f1f8ff; border: 1px solid #c8e1ff; padding: 12px 16px; border-radius: 6px; font-size: 13.5px; margin: 16px 0; }}
        .fig-box {{ text-align: center; margin: 24px 0; background: #fafbfc; padding: 15px; border: 1px solid #e1e4e8; border-radius: 6px; }}
        .fig-box img {{ max-width: 100%; border-radius: 4px; border: 1px solid #eaecef; }}
    </style>
</head>
<body>
<div class="container">
    <h1>BEANWISE: Feature Engineering Technical Report</h1>
    <p><strong>Phase 4 Milestone Report</strong></p>

    <div class="badge">
        <strong>Input Dataset:</strong> dataset/processed/cleaned_dry_beans.csv (13,543 records)<br>
        <strong>Engineered Features Created:</strong> 6 candidate ratios<br>
        <strong>Total Feature Dimension:</strong> 22 attributes (16 Original + 6 Engineered)
    </div>

    <h2>1. Engineered Feature Formulations</h2>
    <table>
        <thead>
            <tr><th>Feature Name</th><th>Formula</th><th>Physical / Morphological Purpose</th></tr>
        </thead>
        <tbody>
            <tr><td>Area_to_Perimeter</td><td>Area / Perimeter</td><td>Dimensional boundary efficiency</td></tr>
            <tr><td>Axis_Ratio</td><td>MajorAxisLength / MinorAxisLength</td><td>Cross-validation of AspectRation</td></tr>
            <tr><td>Axis_Difference</td><td>MajorAxisLength - MinorAxisLength</td><td>Absolute axial disparity in pixels</td></tr>
            <tr><td>Convex_Area_Ratio</td><td>ConvexArea / Area</td><td>Inverse solidity measuring boundary concavity</td></tr>
            <tr><td>Compactness_Roundness_Ratio</td><td>Compactness / roundness</td><td>Relative ratio between radial and circular compact forms</td></tr>
            <tr><td>EquivDiameter_to_MajorAxis</td><td>EquivDiameter / MajorAxisLength</td><td>Normalized circle-equivalent diameter fraction</td></tr>
        </tbody>
    </table>

    <div class="fig-box">
        <img src="../figures/phase4_feature_correlation.png" alt="Feature Correlation Heatmap">
    </div>

    <h2>2. Data Leakage Prevention and Partitioning</h2>
    <p>
        The dataset was split using stratified 3-way sampling with fixed random state 42:
        Training Set: 9,480 samples (70.0%) | Validation Set: 2,031 samples (15.0%) | Test Set: 2,032 samples (15.0%).
        All feature evaluation and selection steps were performed exclusively on the training partition.
    </p>
    {split_summary_df.to_html(classes="table", index=False)}
</div>
</body>
</html>
"""
    with open(phase4_rep_dir / "feature_engineering_report.html", "w", encoding="utf-8") as f:
        f.write(html_fe)

    print("Phase 4 Artifact Generation Completed Successfully.")


if __name__ == "__main__":
    run_phase4_artifacts()
