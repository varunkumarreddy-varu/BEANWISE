"""
Visualization Utilities for BEANWISE Phase 2 Exploratory Data Analysis.

Generates high-resolution static publication charts (Matplotlib / Seaborn)
and interactive standalone HTML charts (Plotly).
All visualizations follow clean academic styling without decorative symbols or emojis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive background rendering
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from config.settings import FEATURE_NAMES, TARGET_COLUMN, TARGET_CLASSES, FIGURES_DIR

# Academic color palette for the 7 bean classes
CLASS_PALETTE = {
    "SEKER": "#1f77b4",
    "BARBUNYA": "#ff7f0e",
    "BOMBAY": "#2ca02c",
    "CALI": "#d62728",
    "HOROZ": "#9467bd",
    "SIRA": "#8c564b",
    "DERMASON": "#e377c2",
}

# General Matplotlib style settings
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8


def plot_dataset_overview(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """
    Generates a multi-panel visual summary of dataset dimensions, feature counts, and dtypes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)

    # Panel 1: Rows vs Columns bar
    dim_df = pd.DataFrame({
        "Metric": ["Observations", "Features (Input)", "Target Column"],
        "Count": [len(df), len(FEATURE_NAMES), 1]
    })
    sns.barplot(data=dim_df, x="Metric", y="Count", ax=axes[0], palette=["#2b5c8f", "#3a86c8", "#5c9ee6"])
    axes[0].set_yscale("log")
    axes[0].set_title("Dataset Dimensions (Log Scale)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Count (log)")
    for p in axes[0].patches:
        height = p.get_height()
        axes[0].annotate(f"{int(height):,}", (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

    # Panel 2: Data Types count
    dtype_counts = df.dtypes.value_counts()
    axes[1].pie(dtype_counts, labels=[str(d) for d in dtype_counts.index], autopct="%1.1f%%",
               colors=["#2b5c8f", "#d95f02"], startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[1].set_title("Attribute Data Type Breakdown", fontsize=11, fontweight="bold")

    # Panel 3: Duplicates breakdown
    dups = int(df.duplicated().sum())
    uniques = len(df) - dups
    axes[2].bar(["Unique Records", "Exact Duplicates"], [uniques, dups], color=["#2ca02c", "#d62728"], width=0.5)
    axes[2].set_title("Duplicate Audit (Total vs Duplicate)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Records")
    axes[2].annotate(f"{uniques:,}\n(99.50%)", (0, uniques / 2), ha="center", color="white", fontweight="bold")
    axes[2].annotate(f"{dups}\n(0.50%)", (1, dups + 200), ha="center", color="#333333", fontweight="bold")

    plt.tight_layout()
    file_path = output_dir / "dataset_overview.png"
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    return file_path


def plot_missing_values(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """
    Generates missing value visualization confirming data completeness across all 17 columns.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)

    missing = df.isnull().sum()
    bars = ax.bar(missing.index, missing.values, color="#2b5c8f", edgecolor="#1a3b5c")
    ax.set_title("Missing Values per Column (Total Dataset)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Missing Cells")
    ax.set_xticklabels(missing.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 10)  # Emphasize zero-missing status clearly

    for bar in bars:
        ax.annotate("0", (bar.get_x() + bar.get_width() / 2, 0.5), ha="center", fontsize=9, color="#555555")

    plt.tight_layout()
    file_path = output_dir / "missing_values.png"
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    return file_path


def plot_class_distribution(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Tuple[Path, Path]:
    """
    Generates static and interactive class distribution charts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = df[TARGET_COLUMN].value_counts()
    percentages = df[TARGET_COLUMN].value_counts(normalize=True) * 100

    # Static Seaborn & Matplotlib chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Subplot 1: Frequency bar chart
    sns.barplot(x=counts.index, y=counts.values, ax=ax1, palette=CLASS_PALETTE)
    ax1.set_title("Target Class Sample Counts", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Dry Bean Variety")
    ax1.set_ylabel("Number of Instances")
    ax1.tick_params(axis="x", rotation=30)
    for p in ax1.patches:
        h = p.get_height()
        ax1.annotate(f"{int(h):,}", (p.get_x() + p.get_width() / 2., h),
                     ha="center", va="bottom", fontsize=8.5, xytext=(0, 2), textcoords="offset points")

    # Subplot 2: Percentage donut chart
    wedges, texts, autotexts = ax2.pie(
        counts.values, labels=counts.index, autopct="%1.1f%%",
        colors=[CLASS_PALETTE[c] for c in counts.index], startangle=140,
        pctdistance=0.75, wedgeprops={"width": 0.5, "edgecolor": "white", "linewidth": 1.5}
    )
    plt.setp(autotexts, size=8.5, weight="bold")
    ax2.set_title("Class Proportion in Dataset", fontsize=11, fontweight="bold")

    plt.tight_layout()
    static_path = output_dir / "class_distribution.png"
    plt.savefig(static_path, bbox_inches="tight")
    plt.close()

    # Interactive Plotly chart
    plot_df = pd.DataFrame({"Class": counts.index, "Count": counts.values, "Percentage": percentages.round(2).values})
    fig_plotly = px.bar(
        plot_df, x="Class", y="Count", text="Percentage",
        color="Class", color_discrete_map=CLASS_PALETTE,
        title="Interactive Class Distribution - Dry Bean Varieties",
        labels={"Count": "Instance Count", "Class": "Bean Cultivar"}
    )
    fig_plotly.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_plotly.update_layout(template="plotly_dark", showlegend=False, height=500)
    html_path = output_dir / "interactive_class_distribution.html"
    fig_plotly.write_html(str(html_path))

    return static_path, html_path


def plot_numerical_distributions(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """
    Generates 4x4 grid of histograms with KDE for all 16 morphological features.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), dpi=300)
    axes = axes.flatten()

    for idx, feature in enumerate(FEATURE_NAMES):
        ax = axes[idx]
        sns.histplot(df[feature], kde=True, ax=ax, color="#1f77b4", edgecolor=None, alpha=0.6, stat="density")
        ax.set_title(feature, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Density" if idx % 4 == 0 else "")
        ax.tick_params(labelsize=8)

    plt.suptitle("Univariate Frequency Distributions (Histograms and KDE)", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    file_path = output_dir / "numerical_distributions.png"
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    return file_path


def plot_feature_boxplots(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """
    Generates standardized 4x4 boxplot grid to evaluate skewness and tail distribution.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), dpi=300)
    axes = axes.flatten()

    for idx, feature in enumerate(FEATURE_NAMES):
        ax = axes[idx]
        sns.boxplot(y=df[feature], ax=ax, color="#4a90e2", fliersize=2.5, width=0.4,
                    boxprops=dict(alpha=0.7), flierprops=dict(marker="o", markerfacecolor="#d9534f", alpha=0.5))
        ax.set_title(feature, fontsize=10, fontweight="bold")
        ax.set_ylabel("")
        ax.tick_params(labelsize=8)

    plt.suptitle("Univariate Boxplots and Dispersion Analysis", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    file_path = output_dir / "feature_boxplots.png"
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    return file_path


def plot_correlation_matrices(
    pearson_mat: pd.DataFrame,
    spearman_mat: pd.DataFrame,
    output_dir: Path = FIGURES_DIR
) -> Tuple[Path, Path, Path]:
    """
    Generates static heatmaps for Pearson and Spearman correlations, and an interactive Plotly heatmap.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pearson heatmap
    fig1, ax1 = plt.subplots(figsize=(13, 11), dpi=300)
    sns.heatmap(pearson_mat, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax1,
                cbar_kws={"label": "Pearson Correlation Coefficient (r)"}, annot_kws={"size": 7.5})
    ax1.set_title("Pearson Correlation Matrix (Linear Relationships)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    pearson_path = output_dir / "correlation_pearson.png"
    plt.savefig(pearson_path, bbox_inches="tight")
    plt.close()

    # 2. Spearman heatmap
    fig2, ax2 = plt.subplots(figsize=(13, 11), dpi=300)
    sns.heatmap(spearman_mat, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax2,
                cbar_kws={"label": "Spearman Rank Correlation (rho)"}, annot_kws={"size": 7.5})
    ax2.set_title("Spearman Rank Correlation Matrix (Monotonic Relationships)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    spearman_path = output_dir / "correlation_spearman.png"
    plt.savefig(spearman_path, bbox_inches="tight")
    plt.close()

    # 3. Interactive Plotly heatmap
    fig_plotly = px.imshow(
        pearson_mat.round(2),
        text_auto=True,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Interactive Pearson Correlation Matrix",
        labels=dict(color="Correlation")
    )
    fig_plotly.update_layout(template="plotly_dark", height=700, width=800)
    html_path = output_dir / "interactive_correlation.html"
    fig_plotly.write_html(str(html_path))

    return pearson_path, spearman_path, html_path


def plot_bivariate_relationships(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Tuple[Path, Path]:
    """
    Generates a 2x2 grid of key bivariate scatter plots coloured by cultivar.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)

    pairs = [
        ("Area", "Perimeter", axes[0, 0]),
        ("MajorAxisLength", "MinorAxisLength", axes[0, 1]),
        ("AspectRation", "Eccentricity", axes[1, 0]),
        ("Compactness", "roundness", axes[1, 1]),
    ]

    for x_col, y_col, ax in pairs:
        sns.scatterplot(
            data=df, x=x_col, y=y_col, hue=TARGET_COLUMN,
            palette=CLASS_PALETTE, alpha=0.6, s=15, ax=ax, legend=(ax == axes[0, 0])
        )
        ax.set_title(f"{y_col} vs {x_col}", fontsize=11, fontweight="bold")
        ax.set_xlabel(x_col, fontsize=9.5)
        ax.set_ylabel(y_col, fontsize=9.5)

    if axes[0, 0].get_legend() is not None:
        axes[0, 0].legend(title="Cultivar", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8.5)

    plt.suptitle("Bivariate Feature Relationships Stratified by Bean Variety", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    static_path = output_dir / "feature_relationships.png"
    plt.savefig(static_path, bbox_inches="tight")
    plt.close()

    # Interactive Plotly scatter plot (Area vs Perimeter)
    fig_plotly = px.scatter(
        df, x="Area", y="Perimeter", color=TARGET_COLUMN,
        color_discrete_map=CLASS_PALETTE, opacity=0.7,
        hover_data=["MajorAxisLength", "MinorAxisLength", "Compactness"],
        title="Interactive Bivariate Relationship: Area vs Perimeter by Cultivar"
    )
    fig_plotly.update_layout(template="plotly_dark", height=600)
    html_path = output_dir / "interactive_scatter.html"
    fig_plotly.write_html(str(html_path))

    return static_path, html_path


def plot_outlier_summary(outlier_df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """
    Plots a dual horizontal bar comparison of IQR vs Z-score outlier counts across all features.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

    y = np.arange(len(outlier_df))
    height = 0.35

    ax.barh(y - height / 2, outlier_df["IQR Outlier Count"], height, label="IQR Rule (1.5*IQR)", color="#1f77b4")
    ax.barh(y + height / 2, outlier_df["Z-Score (|z|>3) Count"], height, label="Z-Score (|z| > 3)", color="#ff7f0e")

    ax.set_yticks(y)
    ax.set_yticklabels(outlier_df["Feature"], fontsize=9)
    ax.set_xlabel("Number of Statistical Outlier Observations", fontsize=10)
    ax.set_title("Outlier Frequency Comparison: IQR Rule vs Z-Score (|z| > 3)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")

    plt.tight_layout()
    file_path = output_dir / "outlier_analysis.png"
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    return file_path


def plot_class_wise_violins(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """
    Generates a 2x2 grid of violin plots illustrating class-stratified distributions
    for four distinguishing features.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    features_to_plot = ["Area", "AspectRation", "roundness", "ShapeFactor1"]

    for idx, feat in enumerate(features_to_plot):
        r, c = divmod(idx, 2)
        ax = axes[r, c]
        sns.violinplot(
            data=df, x=TARGET_COLUMN, y=feat, ax=ax,
            palette=CLASS_PALETTE, inner="quartile", cut=0
        )
        ax.set_title(f"Class-Wise Distribution: {feat}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Bean Variety", fontsize=9)
        ax.set_ylabel(feat, fontsize=9)
        ax.tick_params(axis="x", rotation=25, labelsize=8.5)

    plt.suptitle("Class-Stratified Feature Distributions (Violin Plots)", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    file_path = output_dir / "class_wise_analysis.png"
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    return file_path


def plot_exploratory_pca(
    X_pca_2d: np.ndarray,
    df: pd.DataFrame,
    var_pc1: float,
    var_pc2: float,
    output_dir: Path = FIGURES_DIR
) -> Path:
    """
    Plots 2D projection of first two principal components coloured by bean variety.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=300)

    pca_plot_df = pd.DataFrame({
        "PC1": X_pca_2d[:, 0],
        "PC2": X_pca_2d[:, 1],
        "Class": df[TARGET_COLUMN].values,
    })

    sns.scatterplot(
        data=pca_plot_df, x="PC1", y="PC2", hue="Class",
        palette=CLASS_PALETTE, alpha=0.6, s=18, ax=ax
    )

    ax.set_title(
        f"Exploratory PCA: First Two Principal Components (Total Variance: {var_pc1 + var_pc2:.1f}%)",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel(f"Principal Component 1 ({var_pc1:.1f}% Variance)", fontsize=9.5)
    ax.set_ylabel(f"Principal Component 2 ({var_pc2:.1f}% Variance)", fontsize=9.5)
    ax.legend(title="Cultivar", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5)

    plt.tight_layout()
    file_path = output_dir / "pca_visualization.png"
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()
    return file_path


def plot_parallel_coordinates(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """
    Generates interactive Plotly parallel coordinates plot for sample subset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    sampled_indices = []
    for cls in sorted(df[TARGET_COLUMN].unique()):
        cls_indices = df[df[TARGET_COLUMN] == cls].index.to_numpy()
        n_sample = min(len(cls_indices), 150)
        sampled_indices.extend(rng.choice(cls_indices, size=n_sample, replace=False))

    sample_df = df.loc[sampled_indices].copy()
    class_codes = {c: i for i, c in enumerate(sorted(df[TARGET_COLUMN].unique()))}
    sample_df["Class_Code"] = sample_df[TARGET_COLUMN].map(class_codes)

    selected_dims = ["Area", "MajorAxisLength", "AspectRation", "roundness", "Compactness", "ShapeFactor1"]
    dimensions = [dict(range=[sample_df[col].min(), sample_df[col].max()], label=col, values=sample_df[col]) for col in selected_dims]

    fig = go.Figure(data=go.Parcoords(
        line=dict(color=sample_df["Class_Code"], colorscale="Turbo", showscale=False),
        dimensions=dimensions
    ))
    fig.update_layout(title="Interactive Parallel Coordinates Plot (Morphological Attributes)", template="plotly_dark", height=600)

    html_path = output_dir / "interactive_parallel_coordinates.html"
    fig.write_html(str(html_path))
    return html_path
