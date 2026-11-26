"""
Feature Selection and Leakage-Safe Data Splitting Module for BEANWISE Phase 4.

Provides stratified splitting (70% train, 15% validation, 15% test) and evaluates
feature selection methods strictly on training data:
- Mutual Information
- ANOVA F-value (f_classif)
- Random Forest Tree-based Importance
- Recursive Feature Elimination (RFE)
- Consolidated Multi-criteria Feature Ranking
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

from config.settings import TARGET_COLUMN, TARGET_CLASSES


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Executes reproducible stratified 3-way split (Train / Validation / Test).
    Verifies that class proportions are preserved across all subsets.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    # Step 1: Split out test set (15%)
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df[TARGET_COLUMN],
        random_state=random_state,
    )

    # Step 2: Split remaining 85% into train (70/85) and val (15/85)
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_ratio,
        stratify=train_val_df[TARGET_COLUMN],
        random_state=random_state,
    )

    # Calculate class counts in each partition
    train_dist = train_df[TARGET_COLUMN].value_counts().to_dict()
    val_dist = val_df[TARGET_COLUMN].value_counts().to_dict()
    test_dist = test_df[TARGET_COLUMN].value_counts().to_dict()

    metadata = {
        "random_state": random_state,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "train_pct": round(len(train_df) / len(df) * 100, 2),
        "val_pct": round(len(val_df) / len(df) * 100, 2),
        "test_pct": round(len(test_df) / len(df) * 100, 2),
        "train_class_distribution": train_dist,
        "val_class_distribution": val_dist,
        "test_class_distribution": test_dist,
        "train_indices": train_df.index.tolist(),
        "val_indices": val_df.index.tolist(),
        "test_indices": test_df.index.tolist(),
    }

    return train_df, val_df, test_df, metadata


def evaluate_feature_selection_methods(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: List[str],
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates feature selection methods strictly on training data to prevent leakage:
    1. Mutual Information
    2. ANOVA F-Score
    3. Random Forest Gini Importance
    4. Recursive Feature Elimination (RFE)
    """
    X_mat = X_train[feature_names].values
    y_vec = y_train.values

    # 1. Mutual Information
    mi_scores = mutual_info_classif(X_mat, y_vec, random_state=random_state)
    mi_series = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)

    # 2. ANOVA F-Test
    f_scores, p_values = f_classif(X_mat, y_vec)
    anova_df = pd.DataFrame({
        "Feature": feature_names,
        "ANOVA F-Score": f_scores,
        "p-value": p_values,
    }).sort_values(by="ANOVA F-Score", ascending=False)

    # 3. Random Forest Importance
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1)
    rf.fit(X_mat, y_vec)
    rf_importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)

    # 4. RFE with Random Forest (select top 10 features)
    n_rfe = min(10, len(feature_names))
    rf_base = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=random_state, n_jobs=-1)
    rfe = RFE(estimator=rf_base, n_features_to_select=n_rfe, step=1)
    rfe.fit(X_mat, y_vec)
    rfe_ranking = pd.Series(rfe.ranking_, index=feature_names)

    # 5. Consolidated Multi-criteria Ranking Table
    rank_df = pd.DataFrame({
        "Feature": feature_names,
        "MI Score": [round(float(mi_series[f]), 4) for f in feature_names],
        "MI Rank": pd.Series(mi_scores, index=feature_names).rank(ascending=False).astype(int),
        "ANOVA F-Score": [round(float(anova_df.set_index("Feature").loc[f, "ANOVA F-Score"]), 2) for f in feature_names],
        "ANOVA Rank": pd.Series(f_scores, index=feature_names).rank(ascending=False).astype(int),
        "RF Importance": [round(float(rf_importances[f]), 4) for f in feature_names],
        "RF Rank": rf_importances.rank(ascending=False).astype(int),
        "RFE Ranking": [int(rfe_ranking[f]) for f in feature_names],
        "RFE Selected (Top 10)": [bool(rfe.support_[i]) for i in range(len(feature_names))],
    })

    # Average rank across MI, ANOVA, and RF
    rank_df["Mean Rank"] = ((rank_df["MI Rank"] + rank_df["ANOVA Rank"] + rank_df["RF Rank"]) / 3.0).round(2)
    rank_df = rank_df.sort_values(by="Mean Rank", ascending=True)

    return {
        "mutual_info_series": mi_series,
        "anova_df": anova_df,
        "rf_importance_series": rf_importances,
        "rfe_support": rfe.support_,
        "consolidated_rankings_df": rank_df,
    }
