"""Detailed factor-behavior analysis for teacher-loop rounds.

This module turns a trained teacher's out-of-sample scored rows into
actionable factor evidence that can be re-used by:

1. the outer research loop when proposing the next teacher,
2. the inner apprentice loop when teacher signals are hidden,
3. human-readable audit reports.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .registry import build_feature_registry

MAX_CONTRIB_ROWS_PER_YEAR = 3000
MAX_PDP_SAMPLE_ROWS_PER_YEAR = 600
PDP_GRID_POINTS = 11
BIN_COUNT = 10
RANDOM_SEED = 42


@dataclass
class FactorAnalysisResult:
    summary: Dict[str, Any]
    artifact_paths: List[str]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _feature_meta_map() -> Dict[str, Dict[str, Any]]:
    registry = build_feature_registry()
    return {item["feature_name"]: item for item in registry["features"]}


def _broad_group(category: str) -> str:
    mapping = {
        "returns": "momentum",
        "moving_average": "momentum",
        "position": "price_position",
        "price_level": "price_position",
        "kdj": "kdj",
        "volatility": "volatility",
        "amount": "amount_volume",
        "volume": "amount_volume",
        "candle": "candlestick",
    }
    return mapping.get(category, category or "other")


def _feature_matrix(df: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    return df.loc[:, list(feature_columns)].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)


def _predict_scores(model: Any, model_family: str, feature_columns: Sequence[str], df: pd.DataFrame) -> np.ndarray:
    x = _feature_matrix(df, feature_columns)
    if model_family in {"logistic_regression", "xgb_classification_gpu"}:
        return model.predict_proba(x)[:, 1]
    return model.predict(x)


def _score_contributions_for_rows(
    *,
    model: Any,
    model_family: str,
    feature_columns: Sequence[str],
    frame: pd.DataFrame,
) -> tuple[Optional[pd.DataFrame], str]:
    if frame.empty:
        return None, "empty"
    x = _feature_matrix(frame, feature_columns)
    feature_columns = list(feature_columns)

    if model_family in {"xgb_regression_gpu", "xgb_classification_gpu"}:
        import xgboost as xgb

        dmatrix = xgb.DMatrix(x, feature_names=feature_columns)
        contrib = model.get_booster().predict(dmatrix, pred_contribs=True, validate_features=False)
        contrib_df = pd.DataFrame(contrib[:, :-1], columns=feature_columns)
        contrib_df["bias"] = contrib[:, -1]
        return contrib_df, "tree_shap_pred_contribs"

    if model_family in {"ridge_regression", "logistic_regression"}:
        scaler = model.named_steps["scaler"]
        z = scaler.transform(x)
        if model_family == "ridge_regression":
            coeffs = model.named_steps["ridge"].coef_.reshape(-1)
            bias = float(np.asarray(model.named_steps["ridge"].intercept_).reshape(-1)[0])
            method = "linear_exact_margin"
        else:
            coeffs = model.named_steps["logit"].coef_.reshape(-1)
            bias = float(np.asarray(model.named_steps["logit"].intercept_).reshape(-1)[0])
            method = "linear_exact_logit_margin"
        contrib_df = pd.DataFrame(z * coeffs.reshape(1, -1), columns=feature_columns)
        contrib_df["bias"] = bias
        return contrib_df, method

    return None, "unsupported_local_contrib"


def _per_date_corr(group: pd.DataFrame, score_col: str, label_col: str, method: str) -> float:
    if len(group) < 5:
        return np.nan
    score = group[score_col]
    label = group[label_col]
    if score.nunique() < 2 or label.nunique() < 2:
        return np.nan
    return float(score.corr(label, method=method))


def _daily_rank_ic(df: pd.DataFrame, value_col: str, label_col: str = "future_return_5d") -> float:
    grouped = df.groupby("signal_date", sort=True)
    values = grouped.apply(_per_date_corr, score_col=value_col, label_col=label_col, method="spearman").dropna()
    return float(values.mean()) if len(values) else np.nan


def _sample_frame(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy()
    return frame.sample(n=max_rows, random_state=RANDOM_SEED).sort_index().copy()


def _derive_preferred_direction(
    feature: str,
    pool_df: pd.DataFrame,
    selected_df: pd.DataFrame,
) -> str:
    pool_vals = pool_df[feature].dropna()
    selected_vals = selected_df[feature].dropna()
    if len(pool_vals) < 10 or len(selected_vals) < 10:
        return "mixed"
    return "higher" if float(selected_vals.median()) >= float(pool_vals.median()) else "lower"


def _preferred_band_summary(
    feature: str,
    pool_df: pd.DataFrame,
    selected_df: pd.DataFrame,
) -> Dict[str, Any]:
    pool_vals = pool_df[feature].dropna()
    selected_vals = selected_df[feature].dropna()
    if len(pool_vals) < 10 or len(selected_vals) < 10:
        return {
            "feature": feature,
            "preferred_direction": "mixed",
            "selected_q25": None,
            "selected_median": None,
            "selected_q75": None,
            "pool_median": None,
            "effect_size": None,
        }
    std = float(pool_vals.std()) if len(pool_vals) > 1 else 0.0
    sel_med = float(selected_vals.median())
    pool_med = float(pool_vals.median())
    return {
        "feature": feature,
        "preferred_direction": "higher" if sel_med >= pool_med else "lower",
        "selected_q25": float(selected_vals.quantile(0.25)),
        "selected_median": sel_med,
        "selected_q75": float(selected_vals.quantile(0.75)),
        "pool_median": pool_med,
        "effect_size": abs(sel_med - pool_med) / (std + 1e-9),
    }


def _save_bar_chart(df: pd.DataFrame, output_path: Path, title: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(df) * 0.35)))
    ordered = df.sort_values("mean_abs_contribution", ascending=True)
    ax.barh(ordered["feature"], ordered["mean_abs_contribution"], color="#1565c0", alpha=0.85)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean |contribution|")
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_corr_heatmap(corr_df: pd.DataFrame, output_path: Path, title: str) -> None:
    if corr_df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(corr_df.columns) * 0.45), max(7, len(corr_df.columns) * 0.45)))
    mat = corr_df.to_numpy(dtype=float)
    im = ax.imshow(mat, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index, fontsize=7)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_pdp_plot(pdp_df: pd.DataFrame, output_path: Path, title: str) -> None:
    if pdp_df.empty:
        return
    features = list(dict.fromkeys(pdp_df["feature"].tolist()))
    fig, axes = plt.subplots(len(features), 1, figsize=(8.5, max(3.2, 2.8 * len(features))), sharex=False)
    if len(features) == 1:
        axes = [axes]
    for ax, feature in zip(axes, features):
        frame = pdp_df[pdp_df["feature"] == feature].sort_values("grid_index")
        ax.plot(frame["feature_value"], frame["mean_prediction"], color="#2e7d32", linewidth=1.8)
        if "std_prediction" in frame.columns:
            low = frame["mean_prediction"] - frame["std_prediction"]
            high = frame["mean_prediction"] + frame["std_prediction"]
            ax.fill_between(frame["feature_value"], low, high, color="#66bb6a", alpha=0.20)
        ax.set_title(feature, loc="left", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.25)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _single_feature_bins(
    oos_df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for feature in feature_columns:
        feat_series = oos_df[feature].replace([np.inf, -np.inf], np.nan).dropna()
        if feat_series.nunique() < 6:
            continue
        quantiles = min(BIN_COUNT, int(feat_series.nunique()))
        try:
            bucket = pd.qcut(feat_series, q=quantiles, duplicates="drop")
        except ValueError:
            continue
        frame = oos_df.loc[feat_series.index, [feature, "score", "future_return_5d", "bucket"]].copy()
        frame["bin_label"] = bucket.astype(str).to_numpy()
        bin_stats = (
            frame.groupby("bin_label", sort=False)
            .agg(
                sample_count=(feature, "size"),
                feature_mean=(feature, "mean"),
                feature_min=(feature, "min"),
                feature_max=(feature, "max"),
                score_mean=("score", "mean"),
                future_return_mean=("future_return_5d", "mean"),
                q5_share=("bucket", lambda s: float((s == 5).mean())),
            )
            .reset_index()
        )
        bin_stats["feature"] = feature
        bin_stats["bin_index"] = np.arange(len(bin_stats), dtype=int)
        all_rows.extend(bin_stats.to_dict("records"))
        if len(bin_stats) >= 3:
            score_mono = float(pd.Series(bin_stats["bin_index"]).corr(bin_stats["score_mean"], method="spearman"))
            return_mono = float(pd.Series(bin_stats["bin_index"]).corr(bin_stats["future_return_mean"], method="spearman"))
        else:
            score_mono = np.nan
            return_mono = np.nan
        best_row = bin_stats.sort_values("future_return_mean", ascending=False).iloc[0]
        summary_rows.append(
            {
                "feature": feature,
                "bin_count": int(len(bin_stats)),
                "spearman_bin_to_score": score_mono,
                "spearman_bin_to_return": return_mono,
                "best_bin_label": str(best_row["bin_label"]),
                "best_bin_feature_min": _safe_float(best_row["feature_min"]),
                "best_bin_feature_max": _safe_float(best_row["feature_max"]),
                "best_bin_feature_mean": _safe_float(best_row["feature_mean"]),
                "best_bin_return": _safe_float(best_row["future_return_mean"]),
                "best_bin_q5_share": _safe_float(best_row["q5_share"]),
                "shape_hint": (
                    "monotonic_positive"
                    if pd.notna(return_mono) and return_mono >= 0.60
                    else "monotonic_negative"
                    if pd.notna(return_mono) and return_mono <= -0.60
                    else "nonlinear_or_flat"
                ),
            }
        )
    return pd.DataFrame(all_rows), pd.DataFrame(summary_rows)


def _group_effectiveness(
    *,
    oos_df: pd.DataFrame,
    feature_columns: Sequence[str],
    meta_map: Mapping[str, Dict[str, Any]],
    shap_global_df: pd.DataFrame,
) -> pd.DataFrame:
    feature_ic_rows: List[Dict[str, Any]] = []
    for feature in feature_columns:
        frame = oos_df[["signal_date", feature, "future_return_5d"]].dropna().copy()
        if frame.empty:
            continue
        rank_ic = _daily_rank_ic(frame.rename(columns={feature: "value"}), "value")
        feature_ic_rows.append({"feature": feature, "rank_ic_mean": rank_ic})
    ic_df = pd.DataFrame(feature_ic_rows)
    if ic_df.empty:
        return pd.DataFrame()
    ic_df["category"] = ic_df["feature"].map(lambda feat: meta_map.get(feat, {}).get("category", "other"))
    ic_df["broad_group"] = ic_df["category"].map(_broad_group)

    shap_view = shap_global_df[["feature", "mean_abs_contribution"]].copy() if not shap_global_df.empty else pd.DataFrame(columns=["feature", "mean_abs_contribution"])
    shap_view["category"] = shap_view["feature"].map(lambda feat: meta_map.get(feat, {}).get("category", "other"))
    shap_view["broad_group"] = shap_view["category"].map(_broad_group)

    total_abs = float(shap_view["mean_abs_contribution"].sum()) if not shap_view.empty else np.nan
    rows: List[Dict[str, Any]] = []
    for group_name, frame in ic_df.groupby("broad_group", sort=True):
        group_features = frame["feature"].tolist()
        best = frame.reindex(frame["rank_ic_mean"].abs().sort_values(ascending=False).index).iloc[0]
        shap_group = shap_view[shap_view["broad_group"] == group_name]
        shap_abs = float(shap_group["mean_abs_contribution"].sum()) if not shap_group.empty else np.nan
        rows.append(
            {
                "broad_group": group_name,
                "feature_count": int(len(group_features)),
                "features": ", ".join(group_features),
                "mean_abs_rank_ic": float(frame["rank_ic_mean"].abs().mean()),
                "mean_rank_ic": float(frame["rank_ic_mean"].mean()),
                "best_feature": str(best["feature"]),
                "best_feature_rank_ic": _safe_float(best["rank_ic_mean"]),
                "mean_abs_contribution": _safe_float(shap_abs),
                "contribution_share": _safe_float(shap_abs / total_abs) if total_abs and math.isfinite(total_abs) else None,
            }
        )
    return pd.DataFrame(rows).sort_values(["contribution_share", "mean_abs_rank_ic"], ascending=[False, False]).reset_index(drop=True)


def _correlation_outputs(oos_df: pd.DataFrame, feature_columns: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr_df = oos_df.loc[:, list(feature_columns)].replace([np.inf, -np.inf], np.nan).fillna(0.0).corr(method="spearman")
    rows: List[Dict[str, Any]] = []
    cols = list(corr_df.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            corr_val = float(corr_df.loc[left, right])
            rows.append({"feature_left": left, "feature_right": right, "spearman_corr": corr_val, "abs_spearman_corr": abs(corr_val)})
    redundant_df = pd.DataFrame(rows).sort_values("abs_spearman_corr", ascending=False).reset_index(drop=True)
    return corr_df, redundant_df


def _global_contrib_summary(
    contrib_df: pd.DataFrame,
    feature_columns: Sequence[str],
    meta_map: Mapping[str, Dict[str, Any]],
    pool_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    bin_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    if contrib_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    bin_map = {str(row["feature"]): row for row in bin_summary_df.to_dict("records")}
    for feature in feature_columns:
        if feature not in contrib_df.columns:
            continue
        values = contrib_df[feature].to_numpy(dtype=float)
        band = _preferred_band_summary(feature, pool_df=pool_df, selected_df=selected_df)
        bin_row = bin_map.get(feature, {})
        meta = meta_map.get(feature, {})
        rows.append(
            {
                "feature": feature,
                "category": meta.get("category", "other"),
                "broad_group": _broad_group(meta.get("category", "other")),
                "mean_abs_contribution": float(np.mean(np.abs(values))),
                "mean_contribution": float(np.mean(values)),
                "positive_contrib_share": float(np.mean(values > 0)),
                "preferred_direction": band["preferred_direction"],
                "selected_q25": band["selected_q25"],
                "selected_median": band["selected_median"],
                "selected_q75": band["selected_q75"],
                "pool_median": band["pool_median"],
                "band_effect_size": band["effect_size"],
                "best_bin_feature_mean": bin_row.get("best_bin_feature_mean"),
                "best_bin_return": bin_row.get("best_bin_return"),
                "shape_hint": bin_row.get("shape_hint"),
                "spearman_bin_to_return": bin_row.get("spearman_bin_to_return"),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_contribution", ascending=False).reset_index(drop=True)


def _local_examples(
    *,
    models_by_year: Mapping[int, Any],
    scored_frames_by_year: Mapping[int, pd.DataFrame],
    model_family: str,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    example_rows: List[pd.DataFrame] = []
    for test_year, year_df in scored_frames_by_year.items():
        q5 = year_df.sort_values("score", ascending=False).head(3).copy()
        q5["case_type"] = "high_score"
        false_pos = year_df[year_df["future_return_5d"] < 0].sort_values("score", ascending=False).head(2).copy()
        false_pos["case_type"] = "high_score_false_positive"
        examples = pd.concat([q5, false_pos], ignore_index=True).drop_duplicates(subset=["symbol", "signal_date"])
        if not examples.empty:
            example_rows.append(examples.assign(test_year=int(test_year)))
    if not example_rows:
        return pd.DataFrame()
    examples_df = pd.concat(example_rows, ignore_index=True)

    rendered_rows: List[Dict[str, Any]] = []
    for test_year, year_examples in examples_df.groupby("test_year", sort=True):
        contrib_df, _ = _score_contributions_for_rows(
            model=models_by_year[int(test_year)],
            model_family=model_family,
            feature_columns=feature_columns,
            frame=year_examples,
        )
        if contrib_df is None or contrib_df.empty:
            continue
        for idx, (_, row) in enumerate(year_examples.reset_index(drop=True).iterrows()):
            contrib_row = contrib_df.iloc[idx]
            pos_terms = contrib_row[feature_columns].sort_values(ascending=False).head(3)
            neg_terms = contrib_row[feature_columns].sort_values(ascending=True).head(3)
            rendered_rows.append(
                {
                    "test_year": int(test_year),
                    "symbol": str(row["symbol"]),
                    "signal_date": pd.Timestamp(row["signal_date"]).strftime("%Y-%m-%d"),
                    "score": float(row["score"]),
                    "bucket": int(row["bucket"]),
                    "future_return_5d": float(row["future_return_5d"]),
                    "case_type": str(row["case_type"]),
                    "top_positive_contributors": "; ".join(f"{k}={v:.4f}" for k, v in pos_terms.items()),
                    "top_negative_contributors": "; ".join(f"{k}={v:.4f}" for k, v in neg_terms.items()),
                }
            )
    return pd.DataFrame(rendered_rows)


def _pdp_curves(
    *,
    models_by_year: Mapping[int, Any],
    scored_frames_by_year: Mapping[int, pd.DataFrame],
    model_family: str,
    feature_columns: Sequence[str],
    top_features: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for feature in top_features:
        all_values = pd.concat([frame[feature] for frame in scored_frames_by_year.values()], ignore_index=True).dropna()
        if all_values.nunique() < 6:
            continue
        quantiles = np.linspace(0.10, 0.90, PDP_GRID_POINTS)
        grid_values = np.quantile(all_values.to_numpy(dtype=float), quantiles)
        year_preds: List[pd.DataFrame] = []
        for test_year, year_df in scored_frames_by_year.items():
            sample_df = _sample_frame(year_df, MAX_PDP_SAMPLE_ROWS_PER_YEAR)
            baseline = sample_df.copy()
            for grid_index, grid_value in enumerate(grid_values):
                edited = baseline.copy()
                edited[feature] = float(grid_value)
                pred = _predict_scores(models_by_year[int(test_year)], model_family, feature_columns, edited)
                year_preds.append(
                    pd.DataFrame(
                        {
                            "feature": feature,
                            "test_year": int(test_year),
                            "grid_index": int(grid_index),
                            "feature_value": float(grid_value),
                            "mean_prediction": float(np.mean(pred)),
                        },
                        index=[0],
                    )
                )
        if year_preds:
            merged = pd.concat(year_preds, ignore_index=True)
            agg = (
                merged.groupby(["feature", "grid_index", "feature_value"], as_index=False)["mean_prediction"]
                .agg(["mean", "std"])
                .reset_index()
                .rename(columns={"mean": "mean_prediction", "std": "std_prediction"})
            )
            rows.extend(agg.to_dict("records"))
    return pd.DataFrame(rows)


def _combo_effects(
    *,
    oos_df: pd.DataFrame,
    global_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    top_features = global_summary_df["feature"].head(4).tolist()
    if len(top_features) < 2:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for left, right in combinations(top_features[:4], 2):
        left_dir = str(global_summary_df.loc[global_summary_df["feature"] == left, "preferred_direction"].iloc[0])
        right_dir = str(global_summary_df.loc[global_summary_df["feature"] == right, "preferred_direction"].iloc[0])
        left_low, left_high = np.nanquantile(oos_df[left].to_numpy(dtype=float), [0.33, 0.67])
        right_low, right_high = np.nanquantile(oos_df[right].to_numpy(dtype=float), [0.33, 0.67])
        left_fav = oos_df[left] >= left_high if left_dir == "higher" else oos_df[left] <= left_low
        right_fav = oos_df[right] >= right_high if right_dir == "higher" else oos_df[right] <= right_low
        left_bad = oos_df[left] <= left_low if left_dir == "higher" else oos_df[left] >= left_high
        right_bad = oos_df[right] <= right_low if right_dir == "higher" else oos_df[right] >= right_high
        fav_mask = left_fav & right_fav
        bad_mask = left_bad & right_bad
        mixed_mask = (left_fav & right_bad) | (left_bad & right_fav)
        if fav_mask.sum() < 25 or bad_mask.sum() < 25:
            continue
        rows.append(
            {
                "feature_left": left,
                "feature_right": right,
                "left_direction": left_dir,
                "right_direction": right_dir,
                "favored_count": int(fav_mask.sum()),
                "opposite_count": int(bad_mask.sum()),
                "mixed_count": int(mixed_mask.sum()),
                "favored_return": float(oos_df.loc[fav_mask, "future_return_5d"].mean()),
                "opposite_return": float(oos_df.loc[bad_mask, "future_return_5d"].mean()),
                "mixed_return": float(oos_df.loc[mixed_mask, "future_return_5d"].mean()) if mixed_mask.sum() else np.nan,
                "lift_favored_vs_opposite": float(oos_df.loc[fav_mask, "future_return_5d"].mean() - oos_df.loc[bad_mask, "future_return_5d"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("lift_favored_vs_opposite", ascending=False).reset_index(drop=True)


def _format_float(value: Any, digits: int = 4, none_text: str = "na") -> str:
    value = _safe_float(value)
    if value is None:
        return none_text
    return f"{value:.{digits}f}"


def _band_text(row: Mapping[str, Any]) -> str:
    q25 = _safe_float(row.get("selected_q25"))
    q75 = _safe_float(row.get("selected_q75"))
    if q25 is None or q75 is None:
        return "any"
    return f"[{q25:.4f},{q75:.4f}]"


def _strength_label(score: Any, *, strong: float, medium: float) -> str:
    value = _safe_float(score)
    if value is None:
        return "weak"
    value = abs(value)
    if value >= strong:
        return "strong"
    if value >= medium:
        return "medium"
    return "weak"


def _parse_contributor_features(text: Any) -> List[str]:
    features: List[str] = []
    for raw_part in str(text or "").split(";"):
        part = raw_part.strip()
        if not part or "=" not in part:
            continue
        feature = part.split("=", 1)[0].strip()
        if feature:
            features.append(feature)
    return features


def _pdp_effect_summaries(pdp_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if pdp_df.empty:
        return {}
    summaries: Dict[str, Dict[str, Any]] = {}
    for feature, frame in pdp_df.groupby("feature", sort=False):
        frame = frame.sort_values("grid_index").copy()
        start_val = _safe_float(frame["mean_prediction"].iloc[0])
        end_val = _safe_float(frame["mean_prediction"].iloc[-1])
        pred_range = _safe_float(frame["mean_prediction"].max() - frame["mean_prediction"].min())
        pred_delta = _safe_float((end_val or 0.0) - (start_val or 0.0))
        if pred_range is None or pred_range < 1e-6:
            effect_kind = "flat"
        elif pred_delta is not None and abs(pred_delta) / pred_range >= 0.75:
            effect_kind = "rising" if pred_delta > 0 else "falling"
        else:
            effect_kind = "curved_or_threshold"
        summaries[str(feature)] = {
            "feature": str(feature),
            "prediction_delta_top_bottom": pred_delta,
            "prediction_range": pred_range,
            "effect_kind": effect_kind,
        }
    return summaries


def _derive_branch_report_v2(
    *,
    global_summary_df: pd.DataFrame,
    group_df: pd.DataFrame,
    redundant_df: pd.DataFrame,
    combo_df: pd.DataFrame,
    local_df: pd.DataFrame,
    pdp_df: pd.DataFrame,
) -> Dict[str, Any]:
    global_records = global_summary_df.to_dict("records")
    global_map = {str(row["feature"]): row for row in global_records if row.get("feature")}
    pdp_map = _pdp_effect_summaries(pdp_df)

    top_abs_contrib = _safe_float(global_summary_df["mean_abs_contribution"].max()) if not global_summary_df.empty else None
    top_abs_contrib = top_abs_contrib or 1.0

    soft_rules: List[Dict[str, Any]] = []
    for idx, row in enumerate(global_records[:8], start=1):
        contrib_ratio = (_safe_float(row.get("mean_abs_contribution")) or 0.0) / top_abs_contrib
        band_effect = abs(_safe_float(row.get("band_effect_size")) or 0.0)
        strength_score = max(contrib_ratio, band_effect / 1.25)
        strength = "strong" if strength_score >= 0.75 else "medium" if strength_score >= 0.40 else "weak"
        pdp_effect = pdp_map.get(str(row.get("feature")), {})
        if row.get("shape_hint") == "nonlinear_or_flat" or pdp_effect.get("effect_kind") == "curved_or_threshold":
            usage_note = "use as fuzzy preference zone, not a hard cutoff"
        elif row.get("preferred_direction") == "higher":
            usage_note = "more is generally better inside the teacher-selected zone"
        else:
            usage_note = "less is generally better inside the teacher-selected zone"
        soft_rules.append(
            {
                "rule_id": f"SOFT{idx:02d}",
                "feature": str(row.get("feature")),
                "direction": str(row.get("preferred_direction") or "mixed"),
                "band": _band_text(row),
                "strength": strength,
                "shape_hint": str(row.get("shape_hint") or "unknown"),
                "pdp_effect_kind": str(pdp_effect.get("effect_kind") or "unknown"),
                "usage_note": usage_note,
                "evidence_mean_abs_contribution": _safe_float(row.get("mean_abs_contribution")),
                "evidence_band_effect_size": _safe_float(row.get("band_effect_size")),
                "evidence_best_bin_return": _safe_float(row.get("best_bin_return")),
            }
        )

    clear_combo_df = combo_df[
        (combo_df["favored_return"] > combo_df["opposite_return"])
        & (combo_df["favored_return"] > combo_df["mixed_return"])
    ].copy()
    ambiguous_combo_df = combo_df[
        ~(
            (combo_df["favored_return"] > combo_df["opposite_return"])
            & (combo_df["favored_return"] > combo_df["mixed_return"])
        )
    ].copy()

    branch_cards: List[Dict[str, Any]] = []
    branch_iter_df = clear_combo_df if not clear_combo_df.empty else combo_df.head(2).copy()
    for idx, (_, row) in enumerate(branch_iter_df.head(4).iterrows(), start=1):
        left = str(row["feature_left"])
        right = str(row["feature_right"])
        left_row = global_map.get(left, {})
        right_row = global_map.get(right, {})
        support_rules = [
            {
                "feature": rule["feature"],
                "direction": rule["direction"],
                "band": rule["band"],
                "strength": rule["strength"],
            }
            for rule in soft_rules
            if rule["feature"] not in {left, right}
        ][:2]
        favored_return = _safe_float(row.get("favored_return"))
        mixed_return = _safe_float(row.get("mixed_return"))
        opposite_return = _safe_float(row.get("opposite_return"))
        lift = _safe_float(row.get("lift_favored_vs_opposite"))
        branch_cards.append(
            {
                "branch_id": f"BRANCH{idx:02d}",
                "strength": _strength_label(lift, strong=0.030, medium=0.015),
                "anchor_pair": {
                    "left_feature": left,
                    "left_direction": str(row.get("left_direction") or left_row.get("preferred_direction") or "mixed"),
                    "left_band": _band_text(left_row),
                    "right_feature": right,
                    "right_direction": str(row.get("right_direction") or right_row.get("preferred_direction") or "mixed"),
                    "right_band": _band_text(right_row),
                },
                "supporting_features": support_rules,
                "branch_logic": (
                    f"When {left} leans {row.get('left_direction')} around {_band_text(left_row)} "
                    f"and {right} leans {row.get('right_direction')} around {_band_text(right_row)}, "
                    "the teacher tends to upgrade the setup materially."
                ),
                "partial_alignment_note": (
                    f"If only one side aligns, edge decays toward mixed_return={_format_float(mixed_return)} "
                    f"instead of favored_return={_format_float(favored_return)}."
                ),
                "hard_veto_note": (
                    f"If both anchors flip away together, treat it as a near-veto zone because "
                    f"opposite_return only reaches {_format_float(opposite_return)}."
                )
                if opposite_return is not None and favored_return is not None and opposite_return < 0 < favored_return
                else None,
                "evidence": {
                    "favored_count": int(row.get("favored_count") or 0),
                    "mixed_count": int(row.get("mixed_count") or 0),
                    "opposite_count": int(row.get("opposite_count") or 0),
                    "favored_return": favored_return,
                    "mixed_return": mixed_return,
                    "opposite_return": opposite_return,
                    "lift_favored_vs_opposite": lift,
                },
            }
        )

    ambiguous_combo_contexts: List[Dict[str, Any]] = []
    for idx, (_, row) in enumerate(ambiguous_combo_df.head(4).iterrows(), start=1):
        ambiguous_combo_contexts.append(
            {
                "context_id": f"AMBIG{idx:02d}",
                "feature_left": str(row["feature_left"]),
                "feature_right": str(row["feature_right"]),
                "favored_return": _safe_float(row.get("favored_return")),
                "mixed_return": _safe_float(row.get("mixed_return")),
                "opposite_return": _safe_float(row.get("opposite_return")),
                "lift_favored_vs_opposite": _safe_float(row.get("lift_favored_vs_opposite")),
                "guidance": (
                    "pair has some edge versus the opposite side, but mixed alignment is as good as or better than the fully favored side; "
                    "treat it as a fuzzy context feature, not a primary branch gate"
                ),
            }
        )

    hard_veto_rules: List[Dict[str, Any]] = []
    for idx, branch in enumerate(branch_cards, start=1):
        evidence = branch.get("evidence", {})
        opposite_return = _safe_float(evidence.get("opposite_return"))
        favored_return = _safe_float(evidence.get("favored_return"))
        if opposite_return is None or favored_return is None or not (opposite_return < 0 < favored_return):
            continue
        anchor = branch["anchor_pair"]
        hard_veto_rules.append(
            {
                "rule_id": f"VETO{idx:02d}",
                "strength": branch["strength"],
                "trigger": (
                    f"{anchor['left_feature']} flips away from {anchor['left_direction']} and "
                    f"{anchor['right_feature']} flips away from {anchor['right_direction']} together"
                ),
                "reason": (
                    f"teacher edge collapses from favored_return={_format_float(favored_return)} "
                    f"to opposite_return={_format_float(opposite_return)}"
                ),
            }
        )

    trap_neg_counter: Counter[str] = Counter()
    if not local_df.empty:
        trap_df = local_df[local_df["case_type"].astype(str).str.contains("false_positive", na=False)].copy()
        for text in trap_df.get("top_negative_contributors", pd.Series(dtype=object)).tolist():
            trap_neg_counter.update(_parse_contributor_features(text))
        if trap_neg_counter:
            hard_veto_rules.append(
                {
                    "rule_id": f"VETO{len(hard_veto_rules) + 1:02d}",
                    "strength": "medium",
                    "trigger": "false-positive drag stack dominates the local explanation",
                    "reason": "recurring trap-side negative contributors: " + ", ".join([feat for feat, _ in trap_neg_counter.most_common(3)]),
                }
            )

    meta_rules: List[Dict[str, Any]] = []
    if branch_cards:
        first_branch = branch_cards[0]
        evidence = first_branch["evidence"]
        meta_rules.append(
            {
                "rule_id": "META01",
                "theme": "anchor_combo_first",
                "guidance": (
                    f"Start with the top anchor pair before debating weaker factors. "
                    f"Partial alignment only gives mixed_return={_format_float(evidence.get('mixed_return'))} "
                    f"versus favored_return={_format_float(evidence.get('favored_return'))}."
                ),
            }
        )
    if not redundant_df.empty:
        top_redundant = redundant_df.iloc[0]
        meta_rules.append(
            {
                "rule_id": "META02",
                "theme": "avoid_double_counting",
                "guidance": (
                    f"Do not double-count {top_redundant['feature_left']} and {top_redundant['feature_right']}; "
                    f"their spearman correlation is {_format_float(top_redundant['spearman_corr'])}."
                ),
            }
        )
    if not group_df.empty:
        top_group = group_df.iloc[0]
        contribution_share = _safe_float(top_group.get("contribution_share"))
        contribution_share_text = f"{contribution_share:.2%}" if contribution_share is not None else "na"
        meta_rules.append(
            {
                "rule_id": "META03",
                "theme": "regime_priority",
                "guidance": (
                    f"Diagnose the setup through the {top_group['broad_group']} group first; "
                    f"it contributes {contribution_share_text} of total global importance."
                ),
            }
        )
    if ambiguous_combo_contexts:
        top_ambiguous = ambiguous_combo_contexts[0]
        meta_rules.append(
            {
                "rule_id": "META05",
                "theme": "fuzzy_not_binary",
                "guidance": (
                    f"{top_ambiguous['feature_left']} + {top_ambiguous['feature_right']} is not a clean branch gate: "
                    f"mixed_return={_format_float(top_ambiguous['mixed_return'])} is not worse than favored_return={_format_float(top_ambiguous['favored_return'])}. "
                    "Use this kind of interaction as contextual ranking texture, not as a rigid if/else split."
                ),
            }
        )
    if not local_df.empty:
        meta_rules.append(
            {
                "rule_id": "META04",
                "theme": "high_score_not_enough",
                "guidance": "A high raw model score is not sufficient on its own; always inspect whether the local contributor stack looks like a winner archetype or a false-positive trap.",
            }
        )

    positive_cases = local_df[local_df["future_return_5d"] > 0].copy() if not local_df.empty else pd.DataFrame()
    trap_cases = local_df[local_df["case_type"].astype(str).str.contains("false_positive", na=False)].copy() if not local_df.empty else pd.DataFrame()
    contrast_pairs: List[Dict[str, Any]] = []
    if not positive_cases.empty and not trap_cases.empty:
        positive_rows = positive_cases.to_dict("records")
        for idx, trap in enumerate(trap_cases.head(3).to_dict("records"), start=1):
            trap_pos = _parse_contributor_features(trap.get("top_positive_contributors"))
            trap_neg = _parse_contributor_features(trap.get("top_negative_contributors"))
            best_match: Optional[Dict[str, Any]] = None
            best_overlap = -1
            for win in positive_rows:
                overlap = len(set(trap_pos) & set(_parse_contributor_features(win.get("top_positive_contributors"))))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = win
            if best_match is None:
                continue
            win_pos = _parse_contributor_features(best_match.get("top_positive_contributors"))
            shared = list(sorted(set(trap_pos) & set(win_pos)))[:3]
            winner_extra = [feat for feat in win_pos if feat not in shared][:3]
            contrast_pairs.append(
                {
                    "pair_id": f"PAIR{idx:02d}",
                    "trap_case": f"{trap['signal_date']} {trap['symbol']} ret5d={_format_float(trap['future_return_5d'])}",
                    "winner_case": f"{best_match['signal_date']} {best_match['symbol']} ret5d={_format_float(best_match['future_return_5d'])}",
                    "shared_positive_features": shared,
                    "winner_extra_features": winner_extra,
                    "trap_negative_features": trap_neg[:3],
                    "lesson": "similar headline setup, but the winner keeps stronger reinforcing contributors while the trap shows drag-side negatives",
                }
            )

    archetypes: List[Dict[str, Any]] = []
    winner_pos_counter: Counter[str] = Counter()
    winner_neg_counter: Counter[str] = Counter()
    trap_pos_counter: Counter[str] = Counter()
    for text in positive_cases.get("top_positive_contributors", pd.Series(dtype=object)).tolist():
        winner_pos_counter.update(_parse_contributor_features(text))
    for text in positive_cases.get("top_negative_contributors", pd.Series(dtype=object)).tolist():
        winner_neg_counter.update(_parse_contributor_features(text))
    for text in trap_cases.get("top_positive_contributors", pd.Series(dtype=object)).tolist():
        trap_pos_counter.update(_parse_contributor_features(text))
    if winner_pos_counter:
        archetypes.append(
            {
                "archetype_id": "ARCH01",
                "name": "winner_core_stack",
                "type": "positive",
                "core_positive_features": [feat for feat, _ in winner_pos_counter.most_common(4)],
                "common_negative_offsets": [feat for feat, _ in winner_neg_counter.most_common(3)],
                "description": "features that recur most often inside winning high-score local explanations",
            }
        )
    if trap_pos_counter or trap_neg_counter:
        archetypes.append(
            {
                "archetype_id": "ARCH02",
                "name": "false_positive_trap",
                "type": "negative",
                "trap_positive_features": [feat for feat, _ in trap_pos_counter.most_common(4)],
                "trap_negative_features": [feat for feat, _ in trap_neg_counter.most_common(4)],
                "description": "features that keep appearing when the teacher scores high but realized return disappoints",
            }
        )

    return {
        "report_schema_version": "branch_oriented_v2",
        "branch_cards": branch_cards,
        "soft_rules": soft_rules,
        "hard_veto_rules": hard_veto_rules,
        "meta_rules": meta_rules,
        "ambiguous_combo_contexts": ambiguous_combo_contexts,
        "false_positive_contrast_pairs": contrast_pairs,
        "archetypes": archetypes,
        "pdp_effect_summaries": list(pdp_map.values())[:8],
    }


def _json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _factor_report_markdown(
    *,
    spec_title: str,
    oos_rows: int,
    method: str,
    global_summary_df: pd.DataFrame,
    group_df: pd.DataFrame,
    redundant_df: pd.DataFrame,
    combo_df: pd.DataFrame,
    local_df: pd.DataFrame,
    branch_v2: Mapping[str, Any],
) -> str:
    lines = [
        f"# {spec_title} Factor Analysis",
        "",
        f"- out_of_sample_rows: `{oos_rows}`",
        f"- local_explainability_method: `{method}`",
        f"- report_schema_version: `{branch_v2.get('report_schema_version', 'legacy')}`",
        "",
        "## Report V2 Branch Cards",
        "",
    ]
    if not (branch_v2.get("branch_cards") or []):
        lines.append("- unavailable")
    else:
        for branch in branch_v2.get("branch_cards") or []:
            anchor = branch.get("anchor_pair", {})
            evidence = branch.get("evidence", {})
            lines.append(
                f"- {branch['branch_id']} [{branch['strength']}]: "
                f"{anchor.get('left_feature')}({anchor.get('left_direction')} {anchor.get('left_band')}) + "
                f"{anchor.get('right_feature')}({anchor.get('right_direction')} {anchor.get('right_band')}) | "
                f"favored={_format_float(evidence.get('favored_return'))}, mixed={_format_float(evidence.get('mixed_return'))}, "
                f"opposite={_format_float(evidence.get('opposite_return'))}, lift={_format_float(evidence.get('lift_favored_vs_opposite'))}"
            )
            lines.append(f"  logic: {branch['branch_logic']}")
            lines.append(f"  partial: {branch['partial_alignment_note']}")
            if branch.get("hard_veto_note"):
                lines.append(f"  veto: {branch['hard_veto_note']}")
            support_feats = branch.get("supporting_features") or []
            if support_feats:
                lines.append(
                    "  support: "
                    + "; ".join(
                        f"{item['feature']} {item['direction']} {item['band']} [{item['strength']}]"
                        for item in support_feats
                    )
                )
    lines.extend(["", "## Report V2 Ambiguous Combo Contexts", ""])
    if not (branch_v2.get("ambiguous_combo_contexts") or []):
        lines.append("- unavailable")
    else:
        for item in branch_v2.get("ambiguous_combo_contexts") or []:
            lines.append(
                f"- {item['context_id']}: {item['feature_left']} + {item['feature_right']} | "
                f"favored={_format_float(item['favored_return'])}, mixed={_format_float(item['mixed_return'])}, "
                f"opposite={_format_float(item['opposite_return'])}, lift={_format_float(item['lift_favored_vs_opposite'])}"
            )
            lines.append(f"  guidance: {item['guidance']}")
    lines.extend(["", "## Report V2 Soft Preferences", ""])
    if not (branch_v2.get("soft_rules") or []):
        lines.append("- unavailable")
    else:
        for rule in branch_v2.get("soft_rules") or []:
            lines.append(
                f"- {rule['rule_id']} [{rule['strength']}]: {rule['feature']} {rule['direction']} {rule['band']} | "
                f"shape={rule['shape_hint']} | pdp={rule['pdp_effect_kind']} | note={rule['usage_note']}"
            )
    lines.extend(["", "## Report V2 Hard Veto Rules", ""])
    if not (branch_v2.get("hard_veto_rules") or []):
        lines.append("- unavailable")
    else:
        for rule in branch_v2.get("hard_veto_rules") or []:
            lines.append(f"- {rule['rule_id']} [{rule['strength']}]: {rule['trigger']} | {rule['reason']}")
    lines.extend(["", "## Report V2 Meta Rules", ""])
    if not (branch_v2.get("meta_rules") or []):
        lines.append("- unavailable")
    else:
        for rule in branch_v2.get("meta_rules") or []:
            lines.append(f"- {rule['rule_id']} {rule['theme']}: {rule['guidance']}")
    lines.extend(["", "## Report V2 False-Positive Contrast Pairs", ""])
    if not (branch_v2.get("false_positive_contrast_pairs") or []):
        lines.append("- unavailable")
    else:
        for pair in branch_v2.get("false_positive_contrast_pairs") or []:
            lines.append(
                f"- {pair['pair_id']}: trap={pair['trap_case']} vs winner={pair['winner_case']} | "
                f"shared={', '.join(pair['shared_positive_features']) or 'none'} | "
                f"winner_extra={', '.join(pair['winner_extra_features']) or 'none'} | "
                f"trap_drag={', '.join(pair['trap_negative_features']) or 'none'}"
            )
            lines.append(f"  lesson: {pair['lesson']}")
    lines.extend(["", "## Report V2 Archetypes", ""])
    if not (branch_v2.get("archetypes") or []):
        lines.append("- unavailable")
    else:
        for item in branch_v2.get("archetypes") or []:
            lines.append(f"- {item['archetype_id']} {item['name']} ({item['type']}): {item['description']}")
            for key in ("core_positive_features", "common_negative_offsets", "trap_positive_features", "trap_negative_features"):
                values = item.get(key) or []
                if values:
                    lines.append(f"  {key}: {', '.join(values)}")
    lines.extend(["", "## Global SHAP-Style Feature Importance", ""])
    top_global = global_summary_df.head(12)
    if top_global.empty:
        lines.append("- unavailable")
    else:
        for _, row in top_global.iterrows():
            lines.append(
                f"- {row['feature']}: mean|contrib|={row['mean_abs_contribution']:.6f}, "
                f"dir={row['preferred_direction']}, band=[{row['selected_q25'] if pd.notna(row['selected_q25']) else 'na'}, "
                f"{row['selected_q75'] if pd.notna(row['selected_q75']) else 'na'}], shape={row['shape_hint']}"
            )
    lines.extend(["", "## Feature Group Effectiveness", ""])
    if group_df.empty:
        lines.append("- unavailable")
    else:
        for _, row in group_df.iterrows():
            if pd.notna(row["contribution_share"]):
                text = (
                    f"- {row['broad_group']}: features={row['feature_count']}, "
                    f"mean_abs_rank_ic={row['mean_abs_rank_ic']:.6f}, "
                    f"best={row['best_feature']} ({row['best_feature_rank_ic']:.6f}), "
                    f"contribution_share={row['contribution_share']:.2%}"
                )
            else:
                text = (
                    f"- {row['broad_group']}: features={row['feature_count']}, "
                    f"mean_abs_rank_ic={row['mean_abs_rank_ic']:.6f}, "
                    f"best={row['best_feature']} ({row['best_feature_rank_ic']:.6f})"
                )
            lines.append(text)
    lines.extend(["", "## Redundant Feature Pairs", ""])
    if redundant_df.empty:
        lines.append("- unavailable")
    else:
        for _, row in redundant_df.head(10).iterrows():
            lines.append(f"- {row['feature_left']} vs {row['feature_right']}: spearman={row['spearman_corr']:.4f}")
    lines.extend(["", "## Feature Combination Effects", ""])
    if combo_df.empty:
        lines.append("- unavailable")
    else:
        for _, row in combo_df.head(8).iterrows():
            lines.append(
                f"- {row['feature_left']}({row['left_direction']}) + {row['feature_right']}({row['right_direction']}): "
                f"favored_return={row['favored_return']:.4f}, opposite_return={row['opposite_return']:.4f}, "
                f"lift={row['lift_favored_vs_opposite']:.4f}"
            )
    lines.extend(["", "## Local Cases", ""])
    if local_df.empty:
        lines.append("- unavailable")
    else:
        for _, row in local_df.head(10).iterrows():
            lines.append(
                f"- {row['signal_date']} {row['symbol']} {row['case_type']}: score={row['score']:.4f}, "
                f"ret5d={row['future_return_5d']:.4f}, +[{row['top_positive_contributors']}], -[{row['top_negative_contributors']}]"
            )
    return "\n".join(lines) + "\n"


def run_teacher_factor_analysis(
    *,
    spec_title: str,
    model_family: str,
    feature_columns: Sequence[str],
    models_by_year: Mapping[int, Any],
    scored_frames_by_year: Mapping[int, pd.DataFrame],
    report_dir: Path,
    artifact_dir: Path,
) -> FactorAnalysisResult:
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = list(feature_columns)
    meta_map = _feature_meta_map()
    oos_df = pd.concat([frame.copy() for _, frame in sorted(scored_frames_by_year.items())], ignore_index=True)
    pool_df = oos_df.copy()
    selected_df = oos_df[oos_df["bucket"] == 5].copy()
    if selected_df.empty:
        selected_df = oos_df[oos_df["bucket"] >= 4].copy()

    contrib_chunks: List[pd.DataFrame] = []
    contrib_method = "unavailable"
    for test_year, frame in sorted(scored_frames_by_year.items()):
        sampled = _sample_frame(frame, MAX_CONTRIB_ROWS_PER_YEAR)
        contrib_df, method = _score_contributions_for_rows(
            model=models_by_year[int(test_year)],
            model_family=model_family,
            feature_columns=feature_columns,
            frame=sampled,
        )
        if contrib_df is None or contrib_df.empty:
            continue
        contrib_method = method
        contrib_df = contrib_df.assign(
            test_year=int(test_year),
            symbol=sampled["symbol"].astype(str).to_numpy(),
            signal_date=pd.to_datetime(sampled["signal_date"]).dt.strftime("%Y-%m-%d").to_numpy(),
            score=sampled["score"].to_numpy(dtype=float),
            bucket=sampled["bucket"].to_numpy(dtype=int),
            future_return_5d=sampled["future_return_5d"].to_numpy(dtype=float),
        )
        contrib_chunks.append(contrib_df)
    contrib_full_df = pd.concat(contrib_chunks, ignore_index=True) if contrib_chunks else pd.DataFrame()

    bin_detail_df, bin_summary_df = _single_feature_bins(oos_df, feature_columns)
    global_summary_df = _global_contrib_summary(
        contrib_df=contrib_full_df if not contrib_full_df.empty else pd.DataFrame(columns=feature_columns),
        feature_columns=feature_columns,
        meta_map=meta_map,
        pool_df=pool_df,
        selected_df=selected_df,
        bin_summary_df=bin_summary_df,
    ) if not contrib_full_df.empty else pd.DataFrame()
    if global_summary_df.empty:
        fallback_rows: List[Dict[str, Any]] = []
        importance_seed = bin_summary_df.set_index("feature") if not bin_summary_df.empty else pd.DataFrame()
        for feature in feature_columns:
            meta = meta_map.get(feature, {})
            band = _preferred_band_summary(feature, pool_df=pool_df, selected_df=selected_df)
            fallback_rows.append(
                {
                    "feature": feature,
                    "category": meta.get("category", "other"),
                    "broad_group": _broad_group(meta.get("category", "other")),
                    "mean_abs_contribution": abs(band["effect_size"] or 0.0),
                    "mean_contribution": 0.0,
                    "positive_contrib_share": np.nan,
                    "preferred_direction": band["preferred_direction"],
                    "selected_q25": band["selected_q25"],
                    "selected_median": band["selected_median"],
                    "selected_q75": band["selected_q75"],
                    "pool_median": band["pool_median"],
                    "band_effect_size": band["effect_size"],
                    "best_bin_feature_mean": importance_seed.loc[feature, "best_bin_feature_mean"] if feature in importance_seed.index else np.nan,
                    "best_bin_return": importance_seed.loc[feature, "best_bin_return"] if feature in importance_seed.index else np.nan,
                    "shape_hint": importance_seed.loc[feature, "shape_hint"] if feature in importance_seed.index else "unknown",
                    "spearman_bin_to_return": importance_seed.loc[feature, "spearman_bin_to_return"] if feature in importance_seed.index else np.nan,
                }
            )
        global_summary_df = pd.DataFrame(fallback_rows).sort_values("mean_abs_contribution", ascending=False).reset_index(drop=True)

    group_df = _group_effectiveness(
        oos_df=oos_df,
        feature_columns=feature_columns,
        meta_map=meta_map,
        shap_global_df=global_summary_df,
    )
    corr_df, redundant_df = _correlation_outputs(oos_df, feature_columns)
    local_df = _local_examples(
        models_by_year=models_by_year,
        scored_frames_by_year=scored_frames_by_year,
        model_family=model_family,
        feature_columns=feature_columns,
    )
    top_pdp_features = global_summary_df["feature"].head(6).tolist() if not global_summary_df.empty else list(feature_columns[:6])
    pdp_df = _pdp_curves(
        models_by_year=models_by_year,
        scored_frames_by_year=scored_frames_by_year,
        model_family=model_family,
        feature_columns=feature_columns,
        top_features=top_pdp_features,
    )
    combo_df = _combo_effects(oos_df=oos_df, global_summary_df=global_summary_df)

    shap_global_path = report_dir / "shap_global_summary.csv"
    shap_local_path = report_dir / "shap_local_examples.csv"
    bin_detail_path = report_dir / "single_feature_bins.csv"
    bin_summary_path = report_dir / "single_feature_bin_summary.csv"
    group_path = report_dir / "feature_group_effectiveness.csv"
    corr_path = report_dir / "feature_correlation_matrix.csv"
    redundant_path = report_dir / "feature_redundant_pairs.csv"
    pdp_path = report_dir / "feature_pdp_curves.csv"
    combo_path = report_dir / "feature_combo_effects.csv"
    branch_summary_path = report_dir / "branch_rule_cards.json"
    analysis_sample_path = artifact_dir / "factor_analysis_sample.csv.gz"
    report_md_path = report_dir / "FACTOR_ANALYSIS_REPORT.md"
    summary_json_path = report_dir / "factor_analysis_summary.json"
    shap_plot_path = report_dir / "shap_global_summary.png"
    corr_plot_path = report_dir / "feature_correlation_heatmap.png"
    pdp_plot_path = report_dir / "feature_pdp_curves.png"

    global_summary_df.to_csv(shap_global_path, index=False)
    local_df.to_csv(shap_local_path, index=False)
    bin_detail_df.to_csv(bin_detail_path, index=False)
    bin_summary_df.to_csv(bin_summary_path, index=False)
    group_df.to_csv(group_path, index=False)
    corr_df.to_csv(corr_path, index=True)
    redundant_df.to_csv(redundant_path, index=False)
    pdp_df.to_csv(pdp_path, index=False)
    combo_df.to_csv(combo_path, index=False)
    _sample_frame(oos_df, 12000).to_csv(analysis_sample_path, index=False, compression="gzip")

    _save_bar_chart(global_summary_df.head(15), shap_plot_path, f"{spec_title} | Global SHAP-Style Importance")
    _save_corr_heatmap(corr_df, corr_plot_path, f"{spec_title} | Feature Correlation")
    _save_pdp_plot(pdp_df, pdp_plot_path, f"{spec_title} | Partial Dependence")
    branch_v2 = _derive_branch_report_v2(
        global_summary_df=global_summary_df,
        group_df=group_df,
        redundant_df=redundant_df,
        combo_df=combo_df,
        local_df=local_df,
        pdp_df=pdp_df,
    )
    _json_dump(branch_summary_path, branch_v2)

    report_md_path.write_text(
        _factor_report_markdown(
            spec_title=spec_title,
            oos_rows=int(len(oos_df)),
            method=contrib_method,
            global_summary_df=global_summary_df,
            group_df=group_df,
            redundant_df=redundant_df,
            combo_df=combo_df,
            local_df=local_df,
            branch_v2=branch_v2,
        ),
        encoding="utf-8",
    )

    top_global_records = global_summary_df.head(10).to_dict("records")
    top_group_records = group_df.head(6).to_dict("records")
    top_redundant_records = redundant_df.head(10).to_dict("records")
    top_combo_records = combo_df.head(8).to_dict("records")
    pdp_summaries: List[Dict[str, Any]] = []
    if not pdp_df.empty:
        for feature, frame in pdp_df.groupby("feature", sort=False):
            frame = frame.sort_values("grid_index")
            pdp_summaries.append(
                {
                    "feature": feature,
                    "prediction_delta_top_bottom": float(frame["mean_prediction"].iloc[-1] - frame["mean_prediction"].iloc[0]),
                    "prediction_range": float(frame["mean_prediction"].max() - frame["mean_prediction"].min()),
                }
            )
    summary = {
        "spec_title": spec_title,
        "out_of_sample_rows": int(len(oos_df)),
        "selected_rows": int(len(selected_df)),
        "feature_count": int(len(feature_columns)),
        "local_explainability_method": contrib_method,
        "report_schema_version": branch_v2.get("report_schema_version", "legacy"),
        "top_global_features": top_global_records,
        "top_feature_groups": top_group_records,
        "top_redundant_pairs": top_redundant_records,
        "top_feature_combos": top_combo_records,
        "top_pdp_features": pdp_summaries[:8],
        "branch_rule_cards": branch_v2.get("branch_cards", []),
        "soft_rules": branch_v2.get("soft_rules", []),
        "hard_veto_rules": branch_v2.get("hard_veto_rules", []),
        "meta_rules": branch_v2.get("meta_rules", []),
        "ambiguous_combo_contexts": branch_v2.get("ambiguous_combo_contexts", []),
        "false_positive_contrast_pairs": branch_v2.get("false_positive_contrast_pairs", []),
        "archetypes": branch_v2.get("archetypes", []),
        "pdp_effect_summaries": branch_v2.get("pdp_effect_summaries", []),
        "artifact_files": {
            "factor_analysis_report_md": report_md_path.name,
            "branch_rule_cards_json": branch_summary_path.name,
            "shap_global_summary_csv": shap_global_path.name,
            "shap_local_examples_csv": shap_local_path.name,
            "single_feature_bins_csv": bin_detail_path.name,
            "single_feature_bin_summary_csv": bin_summary_path.name,
            "feature_group_effectiveness_csv": group_path.name,
            "feature_correlation_matrix_csv": corr_path.name,
            "feature_redundant_pairs_csv": redundant_path.name,
            "feature_pdp_curves_csv": pdp_path.name,
            "feature_combo_effects_csv": combo_path.name,
            "factor_analysis_sample_csv_gz": analysis_sample_path.name,
            "shap_global_plot_png": shap_plot_path.name,
            "feature_correlation_heatmap_png": corr_plot_path.name,
            "feature_pdp_plot_png": pdp_plot_path.name,
        },
    }
    _json_dump(summary_json_path, summary)

    artifact_paths = [
        str(shap_global_path),
        str(shap_local_path),
        str(bin_detail_path),
        str(bin_summary_path),
        str(group_path),
        str(corr_path),
        str(redundant_path),
        str(pdp_path),
        str(combo_path),
        str(branch_summary_path),
        str(report_md_path),
        str(summary_json_path),
        str(shap_plot_path),
        str(corr_plot_path),
        str(pdp_plot_path),
        str(analysis_sample_path),
    ]
    return FactorAnalysisResult(summary=summary, artifact_paths=artifact_paths)
