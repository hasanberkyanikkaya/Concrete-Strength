"""Metrics, grouped error breakdowns and small table helpers."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y, p) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAE": float(mean_absolute_error(y, p)),
        "R2": float(r2_score(y, p)),
    }


def group_breakdown(groups: pd.Series, y, preds: dict, ids=None) -> pd.DataFrame:
    """RMSE / MAE / bias / MAPE per group. Bias = mean(predicted - actual).

    MAPE (mean |error| / actual, in %) shows the relative error, which matters
    for low-strength early-age concrete where a few MPa is a large fraction.
    `ids` (optional, e.g. row index when predictions from several splits are
    pooled) adds the number of distinct samples behind each group.

    R² is omitted on purpose: within a narrow group the target variance is
    small, so R² is not comparable across groups.
    """
    groups = pd.Series(groups).reset_index(drop=True)
    y = np.asarray(y, dtype=float)
    ids = None if ids is None else np.asarray(ids)
    rows = []
    for name, p in preds.items():
        p = np.asarray(p, dtype=float)
        for g in groups.cat.categories:
            m = (groups == g).to_numpy()
            if not m.any():
                continue
            err = p[m] - y[m]
            row = {"model": name, "group": g, "n": int(m.sum())}
            if ids is not None:
                row["n_unique"] = int(len(np.unique(ids[m])))
            row.update({
                "RMSE": float(np.sqrt(np.mean(err ** 2))),
                "MAE": float(np.mean(np.abs(err))),
                "bias": float(np.mean(err)),
                "MAPE": float(100 * np.mean(np.abs(err) / y[m])),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def paired_comparison(rep: pd.DataFrame, pairs, test_train_ratio: float,
                      metric: str = "RMSE") -> pd.DataFrame:
    """Paired comparison of two models over the same repeated splits.

    diff = metric_a - metric_b on each split (negative => model a is better
    for RMSE). Because the random splits overlap, the split-wise differences
    are not independent and a plain paired t-test is too optimistic. We use
    the corrected resampled t-test of Nadeau & Bengio (2003), which inflates
    the variance by (1/J + n_test/n_train), and report the Wilcoxon
    signed-rank test as a non-parametric check. p-values are Holm-adjusted
    across the comparisons.
    """
    from scipy import stats

    wide = rep.pivot(index="seed", columns="model", values=metric)
    rows = []
    for a, b in pairs:
        if a not in wide or b not in wide:
            continue
        d = (wide[a] - wide[b]).to_numpy()
        J = len(d)
        mean, sd = float(d.mean()), float(d.std(ddof=1))
        se = sd * np.sqrt(1 / J + test_train_ratio)
        t = mean / se if se > 0 else np.nan
        p_t = float(2 * stats.t.sf(abs(t), J - 1)) if se > 0 else np.nan
        half = float(stats.t.ppf(0.975, J - 1) * se)
        p_w = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        rows.append({
            "comparison": f"{a} - {b}", "model_a": a, "model_b": b, "n_splits": J,
            "mean_diff": mean, "std_diff": sd,
            "ci95_low": mean - half, "ci95_high": mean + half,
            "a_better_count": int((d < 0).sum()),
            "p_corrected_t": p_t, "p_wilcoxon": p_w,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["p_corrected_t_holm"] = holm(out["p_corrected_t"])
        out["p_wilcoxon_holm"] = holm(out["p_wilcoxon"])
    return out


def worst_cases(test_df: pd.DataFrame, train_df: pd.DataFrame, y, preds: dict,
                features, top: int = 15) -> pd.DataFrame:
    """Test samples with the largest mean absolute error across all models.

    `outside_train_range` lists every input whose value lies outside the
    training min-max range, i.e. where the models must extrapolate.
    """
    lo, hi = train_df[features].min(), train_df[features].max()
    out = test_df[["mix_id", *features, "age"]].reset_index(drop=True).copy()
    out["actual"] = y
    abs_err = pd.DataFrame({f"abs_err_{n}": np.abs(p - y) for n, p in preds.items()})
    out = pd.concat([out, abs_err], axis=1)
    out["mean_abs_err"] = abs_err.mean(axis=1)
    outside = (test_df[features].reset_index(drop=True) < lo) | (test_df[features].reset_index(drop=True) > hi)
    out["outside_train_range"] = outside.apply(lambda r: ", ".join(r.index[r]) or "-", axis=1)
    return out.sort_values("mean_abs_err", ascending=False).head(top).reset_index(drop=True)


def extrapolation_effect(test_df, train_df, y, preds: dict, features) -> pd.DataFrame:
    """Per-model RMSE for test rows inside vs. outside the training input range."""
    lo, hi = train_df[features].min(), train_df[features].max()
    X = test_df[features].reset_index(drop=True)
    outside = ((X < lo) | (X > hi)).any(axis=1).to_numpy()
    rows = []
    for name, p in preds.items():
        for flag, label in ((False, "inside"), (True, "outside")):
            m = outside == flag
            if m.any():
                rows.append({"model": name, "train_range": label, "n": int(m.sum()),
                             "RMSE": float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))})
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = [f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
