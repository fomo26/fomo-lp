"""Fairness report assembly for the FOMO26 linear-probe pipeline.

Wraps fomo_challenge_metrics to compute per-variable disparities and a
fairness score from pre-binned group labels.
"""
from __future__ import annotations

import math

from fomo_challenge_metrics import (
    compute_ovr_auroc,
    compute_ovr_f1,
    compute_max_disparity,
    compute_fairness_score,
)


def normalize_categorical_feature(value) -> str | None:
    """Normalise a categorical feature value to a canonical string.

    Strips whitespace and lower-cases the value so that equivalent
    spellings compare equal.  Returns None for missing/empty values.
    """
    if value is None:
        return None
    s = str(value).strip()
    return s.lower() if s else None


def bin_continuous_feature(
    value,
    bins: tuple[tuple[int | float, int], ...],
) -> int | None:
    """Map a numeric value to an integer bin label.

    Args:
        value: Raw value (string or numeric).
        bins: Sequence of (upper_bound_inclusive, bin_label) pairs in
              ascending order of upper_bound.  Example for a variable in
              the range 0–100 split into four equal bins::

                  FEATURE1_BINS = ((25, 0), (50, 1), (75, 2), (1000, 3))

    Returns:
        The label of the first bin whose upper bound >= value, or None if
        the value is missing, non-positive, or above all defined bins.
    """
    try:
        a = float(value)
    except (TypeError, ValueError):
        return None
    if not (a > 0):
        return None
    for upper, label in bins:
        if a <= upper:
            return label
    return None


def build_fairness_report(
    y_true: list[int],
    y_scores: list,
    groups_by_variable: dict[str, list],
    metric_fns: dict | None = None,
) -> dict:
    """Build per-variable fairness report from pre-binned group labels.

    Args:
        y_true: Integer class labels for each sample.
        y_scores: Softmax score vectors for each sample.
        groups_by_variable: Mapping of variable name to a list of
            already-binned group labels (one per sample).  None entries
            are treated as unknown and excluded from disparity calculations.
        metric_fns: Optional dict of {name: callable}.  Defaults to
            ovr_f1 and ovr_auroc from fomo_challenge_metrics.
    """
    if metric_fns is None:
        metric_fns = {"ovr_f1": compute_ovr_f1, "ovr_auroc": compute_ovr_auroc}

    overall: dict = {}
    for name, fn in metric_fns.items():
        try:
            overall[name] = float(fn(y_true, y_scores))
        except Exception as exc:
            overall[name] = f"unavailable: {type(exc).__name__}: {exc}"

    per_variable: dict = {}
    for var_name, groups in groups_by_variable.items():
        present = sorted({g for g in groups if g is not None})
        block: dict = {
            "groups": {g: sum(1 for x in groups if x == g) for g in present},
            "disparities": {},
            "per_group_metrics": {},
        }
        filt = [(t, s, g) for t, s, g in zip(y_true, y_scores, groups) if g is not None]
        yt = [x[0] for x in filt]
        ys = [x[1] for x in filt]
        gs = [x[2] for x in filt]
        for mname, mfn in metric_fns.items():
            d = compute_max_disparity(yt, ys, gs, mfn) if yt else float("nan")
            block["disparities"][mname] = None if math.isnan(d) else d
            per_group: dict = {}
            for g in present:
                yt_g = [t for t, x in zip(y_true, groups) if x == g]
                ys_g = [s for s, x in zip(y_scores, groups) if x == g]
                try:
                    per_group[g] = float(mfn(yt_g, ys_g))
                except Exception as exc:
                    per_group[g] = f"unavailable: {type(exc).__name__}"
            block["per_group_metrics"][mname] = per_group
        per_variable[var_name] = block

    fairness_score: dict = {}
    for mname, mfn in metric_fns.items():
        fs = compute_fairness_score(y_true, y_scores, groups_by_variable, mfn)
        fairness_score[mname] = {
            "score": None if math.isnan(fs["score"]) else fs["score"],
            "variables_used": fs["variables_used"],
            "per_variable_contribution": fs["per_variable_contribution"],
        }

    return {"overall": overall, "per_variable": per_variable, "fairness_score": fairness_score}
