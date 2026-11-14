"""
FairLens core: synthetic data, experiments and fairness metrics.

ALL DATA IN THIS MODULE IS SYNTHETIC. It is generated from a documented
statistical process so that we know the ground truth. Results illustrate
mechanisms; they are NOT real-world research findings about any population.

Schema (every dataset):
    qualification_score   0-100   academic / qualification score
    years_experience      years   prior work experience
    technical_score       0-100   technical assessment score
    group                 "A"/"B" group identifier (protected attribute)
    qualified             0/1     actual qualification outcome (ground truth)

Ground truth: every candidate has a latent skill z ~ N(0, 1), drawn from the
SAME distribution for both groups. `qualified` depends only on z, so both
groups are, by construction, equally qualified. Any gap in model errors is
therefore created by the data pipeline or by the model / decision rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["qualification_score", "years_experience", "technical_score"]
QUAL_CUTOFF = 0.25  # qualified if (skill + noise) > cutoff  -> ~40% base rate


# --------------------------------------------------------------------------
# Data generation
# --------------------------------------------------------------------------
def _latent(n, rng):
    z = rng.normal(0, 1, n)
    qualified = ((z + rng.normal(0, 0.35, n)) > QUAL_CUTOFF).astype(int)
    return z, qualified


def make_pathway_data(n_a: int, n_b: int, seed: int) -> pd.DataFrame:
    """Experiment A population.

    Group B candidates come through a different education pathway:
    their qualification scores are graded on a stricter scale (about 12
    points lower for the same skill) and they have fewer years of formal
    experience, but the technical assessment measures skill identically.
    The SAME skill gives the SAME chance of being qualified in both groups.
    A model can only learn how B's features map to skill if it sees enough
    Group B examples: that is representation bias.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g, n in (("A", n_a), ("B", n_b)):
        if n == 0:
            continue
        z, y = _latent(n, rng)
        if g == "A":
            qual = 70 + 10 * z + rng.normal(0, 5, n)
            exp = 6.0 + 1.5 * z + rng.normal(0, 1.2, n)
        else:
            qual = 58 + 10 * z + rng.normal(0, 5, n)
            exp = 2.5 + 1.0 * z + rng.normal(0, 0.8, n)
        tech = 62 + 10 * z + rng.normal(0, 9, n)
        rows.append(pd.DataFrame({
            "qualification_score": np.clip(qual, 0, 100).round(1),
            "years_experience": np.clip(exp, 0, None).round(1),
            "technical_score": np.clip(tech, 0, 100).round(1),
            "group": g,
            "qualified": y,
        }))
    return pd.concat(rows, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def make_career_break_data(n_per_group: int, seed: int) -> pd.DataFrame:
    """Experiment B population (balanced: equal numbers, equal skill,
    equal base rate).

    Qualification and technical scores are generated identically for both
    groups. The only difference: Group B candidates have on average
    2 fewer years of recorded experience for the same skill (for example
    career breaks for caregiving). Experience is therefore a weaker and
    group-correlated signal - a proxy. Whether that difference turns into
    unequal errors depends on algorithmic choices: which features are used
    and which decision rule is applied.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in ("A", "B"):
        n = n_per_group
        z, y = _latent(n, rng)
        qual = 65 + 9 * z + rng.normal(0, 9, n)
        tech = 62 + 9 * z + rng.normal(0, 10, n)
        exp = 5.0 + 2.0 * z + rng.normal(0, 1.0, n) - (2.0 if g == "B" else 0.0)
        rows.append(pd.DataFrame({
            "qualification_score": np.clip(qual, 0, 100).round(1),
            "years_experience": np.clip(exp, 0, None).round(1),
            "technical_score": np.clip(tech, 0, 100).round(1),
            "group": g,
            "qualified": y,
        }))
    return pd.concat(rows, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def confusion_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n = tp + tn + fp + fn
    div = lambda a, b: a / b if b else float("nan")
    return {
        "n": n,
        "accuracy": div(tp + tn, n),
        "precision": div(tp, tp + fp),
        "recall_tpr": div(tp, tp + fn),
        "fpr": div(fp, fp + tn),
        "fnr": div(fn, fn + tp),
        "selection_rate": div(tp + fp, n),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


def group_report(df: pd.DataFrame, y_pred) -> pd.DataFrame:
    """Overall + per-group metrics table."""
    out = {"Overall": confusion_metrics(df["qualified"], y_pred)}
    for g in ("A", "B"):
        m = (df["group"] == g).to_numpy()
        out[f"Group {g}"] = confusion_metrics(df["qualified"].to_numpy()[m], np.asarray(y_pred)[m])
    return pd.DataFrame(out).T


def fairness_gaps(report: pd.DataFrame) -> dict:
    a, b = report.loc["Group A"], report.loc["Group B"]
    return {
        "accuracy_gap": float(a.accuracy - b.accuracy),
        "tpr_gap (equal opportunity)": float(a.recall_tpr - b.recall_tpr),
        "fpr_gap": float(a.fpr - b.fpr),
        "fnr_gap": float(b.fnr - a.fnr),
        "selection_rate_gap (demographic parity)": float(a.selection_rate - b.selection_rate),
    }


# --------------------------------------------------------------------------
# Experiment A: dataset representation bias
# --------------------------------------------------------------------------
def run_experiment_a(share_b_skewed: float = 0.05, n_train: int = 4000,
                     n_test_per_group: int = 1500, seed: int = 7) -> dict:
    """Same model, same features, same balanced test set.
    Only the composition of the TRAINING data changes."""
    test = make_pathway_data(n_test_per_group, n_test_per_group, seed=seed + 1000)
    results = {"test": test}
    for name, share_b in (("skewed", share_b_skewed), ("representative", 0.5)):
        n_b = int(round(n_train * share_b))
        train = make_pathway_data(n_train - n_b, n_b, seed=seed)
        model = GradientBoostingClassifier(random_state=0, max_depth=3, n_estimators=150)
        model.fit(train[FEATURES], train["qualified"])
        pred = model.predict(test[FEATURES])
        rep = group_report(test, pred)
        results[name] = {"train": train, "model": model, "pred": pred,
                         "report": rep, "gaps": fairness_gaps(rep), "share_b": share_b}
    return results


# --------------------------------------------------------------------------
# Experiment B: algorithmic bias (data held fixed)
# --------------------------------------------------------------------------
def _best_global_threshold(scores, y):
    grid = np.linspace(0.05, 0.95, 181)
    accs = [((scores >= t).astype(int) == y).mean() for t in grid]
    return float(grid[int(np.argmax(accs))])


def _equal_opportunity_thresholds(scores, y, groups, target_tpr):
    """Per-group threshold whose TPR is closest to target_tpr (post-processing
    in the spirit of Hardt, Price & Srebro, 2016)."""
    th = {}
    grid = np.linspace(0.02, 0.98, 481)
    for g in ("A", "B"):
        m = groups == g
        pos = scores[m][y[m] == 1]
        tprs = np.array([(pos >= t).mean() for t in grid])
        th[g] = float(grid[int(np.argmin(np.abs(tprs - target_tpr)))])
    return th


def run_experiment_b(n_per_group: int = 3000, seed: int = 11,
                     global_threshold: float | None = None) -> dict:
    """Training data, validation data and test data are FIXED and balanced.
    Three algorithmic designs are compared:

    B1  all features, one global threshold chosen to maximise overall accuracy
    B2  feature-selection change: drop the proxy feature `years_experience`
    B3  keep all features, use group-specific thresholds chosen on validation
        data so both groups get the same true-positive rate (equal opportunity)
    """
    train = make_career_break_data(n_per_group, seed=seed)
    val = make_career_break_data(n_per_group // 2, seed=seed + 1)
    test = make_career_break_data(n_per_group // 2, seed=seed + 2)

    def lr():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    out = {"train": train, "test": test}

    # B1 baseline
    m1 = lr().fit(train[FEATURES], train["qualified"])
    s_val = m1.predict_proba(val[FEATURES])[:, 1]
    t_star = _best_global_threshold(s_val, val["qualified"].to_numpy())
    t1 = global_threshold if global_threshold is not None else t_star
    s_test = m1.predict_proba(test[FEATURES])[:, 1]
    p1 = (s_test >= t1).astype(int)
    r1 = group_report(test, p1)
    out["B1"] = {"label": "Baseline", "threshold": t1,
                 "accuracy_optimal_threshold": t_star, "pred": p1, "scores": s_test,
                 "report": r1, "gaps": fairness_gaps(r1),
                 "coef": dict(zip(FEATURES, m1[-1].coef_[0].round(3)))}

    # B2 drop proxy feature
    f2 = ["qualification_score", "technical_score"]
    m2 = lr().fit(train[f2], train["qualified"])
    t2 = _best_global_threshold(m2.predict_proba(val[f2])[:, 1], val["qualified"].to_numpy())
    p2 = (m2.predict_proba(test[f2])[:, 1] >= t2).astype(int)
    r2 = group_report(test, p2)
    out["B2"] = {"label": "Drop proxy", "threshold": t2,
                 "pred": p2, "report": r2, "gaps": fairness_gaps(r2)}

    # B3 equal-opportunity thresholds (same model as B1)
    target = float(group_report(val, (s_val >= t_star).astype(int)).loc["Overall", "recall_tpr"])
    th = _equal_opportunity_thresholds(s_val, val["qualified"].to_numpy(),
                                       val["group"].to_numpy(), target)
    thr_vec = np.where(test["group"].to_numpy() == "A", th["A"], th["B"])
    p3 = (s_test >= thr_vec).astype(int)
    r3 = group_report(test, p3)
    out["B3"] = {"label": "Group thresholds", "threshold": th,
                 "pred": p3, "report": r3, "gaps": fairness_gaps(r3)}
    return out
