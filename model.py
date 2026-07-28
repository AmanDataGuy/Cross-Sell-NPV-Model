"""
Model: logistic regression (statsmodels, for odds ratios + CI)
vs. a gradient-boosting benchmark (xgboost); then calibration and a
cost-sensitive targeting threshold picked by expected profit instead of a
default 0.5 cutoff.

Whichever model is recommended, its calibrated probabilities are what
downstream NPV math (assumptions.py, build_npv_excel.py) multiplies by
dollars - so calibration quality matters as much as AUC here.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from assumptions import CAMPAIGN_COST_PER_CONTACT, expected_revenue
from db import get_modelling_table
from features import encode_features

FIGURES = Path("figures")
OUTPUTS = Path("outputs")
AUC_MARGIN_FOR_GBM = 0.02  # GBM must beat logistic by more than this to be worth losing interpretability


def fit_logistic_with_inference(X_train: pd.DataFrame, y_train: pd.Series):
    """statsmodels Logit -> fitted model + a table of odds ratios, 95% CI, p-values."""
    X_sm = sm.add_constant(X_train.astype(float))
    fit = sm.Logit(y_train, X_sm).fit(disp=False)
    ci = fit.conf_int()
    table = pd.DataFrame({
        "odds_ratio": np.exp(fit.params),
        "ci_low": np.exp(ci[0]),
        "ci_high": np.exp(ci[1]),
        "p_value": fit.pvalues,
    })
    return fit, table


def plot_lift_gains(y_true: pd.Series, propensity: np.ndarray, path: Path) -> None:
    order = np.argsort(-propensity)
    y_sorted = np.asarray(y_true)[order]
    cum_positive = np.cumsum(y_sorted) / y_sorted.sum()
    pct_contacted = np.arange(1, len(y_sorted) + 1) / len(y_sorted)

    fig, ax = plt.subplots()
    ax.plot(pct_contacted, cum_positive, label="Model")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    ax.set_xlabel("% of customers contacted (ranked by propensity)")
    ax.set_ylabel("% of subscribers captured")
    ax.set_title("Lift / gains chart")
    ax.legend()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(y_true, raw_probs, calibrated_probs, path: Path) -> None:
    fig, ax = plt.subplots()
    for label, probs in [("raw", raw_probs), ("recalibrated (isotonic)", calibrated_probs)]:
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted propensity")
    ax.set_ylabel("Observed subscription rate")
    ax.set_title("Calibration curve")
    ax.legend()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_profit_curve(propensity: np.ndarray, balance: pd.Series, path: Path) -> float:
    """Total expected profit from targeting everyone above threshold t, swept over t. Returns the best t."""
    revenue = balance.apply(expected_revenue).to_numpy()
    thresholds = np.linspace(0.01, 0.5, 50)
    profits = np.array([
        (propensity[propensity >= t] * revenue[propensity >= t] - CAMPAIGN_COST_PER_CONTACT).sum()
        for t in thresholds
    ])
    best_t = thresholds[profits.argmax()]

    fig, ax = plt.subplots()
    ax.plot(thresholds, profits)
    ax.axvline(best_t, color="red", linestyle="--", label=f"best threshold = {best_t:.2f}")
    ax.set_xlabel("Propensity threshold")
    ax.set_ylabel("Total expected profit ($)")
    ax.set_title("Cost-sensitive targeting threshold")
    ax.legend()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return best_t


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)

    df = get_modelling_table()
    X, y = encode_features(df)
    idx_train, idx_test = train_test_split(df.index, test_size=0.25, stratify=y, random_state=42)
    X_train, X_test = X.loc[idx_train], X.loc[idx_test]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]

    # --- Logistic regression, with inference ---
    logit_fit, coef_table = fit_logistic_with_inference(X_train, y_train)
    coef_table.to_csv(OUTPUTS / "logistic_coefficients.csv")
    logit_train_probs = logit_fit.predict(sm.add_constant(X_train.astype(float), has_constant="add")).to_numpy()
    logit_test_probs = logit_fit.predict(sm.add_constant(X_test.astype(float), has_constant="add")).to_numpy()
    logit_full_probs = logit_fit.predict(sm.add_constant(X.astype(float), has_constant="add")).to_numpy()

    # --- Gradient boosting benchmark ---
    gbm = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, eval_metric="logloss", random_state=42)
    gbm.fit(X_train, y_train)
    gbm_train_probs = gbm.predict_proba(X_train)[:, 1]
    gbm_test_probs = gbm.predict_proba(X_test)[:, 1]
    gbm_full_probs = gbm.predict_proba(X)[:, 1]

    results = pd.DataFrame({
        "model": ["logistic", "gbm"],
        "auc": [roc_auc_score(y_test, logit_test_probs), roc_auc_score(y_test, gbm_test_probs)],
        "pr_auc": [average_precision_score(y_test, logit_test_probs), average_precision_score(y_test, gbm_test_probs)],
    }).set_index("model")
    results.to_csv(OUTPUTS / "model_comparison.csv")
    print(results)

    gbm_edge = results.loc["gbm", "auc"] - results.loc["logistic", "auc"]
    recommendation = "gbm" if gbm_edge > AUC_MARGIN_FOR_GBM else "logistic"
    print(f"Recommendation: {recommendation} (GBM AUC edge = {gbm_edge:+.3f}, switch threshold = {AUC_MARGIN_FOR_GBM})")

    train_probs, test_probs, full_probs = (
        (gbm_train_probs, gbm_test_probs, gbm_full_probs) if recommendation == "gbm"
        else (logit_train_probs, logit_test_probs, logit_full_probs)
    )

    # --- Calibration: fit isotonic mapping on train, apply to test/full ---
    iso = IsotonicRegression(out_of_bounds="clip").fit(train_probs, y_train)
    calibrated_test = iso.transform(test_probs)
    calibrated_full = iso.transform(full_probs)

    brier_before = brier_score_loss(y_test, test_probs)
    brier_after = brier_score_loss(y_test, calibrated_test)
    print(f"Brier score: raw={brier_before:.4f}  recalibrated={brier_after:.4f}")
    plot_calibration(y_test, test_probs, calibrated_test, FIGURES / "calibration_curve.png")

    use_calibrated = brier_after < brier_before
    final_test_probs = calibrated_test if use_calibrated else test_probs
    final_full_probs = calibrated_full if use_calibrated else full_probs
    print(f"Using {'recalibrated' if use_calibrated else 'raw'} probabilities downstream.")

    plot_lift_gains(y_test, final_test_probs, FIGURES / "lift_gains_chart.png")

    # --- Cost-sensitive threshold ---
    best_threshold = plot_profit_curve(final_test_probs, df.loc[idx_test, "balance"], FIGURES / "profit_curve.png")
    print(f"Cost-sensitive targeting threshold: {best_threshold:.3f} (vs. naive 0.5)")

    # --- Persist propensities for segment.py / backtest.py / build_npv_excel.py ---
    out = df[["customer_id", "balance", "y"]].copy()
    out["propensity"] = final_full_probs
    out["in_test_set"] = df.index.isin(idx_test)
    out.to_csv(OUTPUTS / "propensities.csv", index=False)

    assert final_full_probs.min() >= 0 and final_full_probs.max() <= 1
    assert results["auc"].min() > 0.5, "model should beat random guessing"


if __name__ == "__main__":
    main()
