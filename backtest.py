"""
Back-test: train on the chronologically earlier 80% of
contacts, hold out the most recent 20% as a stand-in for "the next
campaign", and check that customers the model would rank High actually
subscribed at a higher rate than Medium/Low in that unseen period. Turns
"the model should work" into "it did, on data it never saw".

Bank Marketing's rows are in original contact-date order (UCI docs: May
2008 - Nov 2010), and customer_id was assigned in that same row order by
load.py, so sorting by customer_id recovers chronological order without
needing an explicit date column.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression

from db import get_modelling_table
from features import encode_features
from model import fit_logistic_with_inference

FIGURES = Path("figures")
OUTPUTS = Path("outputs")
HOLDOUT_FRACTION = 0.2


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)

    df = get_modelling_table().sort_values("customer_id")
    split = int(len(df) * (1 - HOLDOUT_FRACTION))
    train_idx, holdout_idx = df.index[:split], df.index[split:]

    X, y = encode_features(df)
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_holdout, y_holdout = X.loc[holdout_idx], y.loc[holdout_idx]

    # a category seen only in the holdout period leaves its dummy column
    # constant (all-zero) in X_train, which makes the MLE Hessian singular
    non_constant = X_train.columns[X_train.nunique() > 1]
    dropped = X_train.columns.difference(non_constant)
    if len(dropped):
        print(f"Dropped {len(dropped)} column(s) constant in the training period: {list(dropped)}")
    X_train, X_holdout = X_train[non_constant], X_holdout[non_constant]

    logit_fit, _ = fit_logistic_with_inference(X_train, y_train)
    train_probs = logit_fit.predict(sm.add_constant(X_train.astype(float), has_constant="add")).to_numpy()
    holdout_probs = logit_fit.predict(sm.add_constant(X_holdout.astype(float), has_constant="add")).to_numpy()

    iso = IsotonicRegression(out_of_bounds="clip").fit(train_probs, y_train)
    calibrated_holdout = iso.transform(holdout_probs)

    # rank -> qcut so tied isotonic outputs still split into 3 equal-sized tiers
    ranked = pd.Series(calibrated_holdout, index=holdout_idx).rank(method="first")
    tier = pd.qcut(ranked, q=3, labels=["Low", "Medium", "High"])

    actual_rate = pd.DataFrame({"tier": tier, "y": y_holdout}).groupby("tier", observed=True)["y"].mean()
    actual_rate = actual_rate.reindex(["Low", "Medium", "High"])
    print("Actual subscription rate by predicted tier, on held-out future data:")
    print(actual_rate)

    assert actual_rate["High"] > actual_rate["Medium"] > actual_rate["Low"], \
        "back-test failed: predicted ranking did not hold on unseen data"

    fig, ax = plt.subplots()
    actual_rate.plot(kind="bar", ax=ax, color=["#999999", "#5b9bd5", "#2e75b6"])
    ax.set_ylabel("Actual subscription rate")
    ax.set_title(f"Back-test: actual rate by tier (last {HOLDOUT_FRACTION:.0%} of contacts, out-of-time)")
    fig.savefig(FIGURES / "backtest_tier_actual_rate.png", bbox_inches="tight")
    plt.close(fig)

    actual_rate.to_csv(OUTPUTS / "backtest_tier_actual_rate.csv")


if __name__ == "__main__":
    main()
