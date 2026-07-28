"""
Segment: a shallow decision tree (FJ's own method) splits
customers into High/Med/Low target tiers.

Rather than a separate classifier that might disagree with model.py's
calibrated propensity, the tree is fit *on* the propensity itself
(max_depth=3) - it's a surrogate that explains the model's own risk
ranking in a handful of human-readable rules, which is what makes a tier
defensible to a client ("why is this customer High?").
"""
from pathlib import Path

import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text

from assumptions import CAMPAIGN_COST_PER_CONTACT, expected_revenue
from db import get_modelling_table
from features import encode_features

OUTPUTS = Path("outputs")
TIER_ORDER = ["Low", "Medium", "High"]


def assign_tiers(df: pd.DataFrame) -> tuple[pd.Series, DecisionTreeRegressor, pd.DataFrame]:
    X, _ = encode_features(df)
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=0.05, random_state=42)
    tree.fit(X, df["propensity"])

    leaf_id = pd.Series(tree.apply(X), index=df.index)
    leaf_mean = df["propensity"].groupby(leaf_id).mean()
    q1, q2 = leaf_mean.quantile([1 / 3, 2 / 3])

    def tier_for(mean_propensity: float) -> str:
        if mean_propensity >= q2:
            return "High"
        if mean_propensity >= q1:
            return "Medium"
        return "Low"

    leaf_tier = leaf_mean.apply(tier_for)
    tier = leaf_id.map(leaf_tier)
    return tier, tree, X


def main() -> None:
    propensities = pd.read_csv(OUTPUTS / "propensities.csv")
    df = get_modelling_table().merge(propensities[["customer_id", "propensity"]], on="customer_id")

    tier, tree, X = assign_tiers(df)
    df["tier"] = pd.Categorical(tier, categories=TIER_ORDER, ordered=True)

    tier_summary = df.groupby("tier", observed=True)["propensity"].agg(["mean", "count"])
    print(tier_summary)
    assert tier_summary.loc["High", "mean"] > tier_summary.loc["Medium", "mean"] > tier_summary.loc["Low", "mean"], \
        "tiers should have distinct, ordered propensity"

    df[["customer_id", "balance", "propensity", "tier"]].to_csv(OUTPUTS / "tiers.csv", index=False)
    (OUTPUTS / "segment_tree_rules.txt").write_text(export_text(tree, feature_names=list(X.columns)))

    target_list = df[["customer_id", "propensity", "tier", "balance"]].copy()
    target_list["expected_npv"] = (
        target_list["propensity"] * target_list["balance"].apply(expected_revenue)
        - CAMPAIGN_COST_PER_CONTACT
    )
    target_list = target_list.drop(columns="balance").sort_values("expected_npv", ascending=False)
    target_list.to_csv("target_list.csv", index=False)
    print(f"\ntarget_list.csv written: {len(target_list)} customers, "
          f"total expected NPV if all targeted = ${target_list['expected_npv'].sum():,.0f}")


if __name__ == "__main__":
    main()
