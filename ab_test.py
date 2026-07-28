"""
Checks whether the High tier's targeting can be trusted causally, not just
correlationally - and if not yet, what it would take to find out.

Every customer here was already contacted (bank-full.csv is a completed
campaign's contact history) - there is no genuine "not contacted" control
group with an observed outcome. An 80/20 split and "lift" comparison here
is NOT a causal effect estimate; both arms received the identical
historical treatment. What that split IS useful for: a negative control,
confirming the test doesn't manufacture a spurious "lift" out of two random
samples of the same population. The actual deliverable is the power
calculation - the sample size a REAL randomized holdout on a future
campaign would need to detect a meaningful effect.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

OUTPUTS = Path("outputs")
MDE = 0.05  # minimum detectable effect: 5 percentage points
ALPHA = 0.05
POWER = 0.80
SPLIT_SEED = 42
ARM_A_FRACTION = 0.8


def load_high_tier() -> pd.DataFrame:
    tiers = pd.read_csv(OUTPUTS / "tiers.csv")
    outcomes = pd.read_csv(OUTPUTS / "propensities.csv")[["customer_id", "y"]]
    return tiers[tiers["tier"] == "High"].merge(outcomes, on="customer_id")


def negative_control_split(high: pd.DataFrame) -> dict:
    """80/20 random split of the SAME already-contacted population. Both
    arms received identical treatment, so any real difference is noise -
    this checks the test doesn't manufacture a false lift on its own."""
    rng = np.random.default_rng(SPLIT_SEED)
    is_arm_a = rng.random(len(high)) < ARM_A_FRACTION
    arm_a, arm_b = high[is_arm_a], high[~is_arm_a]

    rate_a, rate_b = arm_a["y"].mean(), arm_b["y"].mean()
    count = np.array([arm_a["y"].sum(), arm_b["y"].sum()])
    nobs = np.array([len(arm_a), len(arm_b)])
    _, p_value = proportions_ztest(count, nobs)

    se = np.sqrt(rate_a * (1 - rate_a) / len(arm_a) + rate_b * (1 - rate_b) / len(arm_b))
    diff = rate_a - rate_b

    return {
        "arm_a_n": len(arm_a), "arm_a_rate": rate_a,
        "arm_b_n": len(arm_b), "arm_b_rate": rate_b,
        "diff_pp": diff * 100, "ci_low_pp": (diff - 1.96 * se) * 100, "ci_high_pp": (diff + 1.96 * se) * 100,
        "p_value": p_value,
    }


def required_sample_size(baseline_rate: float) -> float:
    effect_size = proportion_effectsize(baseline_rate + MDE, baseline_rate)
    return NormalIndPower().solve_power(effect_size=effect_size, alpha=ALPHA, power=POWER, ratio=1.0)


def main() -> None:
    high = load_high_tier()
    baseline_rate = high["y"].mean()
    print(f"High tier: {len(high)} customers, actual subscription rate = {baseline_rate:.1%}")

    control = negative_control_split(high)
    print("\nNegative-control split (both arms already contacted - identical treatment):")
    print(f"  Arm A: n={control['arm_a_n']}, rate={control['arm_a_rate']:.1%}")
    print(f"  Arm B: n={control['arm_b_n']}, rate={control['arm_b_rate']:.1%}")
    print(f"  Difference: {control['diff_pp']:+.1f}pp "
          f"(95% CI {control['ci_low_pp']:+.1f} to {control['ci_high_pp']:+.1f}pp), p={control['p_value']:.3f}")
    assert control["ci_low_pp"] < 0 < control["ci_high_pp"], \
        "negative control found a 'significant' difference between two random samples of the same population"
    print("  Confirmed: no significant difference, as expected - both arms received identical treatment.")

    n_per_arm = required_sample_size(baseline_rate)
    print(f"\nTo detect a {MDE:.0%} lift from a {baseline_rate:.1%} baseline at "
          f"{POWER:.0%} power / {ALPHA:.0%} significance, a REAL randomized holdout needs "
          f"~{n_per_arm:.0f} customers per arm.")
    covers = "comfortably covers" if len(high) >= 2 * n_per_arm else "does not cover"
    print(f"The High tier ({len(high)} customers) {covers} a two-arm test at this size.")

    pd.DataFrame([{**control, "baseline_rate": baseline_rate, "mde_pp": MDE * 100,
                   "required_n_per_arm": n_per_arm}]).to_csv(OUTPUTS / "ab_test_results.csv", index=False)


if __name__ == "__main__":
    main()
