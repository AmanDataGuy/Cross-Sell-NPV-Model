"""
Which drivers are real: chi-square (categorical) / t-test
(numeric) per candidate driver against `y`, with Benjamini-Hochberg (FDR)
correction across all of them - so "significant" survives testing many
drivers at once, not just one lucky p < 0.05.

Statistical significance and practical importance are reported separately:
with 45k rows, tiny, useless effects are easy to make "significant".
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
from statsmodels.stats.multitest import multipletests

from db import get_modelling_table

OUTPUTS = Path("outputs")

CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
NUMERIC = ["age", "balance", "day", "campaign", "pdays", "previous"]

# rule-of-thumb practical-importance cutoffs (small effect sizes, Cohen's conventions)
CRAMERS_V_CUTOFF = 0.10
COHENS_D_CUTOFF = 0.20


def cramers_v(table: np.ndarray) -> float:
    chi2, _, _, _ = chi2_contingency(table)
    n = table.sum()
    r, k = table.shape
    return float(np.sqrt((chi2 / n) / (min(r - 1, k - 1))))


def test_categorical(df: pd.DataFrame, col: str) -> tuple[float, float]:
    table = pd.crosstab(df[col], df["y"]).to_numpy()
    _, p_value, _, _ = chi2_contingency(table)
    return p_value, cramers_v(table)


def test_numeric(df: pd.DataFrame, col: str) -> tuple[float, float]:
    a = df.loc[df["y"] == 1, col]
    b = df.loc[df["y"] == 0, col]
    _, p_value = ttest_ind(a, b, equal_var=False)
    pooled_std = np.sqrt((a.var() + b.var()) / 2)
    cohens_d = (a.mean() - b.mean()) / pooled_std
    return p_value, abs(float(cohens_d))


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    df = get_modelling_table()

    rows = []
    for col in CATEGORICAL:
        p_value, effect = test_categorical(df, col)
        rows.append({"driver": col, "test_type": "chi-square", "p_value": p_value,
                      "effect_size": effect, "practically_important": effect >= CRAMERS_V_CUTOFF})
    for col in NUMERIC:
        p_value, effect = test_numeric(df, col)
        rows.append({"driver": col, "test_type": "t-test", "p_value": p_value,
                      "effect_size": effect, "practically_important": effect >= COHENS_D_CUTOFF})

    results = pd.DataFrame(rows)
    _, p_adj, _, _ = multipletests(results["p_value"], method="fdr_bh")
    results["p_value_adj"] = p_adj
    results["significant_fdr"] = results["p_value_adj"] < 0.05
    results = results.sort_values("effect_size", ascending=False)
    results.to_csv(OUTPUTS / "driver_significance.csv", index=False)

    print(results.to_string(index=False))
    real_drivers = results[results["significant_fdr"] & results["practically_important"]]
    print(f"\n{len(real_drivers)} drivers are both FDR-significant and practically important:")
    print(real_drivers["driver"].tolist())

    assert results["significant_fdr"].any(), "expected at least one FDR-significant driver"


if __name__ == "__main__":
    main()
