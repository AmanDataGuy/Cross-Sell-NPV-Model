"""
npv_model.xlsx: per-tier NPV with live formulas, so
a reviewer can change an assumption in the Assumptions sheet and watch the
Tier NPV sheet actually move (rather than trusting a number we computed in
Python and pasted in).

NPV per tier = customers x avg_propensity x (avg_balance x margin x
annuity_factor(years, discount_rate)) - customers x campaign_cost, where
annuity_factor is the standard PV-of-an-annuity formula.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import Workbook

from assumptions import CAMPAIGN_COST_PER_CONTACT, DISCOUNT_RATE, NET_INTEREST_MARGIN, RETENTION_YEARS

OUTPUTS = Path("outputs")
FIGURES = Path("figures")

# (Base, Best, Worst) per assumption - Best/Worst are +/-25% moves off the base case.
ASSUMPTIONS = [
    ("Net interest margin", NET_INTEREST_MARGIN, NET_INTEREST_MARGIN * 1.5, NET_INTEREST_MARGIN * 0.75),
    ("Retention years", RETENTION_YEARS, RETENTION_YEARS + 1, max(RETENTION_YEARS - 1, 1)),
    ("Discount rate", DISCOUNT_RATE, DISCOUNT_RATE * 0.75, DISCOUNT_RATE * 1.25),
    ("Campaign cost per contact", CAMPAIGN_COST_PER_CONTACT, CAMPAIGN_COST_PER_CONTACT * 0.6, CAMPAIGN_COST_PER_CONTACT * 1.6),
]


def build_assumptions_sheet(wb: Workbook) -> None:
    ws = wb.active
    ws.title = "Assumptions"
    ws.append(["Assumption", "Base", "Best", "Worst"])
    for name, base, best, worst in ASSUMPTIONS:
        ws.append([name, base, best, worst])


def build_tier_sheet(wb: Workbook, tier_stats: pd.DataFrame) -> None:
    ws = wb.create_sheet("Tier NPV")
    ws.append(["Tier", "Customers", "Avg Propensity", "Avg Balance (floored at 0)",
               "NPV Base", "NPV Best", "NPV Worst"])

    def npv_formula(row: int, scenario_col: str) -> str:
        margin = f"Assumptions!${scenario_col}$2"
        years = f"Assumptions!${scenario_col}$3"
        discount = f"Assumptions!${scenario_col}$4"
        cost = f"Assumptions!${scenario_col}$5"
        annuity = f"((1-(1+{discount})^-{years})/{discount})"
        return f"=B{row}*C{row}*(D{row}*{margin}*{annuity})-B{row}*{cost}"

    first_row = 2
    for i, (tier, stats) in enumerate(tier_stats.iterrows()):
        r = first_row + i
        ws.append([
            tier, int(stats["customers"]), stats["avg_propensity"], stats["avg_balance"],
            npv_formula(r, "B"), npv_formula(r, "C"), npv_formula(r, "D"),
        ])

    total_row = first_row + len(tier_stats)
    ws.append(["Total", f"=SUM(B{first_row}:B{total_row - 1})", "", "",
               f"=SUM(E{first_row}:E{total_row - 1})",
               f"=SUM(F{first_row}:F{total_row - 1})",
               f"=SUM(G{first_row}:G{total_row - 1})"])


def annuity_factor(years: float, discount_rate: float) -> float:
    return (1 - (1 + discount_rate) ** -years) / discount_rate


def npv_by_scenario(tier_stats: pd.DataFrame) -> pd.DataFrame:
    """Same formula as the Excel sheet's live formulas, computed in Python
    once so there's a number to plot - the chart mirrors the workbook, it
    doesn't replace it.

    Uses tier-level averages (count x avg_propensity x avg_balance), same
    simplification as the Excel sheet - keeps the formulas toggle-able by a
    human instead of hiding 45k rows. target_list.csv sums each customer's
    own propensity x balance instead, so it's a few percent more precise;
    the two are expected to be close, not identical.
    """
    margin, years, discount, cost = ASSUMPTIONS
    out = {}
    for i, scenario in enumerate(["Base", "Best", "Worst"]):
        factor = annuity_factor(years[1 + i], discount[1 + i])
        out[scenario] = (
            tier_stats["customers"] * tier_stats["avg_propensity"]
            * (tier_stats["avg_balance"] * margin[1 + i] * factor)
            - tier_stats["customers"] * cost[1 + i]
        )
    return pd.DataFrame(out)


def plot_npv_by_tier(npv: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots()
    npv.plot(kind="bar", ax=ax, color=["#2e75b6", "#70ad47", "#c00000"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Expected NPV ($)")
    ax.set_title("Expected NPV by tier (best / base / worst)")
    ax.set_xticklabels(npv.index, rotation=0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    tiers = pd.read_csv(OUTPUTS / "tiers.csv")
    tiers["balance_floored"] = tiers["balance"].clip(lower=0)
    tier_stats = tiers.groupby("tier").agg(
        customers=("customer_id", "count"),
        avg_propensity=("propensity", "mean"),
        avg_balance=("balance_floored", "mean"),
    ).reindex(["High", "Medium", "Low"])

    wb = Workbook()
    build_assumptions_sheet(wb)
    build_tier_sheet(wb, tier_stats)
    wb.save("npv_model.xlsx")
    print("npv_model.xlsx written.")
    print(tier_stats)

    FIGURES.mkdir(exist_ok=True)
    npv = npv_by_scenario(tier_stats)
    plot_npv_by_tier(npv, FIGURES / "npv_by_tier_chart.png")
    print("\nfigures/npv_by_tier_chart.png written.")
    print(npv)


if __name__ == "__main__":
    main()
