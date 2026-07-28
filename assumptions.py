"""
Business assumptions behind the dollar math, in one place so model.py's
cost-sensitive threshold and build_npv_excel.py's NPV model can't drift
apart. `balance` is the only value signal in this dataset, so a converted
customer's expected revenue is modelled as net interest margin earned on
their balance, discounted over an assumed retention period.

These are placeholder base-case numbers, not real bank economics -
build_npv_excel.py exposes them as toggleable cells (with best/worst
variants) so a reviewer can see the NPV move when they change.
"""

NET_INTEREST_MARGIN = 0.02       # bank's margin on funds under management, base case
RETENTION_YEARS = 3               # years a converted customer is assumed to keep the product
DISCOUNT_RATE = 0.08               # annual discount rate
CAMPAIGN_COST_PER_CONTACT = 5.0     # $ cost to contact one customer (call/mailer), base case


def expected_revenue(balance: float, margin: float = NET_INTEREST_MARGIN,
                      years: int = RETENTION_YEARS, discount_rate: float = DISCOUNT_RATE) -> float:
    """One conversion's expected discounted revenue over `years`, balance floored at 0."""
    principal = max(balance, 0)
    annual_revenue = principal * margin
    return sum(annual_revenue / (1 + discount_rate) ** t for t in range(1, years + 1))
