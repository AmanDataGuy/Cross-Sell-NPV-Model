# Cross-Sell Propensity & Customer NPV — Memo

## The ask

Of our existing 45,211 customers, which should we target for the next-product offer (a term
deposit), how confident can we be that the attributes we're targeting on are real signal and not
noise, and what is that pursuit worth in dollars once campaign cost and customer retention are
priced in? This mirrors FischerJordan's "Models for Long-Term Profitability" case: propensity plus
profitability modelling, not propensity alone.

## What we did

Built a logistic regression (for interpretable odds ratios) and a gradient-boosting benchmark on
real UCI Bank Marketing data, excluding the one column that would have quietly inflated the
result (`duration`, only known after the call happens). Statistically tested 15 candidate customer
attributes against the outcome, correcting for testing that many at once. Calibrated the winning
model's probabilities before letting them touch a dollar figure, since an overconfident model
produces a confidently wrong NPV. Segmented customers into High/Medium/Low tiers with a shallow,
explainable decision tree, then validated the tiering on a slice of contacts the model never
trained on.

## Key result

- **Model**: GBM (AUC 0.802, PR-AUC 0.460) edges out logistic regression (AUC 0.773, PR-AUC 0.419)
  by +0.029 — enough to justify the accuracy trade-off. Isotonic recalibration improved the Brier
  score from 0.0810 to 0.0806; the targeting cutoff was set at 0.070 by expected profit, not the
  textbook default of 0.5.
- **Real drivers** (statistically significant *and* practically important, of 15 tested):
  `poutcome, pdays, previous, campaign, month, contact, housing, job`. Several attributes that
  looked significant on a raw p-value (e.g. `education`, `age`) didn't clear the practical-importance
  bar once effect size was checked.
- **Tiers**: High (5,702 customers, 48.7% avg propensity), Medium (13,873, 11.5%), Low (25,636,
  3.5%) — cleanly separated, and the separation held on out-of-time data: actual subscription rate
  was 52.6% (High) vs. 23.1% (Medium) vs. 19.1% (Low) on the most recent 20% of contacts, which
  were never used in training.

## Recommendation

**Target the High and Medium tiers — 19,575 customers, ~43% of the base — for an expected +$373,829
NPV.** Do not contact the Low tier indiscriminately: on average it *loses* $77,214, because campaign
cost exceeds the expected return once low propensity is priced in. The full ranked list is in
[`target_list.csv`](target_list.csv); the underlying assumptions (net interest margin, retention
years, discount rate, campaign cost) are toggleable in [`npv_model.xlsx`](npv_model.xlsx) with
best/base/worst scenarios, so this recommendation can be re-priced without re-running the model.

**Caveats**: revenue is proxied from account balance (this dataset has no direct product-margin
figure), and best/worst scenarios are +/-25-60% moves off the base assumption, not
externally-sourced bank economics — both should be replaced with real finance-team numbers before
this drives an actual campaign spend decision.

## Recommended validation before scaling

Everything above is correlational, not causal: every customer in this dataset was already
contacted, so there's no real "not contacted" group to compare against, and the back-test (above)
confirms the *ranking* holds out-of-time, not that contacting High-tier customers *causes* the
higher rate. As a sanity check, splitting the already-contacted High tier randomly in two
(`ab_test.py`) finds no spurious difference between the halves (−1.3pp, 95% CI −4.6 to +2.0pp,
p=0.44) — as expected, since both halves received identical treatment; this confirms the testing
method doesn't manufacture a false signal on its own.

**Before scaling this to a real campaign spend decision**: run a genuine randomized holdout —
contact 80% of the High tier, deliberately withhold 20%, and compare actual outcomes. To detect a
5-percentage-point lift at 80% power / 5% significance, that test needs **~1,568 customers per
arm**; the High tier's 5,702 customers comfortably covers it.

## Appendix

- `outputs/segment_tree_rules.txt` — the actual if/else rules behind each tier, in plain English
  (e.g. "customers with X and Y go High") — this is what makes a tier defensible to a client,
  not just a propensity number
- `outputs/ab_test_results.csv` — the negative-control split and power calculation above
- `figures/lift_gains_chart.png` — model lift vs. random targeting
- `figures/calibration_curve.png` — raw vs. recalibrated probability accuracy
- `figures/profit_curve.png` — how the 0.070 targeting threshold was chosen
- `figures/npv_by_tier_chart.png` — expected NPV by tier, best/base/worst
- `figures/backtest_tier_actual_rate.png` — out-of-time validation of the tier ranking
- `deck.pptx` — 5-slide summary of the above
