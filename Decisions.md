# 🧭 Decisions & Tradeoffs

This file is the long-form companion to the [README.md](README.md). The README answers _"what is this system and how do I use it?"_. This file answers _"why was it built this way?"_.

Each section follows the same shape: the context that raised the question, the decision that was taken, the reasoning, and what got traded away.

---

## 📚 Index

1. [Allocation Strategy: Why Equal-Weight Won](#1-allocation-strategy-why-equal-weight-won)
2. [Why Classification Instead of Regression](#2-why-classification-instead-of-regression)
3. [Why a 5% Threshold for the Target](#3-why-a-5-threshold-for-the-target)
4. [Why ROC-AUC for Model Selection](#4-why-roc-auc-for-model-selection)
5. [Why Logistic Regression Over Tree-Based Models](#5-why-logistic-regression-over-tree-based-models)
6. [Why a Learned `t_min` Instead of 0.5](#6-why-a-learned-t_min-instead-of-05)
7. [Why the Per-IPO 60% Allocation Cap Was Removed](#7-why-the-per-ipo-60-allocation-cap-was-removed)
8. [How Allotment is Approximated, and What That Trades Away](#8-how-allotment-is-approximated-and-what-that-trades-away)
9. [Why NII Was Chosen as the Simulated Investor Category](#9-why-nii-was-chosen-as-the-simulated-investor-category)
10. [Engineered Features: Ratios and Missingness Flag](#10-engineered-features-ratios-and-missingness-flag)
11. [Why Multi-Year Company Financials Were Excluded](#11-why-multi-year-company-financials-were-excluded)
12. [Evaluation Setup: Time-Based Holdout, TimeSeriesSplit, gap=30](#12-evaluation-setup-time-based-holdout-timeseriessplit-gap30)

---

## 1. Allocation Strategy: Why Equal-Weight Won

### The original design
On any given day with `k` IPOs above `t_min`, capital was to be split using:
```
wᵢ ∝ pᵢ^α        with α tuned over {-0.5, -0.25, 0, 0.5, 1, 2}
```
where:
- `α = 0` recovers equal weighting,
- `α > 0` concentrates capital on higher-probability IPOs,
- `α < 0` mildly diversifies toward lower-probability IPOs.

The intuition was that the model's confidence is a stand-in for inverse risk, and capital should follow that signal.

### Why this formulation?
The reasoning was first-principles. An optimal portfolio allocator should size each position by the IPO's risk-adjusted expected return. But on this dataset, expected return cannot be reliably estimated (see [Section 2](#2-why-classification-instead-of-regression)). All what the model produces is a probability, which is best read as a *risk signal*: a higher probability means a lower risk that the trade fails to clear the threshold.

That leaves three plausible allocation rules to size positions purely on risk:

1. **Equal weight (`wᵢ = 1/k`).** Treats every surviving IPO the same. Robust, but ignores the fact that some IPOs are clearly less risky than others.
2. **Linear in probability (`wᵢ ∝ pᵢ`).** Tilts capital toward higher-confidence IPOs. The direction is sensible (safer IPOs get more capital), but there is no guarantee the gradient is right. It might be too cautious or too aggressive.
3. **Power-law in probability (`wᵢ ∝ pᵢ^α`).** Generalizes both. The exponent `α` controls how sharply the allocator concentrates capital with confidence:
   - `α = 0` recovers equal weighting.
   - `α = 1` recovers the linear rule.
   - `α > 1` concentrates capital even more aggressively on the highest-`pᵢ` IPOs.
   - `α < 0` inverts the tilt. Lower-`pᵢ` IPOs get more weight, a deliberate diversification stance for cases where the model overweights its own confidence.

Treating `α` as a hyperparameter is what makes the formulation principled. Rather than guessing whether equal-weighting or linear-in-`pᵢ` is the right gradient, the data picks. The grid `α ∈ {-0.5, -0.25, 0, 0.5, 1, 2}` was tuned via backtesting alongside `t_min` which resulted in the optimal value of `α ~ 1.98.`

### What backtesting showed
Two findings:

1. The tuned `pᵢ^α` rule barely outperformed plain equal-weighting (`α = 0`) on the days where it had a choice to make.
2. On most days, it had no choice to make. EDA revealed that **~85% of IPO closing days have only a single IPO**. On those days `α` is mathematically irrelevant: there is one IPO, it gets all the capital, end of story.

Net effect: the allocation rule was tuning a parameter that mattered on 15% of days, and even there only marginally.

### Decision
Drop the `pᵢ^α` formulation. Use plain equal weighting among IPOs that pass `t_min`.

### Why this is the right call here
Three reasons:

1. **The data structure forces it.** When 85% of days have a single IPO, the system is fundamentally a selection problem, not an allocation problem. The threshold is the lever that matters.
2. **Simpler is more robust.** A complex allocation rule tuned on thin data is exactly the kind of thing that overfits and degrades silently in production.
3. **Less to retune as the model changes.** Every retraining iteration would otherwise need an `α` retune to stay coherent.

### What this trades away
On rare days with several high-conviction IPOs that differ a lot in predicted probability, equal weighting leaves a small amount of expected value on the table. The tradeoff is acceptable given how rarely those days occur.

### Takeaway
> Match the strategy's complexity to the structure of the opportunities. A sophisticated allocator on a dataset that mostly does not need allocation is just a more elaborate way to overfit.

---

## 2. Why Classification Instead of Regression

### Initial approach
The problem was first framed as a regression task: predict the exact listing-day percentage gain. A return-aware allocator could then size positions in proportion to expected return.

### What broke
Regression failed to generalize.
- Cross-validation R² hovered around 0 with high variance across folds.
- Holdout R² was negative (worse than predicting the mean).
- RMSE landed around 25 to 30 percentage points, dominated by the heavy right tail.

### Root cause
IPO listing returns have three properties that make exact-magnitude prediction unreliable:
1. Heavy-tailed distribution: A handful of IPOs with very large gains dominate the loss surface.
2. Weak signal for magnitude: Public pre-listing features carry information about _direction_, not about how big the move will be.
3. Small dataset: Only 720 cleaned samples after filtering, which is well below the threshold for stable regression on a noisy, heavy-tailed target.

### Key insight
The same features that fail at predicting magnitude succeed at predicting whether the magnitude will clear a meaningful threshold. Holdout AUC of 0.84 on the binary task vs negative R² on the regression task is a stark gap.

### Decision
Reframe as binary classification:
```
target = 1 if listing_gain_% > 5 else 0
```

The model is used for two things:
1. **Risk filtering:** Keep IPOs likely to clear a meaningful threshold, drop the rest.
2. **Ranking:** Order surviving IPOs by predicted probability for downstream allocation.

### What this trades away
A return-aware allocator. Without reliable magnitude estimates, the system cannot risk-weight positions in a return-sensitive way. The fix was to fall back on simple, robust allocation heuristics (see [Section 1](#1-allocation-strategy-why-equal-weight-won)).

### Takeaway
> In noisy financial domains with limited data, learning the directional signal is often more useful than learning the exact magnitude. The cost of an unstable regression is much higher than the value of "exact" predictions you cannot trust.

---

## 3. Why a 5% Threshold for the Target

### The question
A binary classifier needs a threshold. The most defensible default is `listing_gain > 0`. Why pick 5% instead?

### The reasoning
A naive `>0` target conflates "the IPO listed up" with "the trade was profitable", and these are not the same thing.

In practice an investor faces:
- Brokerage fees on the buy and the sell side
- STT, stamp duty, exchange charges
- Slippage between the listing-auction discovered open price and the actual fill
- The opportunity cost of capital blocked during the application window

A small positive listing gain (say +1%) does not survive these frictions. The 5% buffer is the smallest round number that comfortably absorbs typical retail/NII trading costs and adds a small risk margin on top.

### Side benefit: class balance
At 5%, the positive class lands at roughly 57% of the dataset. No reweighting or resampling is needed. A naive `>0` target would skew much more positive and weaken the contrast the classifier is trying to learn.

### What this trades away
The 5% threshold is somewhat arbitrary. A more rigorous version would tune the threshold itself (treating it as a hyperparameter of the business problem rather than the model). That was deferred since the dataset is small and adding a second tunable to the target definition risks overfitting the entire pipeline to one cohort.

---

## 4. Why ROC-AUC for Model Selection

### The candidates
Plausible model selection metrics included accuracy, F1, log loss, Brier score, and ROC-AUC.

### Why AUC won
The downstream system does not consume a hard 0/1 prediction. It consumes a calibrated probability and applies a learned threshold `t_min`. Two implications:

1. **The business threshold is not 0.5.** Backtesting eventually pins it around 0.41. So a metric that is sensitive to the choice of a fixed cutoff (accuracy, F1) is measuring the wrong thing.
2. **Ranking quality matters.** When more than one IPO clears `t_min` on the same day, the model's ranking determines who ends up in the portfolio. AUC directly measures rank quality across the full probability range.

AUC also has a clean interpretation in this domain: the probability that a randomly chosen positive IPO is scored higher than a randomly chosen negative IPO.

### What this trades away
AUC is insensitive to absolute probability values, so it does not by itself ensure the model's `0.4` actually means a 40% chance. That gap is closed by the bucketed expected-return analysis in [README → Key insights from the backtest](README.md#-key-insights-from-the-backtest), which validates that the predicted probabilities track realized outcomes monotonically.

---

## 5. Why Logistic Regression Over Tree-Based Models

### The candidates
Logistic regression with elastic-net, random forest, XGBoost, LightGBM. All four were tuned with Optuna on time-series CV.

### What the numbers said
Tree ensembles did not clearly beat a well-regularized logistic regression on time-series CV ROC-AUC. They also showed higher fold-to-fold variance, which is the kind of instability you do not want when the holdout set is just one cohort and there is no second chance.

### Why this is not surprising
Three properties of the problem favor a linear model here:

1. **Small dataset.** Roughly 700 samples after cleaning. Tree ensembles thrive on volume, where they can carve out interactions. With this sample count, a tuned linear model is competitive and safer.
2. **Mostly monotonic feature → outcome relationships.** Higher subscription multiples, smaller issue size, positive GMP all generally point in the same direction. Linear models capture this cleanly. Trees can too, but they pay an overfitting tax for the privilege of expressing non-monotonic shapes that are not actually there.
3. **Need for clean calibrated probabilities downstream.** Logistic regression produces well-behaved probability outputs almost by construction. Tree ensemble outputs typically need more aggressive calibration before they behave like real probabilities.

### Bonus practical wins
- 📐 Interpretable coefficients. A reader can inspect which way each feature pushes the prediction.
- 🪶 Tiny model artifact. The serialized pipeline is a few hundred KB rather than tens of MB.
- ⚡ Fast inference. Useful when this becomes a scheduled job that runs minutes before the bidding window closes.

### What this trades away
Any genuine non-linear interaction in the data (for example, a feature that matters only when GMP is missing, or only in 2020 IPOs) is harder for a linear model to capture. The ratio features (`qib_ratio`, `nii_ratio`, `retail_ratio`) and the `is_gmp_missing` flag exist partly to give the linear model some of those interactions explicitly.

### Takeaway
> Model complexity should be the dependent variable, not the independent one. The data and the downstream system tell you how complex the model is allowed to be.

---

## 6. Why a Learned `t_min` Instead of 0.5

### The naive default
For binary classification the textbook decision boundary is 0.5. In a balanced dataset with symmetric costs, that is the right call.

### Why it is wrong here
The costs are not symmetric. A false positive (trade an IPO that loses money) costs real capital. A false negative (skip an IPO that would have been profitable) only costs opportunity. Worse, the realized return distribution is heavily right-tailed, so the average IPO cleared by the model is positive even when the calibrated probability is well below 0.5.

The bucketed expected-return table from the backtest makes this concrete:

| Probability bucket | Avg listing-day return |
|--------------------|------------------------|
| 0.3 to 0.4         | −3.2%                  |
| 0.4 to 0.5         | +4.1%                  |
| 0.5 to 0.6         | +14.6%                 |
| 0.6 to 0.7         | +24.0%                 |
| 0.7+               | +55.3%                 |

Expected return crosses zero somewhere around `p = 0.4`. That is the right place to draw the line, and it is not 0.5.

### Decision
Treat the trading threshold `t_min` as a hyperparameter of the strategy, not the model. Tune it via backtesting on CV folds. The learned value lands at `0.4091`, which matches the empirical zero-crossing of expected return.

### What this implies for evaluation
The classifier is no longer evaluated by its accuracy at 0.5. It is evaluated by AUC (rank quality across all thresholds) and calibration (do the probabilities mean what they say). The threshold itself is a backtester output.

---

## 7. Why the Per-IPO 60% Allocation Cap Was Removed

### The original guardrail
Early versions capped the allocation to any single IPO at 60% of capital, framed as a risk control. The intuition was textbook: avoid concentration risk.

### What actually happened
On the 85% of days with only one IPO, the cap meant 40% of capital sat idle by construction. That idle capital earned nothing while the model's strongest signals were left unfunded.

The cumulative-return curves showed the cost clearly. The capped strategy systematically under-deployed capital and underperformed simpler baselines on the same selection logic.

### Decision
Remove the cap. Allocate 100% of capital across IPOs that pass `t_min` on a given day, equally weighted.

### Why this is not reckless
The "concentration risk" framing assumes a portfolio of many simultaneous positions where one bad apple can spoil the bunch. That is not the structure here. On most days there is exactly one IPO. Capping a single-position day at 60% is not diversification, it is just leaving capital on the bench.

True risk control in this system lives elsewhere:
- The threshold `t_min` itself (only enter trades with positive expected value).
- The 5% target buffer (only learn to clear a meaningful threshold).
- Discretionary review at decision time (a human-in-the-loop layer is recommended for live use).

### What this trades away
A small amount of theoretical protection against a single catastrophic IPO. That is mitigated by the threshold filter and is acceptable given the cost the cap was imposing on real performance.

### Takeaway
> Risk controls must be aligned with the structure of available opportunities. Generic guardrails imported from textbook portfolio theory can be net-negative in sparse decision settings.

---

## 8. How Allotment is Approximated, and What That Trades Away

### The approximation
For each IPO, the simulated allotment ratio is:
```
allotmentᵢ = 1 / max(1, NII_subscription_multipleᵢ)
```

If an IPO is subscribed 5x in the NII category, an applicant for `X` shares receives roughly `X / 5`. If it is undersubscribed (multiple < 1), the applicant receives the full requested allotment.

### Why this approximation is reasonable
Indian IPO allotment rules in the NII category are pro-rata once the issue is oversubscribed. The simple `1 / max(1, multiple)` form is a clean first-order model of that rule.

### What the approximation misses
1. **Category-specific minimum lot rules.** Real allotment respects minimum bid lots, which can round small allocations to zero.
2. **Lottery dynamics in retail.** The retail category does not follow pro-rata. It is a lottery once oversubscribed, with binary outcomes per applicant. Modeling that requires a different formulation entirely.
3. **NII sub-categories.** Post-2021 SEBI changes split NII into sNII (small) and bNII (big) with different allotment rules. The approximation does not differentiate.
4. **Application size effects.** Real allotment depends on application size and SEBI-mandated rules around minimum allotment in oversubscribed scenarios.

### Why the simple form was kept anyway
The approximation is good enough to remove the most egregious modeling assumption (that the strategy gets full allotment on every IPO regardless of subscription level). Refining it further would require IPO-by-IPO category-rule modeling that the available data does not cleanly support.

### Takeaway
> Earlier iterations assumed 100% allotment, which dramatically overstated returns on heavily oversubscribed IPOs. Replacing that with the pro-rata approximation cut the headline numbers but moved them closer to what an actual NII applicant would experience.

---

## 9. Why NII Was Chosen as the Simulated Investor Category

### The categories
Indian IPO subscription has three categories: QIB (qualified institutional buyers), NII (non-institutional investors), and retail individual investors. Each has different allotment rules and per-application limits.

### Why NII

1. **Retail's ₹2 lakh per-IPO ceiling caps absolute returns.** SEBI caps a single retail application at ₹2 lakh, regardless of how much capital the investor actually has available. That means a retail investor cannot earn returns proportional to their deployable capital. Even with a high-conviction prediction and a large bankroll, the per-IPO upside is bounded by the cap. NII has no equivalent ceiling (sNII covers ₹2L–₹10L, bNII goes above ₹10L), which lets the strategy stay capital-efficient as the portfolio scales.
2. **Pro-rata allotment is deterministic.** NII allocation follows a clean pro-rata rule once the subscription multiple is known. Retail is a lottery once oversubscribed, which would force the backtester to simulate stochastic allotment outcomes per applicant.
3. **Cleanest mapping to publicly observable signal.** The NII subscription multiple at end of bidding window is the field that drives the allotment math, and it is reliably available in the dataset.

### What this trades away
Retail-only investors face two restrictions that break this strategy directly: (a) the ₹2 lakh per-IPO cap limits absolute deployable capital regardless of conviction, and (b) retail allotment is a lottery once oversubscribed, so the pro-rata math falls apart. A retail-targeted version would need to model both, and the headline returns would scale very differently.

---

## 10. Engineered Features: Ratios and Missingness Flag

### `qib_ratio`, `nii_ratio`, `retail_ratio`
The raw subscription multiples (`qib`, `nii`, `retail`) capture the magnitude of demand from each category. The ratio features (each category's share of total subscription) capture the _composition_ of demand independent of magnitude.

This matters because:
- An IPO with 100x QIB and 5x retail has very different market sentiment from one with 100x retail and 5x QIB, even though both might end up with similar `total` subscription.
- The composition is a leading indicator of who is driving the demand and, by extension, who might exit on listing day.

For a linear model, supplying the ratios explicitly removes a non-linearity it would otherwise need to discover from raw multiples.

### `is_gmp_missing`
GMP coverage is incomplete. Earlier or smaller IPOs frequently have no recorded GMP. Naive imputation (with median or zero) silently equates "no signal" with "neutral signal", which is a strong assumption.

The flag turns missingness itself into a signal. The model can then learn separate effects:
- The effect of an actual GMP value when one exists.
- The effect of being in the cohort where GMP was not tracked at all.

This is a small but meaningful regularization in a small-dataset setting.

### What this trades away
A few extra features in a dataset of ~700 samples. Feature selection (during model training) is allowed to drop them if they do not pull their weight, so the downside is bounded.

---

## 11. Why Multi-Year Company Financials Were Excluded

### The temptation
IPO prospectuses include three to five years of audited financials. Trends in revenue growth, margin expansion, and leverage seem like obvious features.

### Why they were excluded

1. **Schema inconsistency.** Different prospectuses report different metrics over different time spans. Some have three years, some four, some only two. Encoding multi-year history would require extensive imputation or a complex schema-normalization layer, both of which inject noise into a small dataset.

2. **Indirect signal already captured.** Investor demand absorbs fundamental signal. If the financials are good, QIB and NII subscription tends to reflect that. The subscription features therefore carry most of the fundamental information indirectly, in cleaner form.

3. **Listing-day dynamics are sentiment-driven.** This is not a long-term valuation problem. It is a short-window, near-listing-day price prediction problem. The relevant signals are subscription patterns, GMP, and issue characteristics. Long-term fundamentals matter less here than they would in a multi-quarter holding-period model.

### What this trades away
A more comprehensive feature set that might catch outlier IPOs whose subscription numbers do not fully reflect their fundamentals. A future iteration could fold in summarized growth metrics (one-year revenue growth, latest operating margin) once the schema-normalization work is done.

---

## 12. Evaluation Setup: Time-Based Holdout, TimeSeriesSplit, gap=30

### Why time-based, not random
IPO outcomes are not IID. Market regimes shift, regulatory environments change, GMP reporting evolves. A random train/test split would let the model peek at the future via leakage from later cohorts into training folds. A time-based split forces the evaluation to mirror how the model would actually be used: train on the past, predict on the future.

### Why holdout = last 15%
A time-based holdout requires reserving the most recent block of data. 15% is the standard tradeoff:
- Big enough to give a non-trivial out-of-sample read.
- Small enough to leave the bulk of data for training and CV tuning.

### Why `TimeSeriesSplit` for CV
Standard k-fold CV violates the temporal structure. `TimeSeriesSplit` walks forward through the data, training on a growing prefix and validating on the next chunk. This matches the deployment pattern.

### Why `gap=30`
An IPO listing on day `T` could in principle leak information about market conditions that influence an IPO listing on day `T+1` (sentiment carry, news cycle). A 30-IPO gap between train and validation folds removes near-boundary leakage and keeps the evaluation honest.

### What this trades away
Slightly less effective sample usage. Each CV fold pays a 30-IPO buffer cost. On a 444-IPO evaluation dataset that is not free, but it is the right price to pay to keep the evaluation faithful to the deployment scenario.

---

## ✏️ A note on how this file is meant to evolve

Decisions are not final. As more IPO data accumulates, as the strategy is refined, or as the deployment context changes, some of the tradeoffs above will be revisited.

When that happens, the right move is to update the relevant section in place rather than appending a new one. The goal of this file is to be a current snapshot of why the system looks the way it does today, not a changelog.
