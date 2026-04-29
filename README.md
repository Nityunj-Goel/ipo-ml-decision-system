# 📈 ML-Driven IPO Investment Decision System

A machine-learning based decision system for the Indian mainboard IPO market. The model predicts the probability that a freshly-issued IPO will list at more than a 5% gain over its issue price, and a backtested strategy converts those probabilities into daily portfolio allocations.

> 🌐 **Live dashboard:** _<placeholder for Streamlit Cloud URL>_
>
> 💻 **Source:** [https://github.com/Nityunj-Goel/ipo-ml-decision-system](https://github.com/Nityunj-Goel/ipo-ml-decision-system)

---

## ⚡ TL;DR

- 🎯 **Task framing:** Binary classification — `P(listing_gain > 5%)` from pre-listing public signals (subscription multiples, issue size, price band, GMP, year, etc.).

- 🤖 **Model:** **logistic regression** (elastic-net), chosen over random forest, XGBoost, and LightGBM on time-series CV ROC-AUC.

- ⚖️ **Decision rule:** On each IPO listing day, allocate capital equally across all IPOs whose predicted probability exceeds a learned threshold `t_min ≈ 0.41`. Per-IPO allotment is approximated as `1 / max(1, NII subscription multiple)` [ Assuming NII category ].

- 🧪 **Holdout result** (unseen 2025 data, 75 trade days, 108 IPOs): Cumulative return **+16.4%**, mean per-trade-day return **+0.20%**, win rate **61.3%**, Sharpe-like **0.43**.

- 📊 **Full backtest** (2017–2025, 444 IPOs, 358 trade days): Cumulative return **+242.5%**, mean per-trade-day return **+0.35%**, win rate **62%**, Sharpe-like **0.33**.

---

## 🧭 Table of Contents

1. [The Problem We're Solving](#1--the-problem-were-solving)
2. [Problem Framing](#2--problem-framing)
3. [Data](#3--data)
4. [Pipeline Architecture](#4--pipeline-architecture)
5. [Modeling](#5--modeling)
6. [Decision Engine](#6--decision-engine)
7. [Business Performance Metrics — Definitions](#7--business-performance-metrics--definitions)
8. [Backtesting & Results](#8--backtesting--results)
9. [Prediction API](#9--prediction-api)
10. [Repository Structure](#10--repository-structure)
11. [Getting Started](#11--getting-started)
12. [Limitations & Assumptions](#12--limitations--assumptions)
13. [Future Scope](#13--future-scope)
14. [FAQ](#14--faq)
15. [Data Sources & Credits](#15--data-sources--credits)
16. [Project Learnings](#16--project-learnings)
17. [Disclaimer](#-disclaimer)

---

## 1. 🎯 The Problem We're Solving

Indian IPO listings are a high-variance event. Some double on day one, others nosedive 30%, and most land somewhere in between. The only thing an investor actually sees, _before_ closing, is a handful of public signals. Subscription numbers, issue size, price band, grey market premium.

So, on every IPO closing day, an investor faces two very practical questions:

> **"Is this IPO worth subscribing to at all?"**
>
> **"If multiple IPOs close today, how do I split my capital between them?"**

This project is an end-to-end answer to both. It does **not** try to predict exact listing-day returns since it turned out to be unreliable on this dataset (see [Decisions.md → Why Classification Instead of Regression](Decisions.md#2-why-classification-instead-of-regression)). Instead, it learns **where the positive expected value lives** in the IPO opportunity space, and route capital there with a simple, robust rule.

The deliverable is three things stitched together:
- 🧠 a calibrated probability model,
- 📐 a backtested decision rule on top of it, and
- 🖥️ a live dashboard + prediction API to actually use it.

---

## 2. 🧩 Problem framing

### What is "listing gain"?
**Listing gain** is the percentage difference between an IPO's listing-day open price and its issue price:

```
listing_gain_% = ((listing_day_open − issue_price) / issue_price) × 100
```

It's the standard market measure of IPO underpricing. How much value the opening auction discovered above what the issuer charged.

### Target
```
target = 1 if listing_gain_% > 5 else 0
```

The 5% buffer absorbs slippage, brokerage, and taxes, making the target closer to realizable profit than to theoretical underpricing.

### Target distribution
A quick read of `listing_gain_%` across the dataset:

- 📈 **Heavy right skew**, with a heavy tail of extreme outliers (a few IPOs with very large gains dominate the variance).
- 🎯 **Dense cluster in the 0–10% range**, where most IPOs land.
- ⚖️ **At the 5% threshold the positive class is ~57% of samples**, so no class imbalance to fight.

The first two facts are why exact-magnitude regression is unreliable on this dataset (see next subsection). The third is why the classifier can be trained without resampling or class weights.

### Why classification, not regression
Regression on listing gain produced cross-validation R² ≈ 0 and holdout R² < 0 due to the heavy-tailed return distribution and weak signal for exact magnitude. Reframing as _"is this IPO likely to clear a meaningful threshold?"_ turned an unstable problem into a stable one (holdout AUC ≈ 0.84). Full reasoning in [Decisions.md → Why Classification Instead of Regression](Decisions.md#2-why-classification-instead-of-regression).

### 🪄 The plot twist: this is mostly a *selection* problem, not a portfolio one

On a multi-IPO day this looks like a classic portfolio allocation problem (which subset to fund, and in what proportions). EDA revealed it usually isn't. **~85% of IPO closing days have only a single IPO**, so the dominant decision is trade vs. no-trade, not how to weight a basket. The system therefore reduces to a two-step rule: filter by a learned probability threshold, then split capital equally among the survivors.

Full design history (including the original `wᵢ ∝ pᵢ^α` rule that was dropped) lives in [Decisions.md → Allocation Strategy](Decisions.md#1-allocation-strategy-why-equal-weight-won).

---

## 3. 📊 Data

### Sources
- 🏢 **[Chittorgarh](https://www.chittorgarh.com/)** — IPO subscription, issue, and listing data
- 💹 **[Investorgain](https://www.investorgain.com/)** — Grey market premium (GMP) snapshots
- 📈 **NSE** — listing-day open prices
- 📉 **BSE** — IPO start/end dates and price band (low/high), used to **backfill missing fields** in the NSE-aggregated dataset. ~407 of 720 IPOs were enriched this way (393 exact-name matches + 14 fuzzy matches via `rapidfuzz`).

### Scope
- 🇮🇳 Indian mainboard IPOs only (SME segment excluded)
- 🗓️ Years: **2006–2025**
- 📦 720 IPOs after cleaning

### Feature set
Pre-listing, publicly observable at end of bidding window:

| Feature                                  | Notes                                                                                                      |
|------------------------------------------|------------------------------------------------------------------------------------------------------------|
| `qib`, `nii`, `retail`, `total`          | Raw subscription multiples by category (and overall)                                                       |
| `qib_ratio`, `nii_ratio`, `retail_ratio` | Each category's share of total subscription. Captures the *composition* of demand independent of magnitude |
| `issue_amount`                           | Issue size (Rs. crores)                                                                                    |
| `price_band_high`, `price_band_low`      | Issue price band                                                                                           |
| `gmp`                                    | Grey market premium (nullable — not always available)                                                      |
| `is_gmp_missing`                         | Binary indicator for missing GMP — turns missingness itself into a signal                                  |
| `year`                                   | Captures regime / cohort effects                                                                           |

### Data quality caveats
- Subscription numbers used at training time are end-of-window snapshots; live inference uses the same field captured ~30 minutes before close, so a small training-serving skew exists.
- GMP coverage is incomplete and reliability varies by source (handled via the `is_gmp_missing` flag).
- See `Limitations` below.

---

## 4. 🏗️ Pipeline architecture

```
Raw scraped data
      │
      ▼
[Data Pipeline]   data collection → aggregation → cleaning → feature engineering
      │
      ▼
[Model Pipeline]  preprocessing → feature selection → training → evaluation (ROC-AUC) → calibration
      │
      ▼
[Decision Engine] probability → threshold filter → equal-weight allocation
      │
      ▼
[Action]          API response / dashboard / (future: broker execution)
```

> 🖼️ _<placeholder: architecture diagram — to be added by author>_


---

## 5. 🤖 Modeling

### Models evaluated
- **Logistic regression (elastic-net) — selected ✅**
- Random forest
- XGBoost
- LightGBM

### Why logistic regression won
On this dataset (~700 IPOs, ~10 features), the tree-based models including hyperparameter-tuned XGBoost and LightGBM failed to clearly outperform a well-regularized logistic regression on time-series CV ROC-AUC, and showed more variance across folds. The combination of a small dataset, mostly-monotonic feature → outcome relationships, and the need for clean calibrated probabilities downstream all favored a simpler model. Logistic regression also gives:

- 📐 well-behaved, interpretable coefficients
- 📊 calibrated probabilities out of the box (with minor post-fit calibration)
- 🪶 a much smaller artifact and faster inference path

Final hyperparameters live in [configs/config.yml](configs/config.yml) under `logistic_regression`:
```yaml
C: 0.0091
l1_ratio: 0.183     # elastic-net (mix of L1 and L2)
solver: saga
class_weight: balanced
```

### Selection methodology
1. Sort dataset by IPO date; reserve **last 15%** as a holdout test set.
2. On the remaining 85%, run [`TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) CV with `n_splits=5`, `gap=30` days to prevent leakage across folds.
3. For each model × hyperparameter set: feature selection (where applicable), train, evaluate ROC-AUC.
4. Hyperparameters tuned with Optuna.
5. Select top model based on best mean ROC-AUC across folds. 
6. Top model retrained on the full pre-holdout set, scored once on holdout.

### Why ROC-AUC
AUC measures separability between the positive and negative class across all thresholds. Desirable here because the business threshold is learned downstream by the backtester. AUC rewards a model that ranks IPOs well, which is exactly what the allocation engine needs.

### Final model performance

| Split                       | ROC-AUC   |
|-----------------------------|-----------|
| Time-series CV (validation) | **0.867** |
| Holdout (unseen)            | **0.839** |

---

## 6. ⚖️ Decision engine

### Threshold filter
On each IPO listing day, only IPOs with `prob ≥ t_min` are considered for trading. `t_min` is **learned, not assumed**. It's tuned in the backtester to maximize cumulative return / stability.

```yaml
portfolio.trade_threshold: 0.4091   # learned via backtesting
```

### Allocation
Capital is split **equally** among all IPOs that pass the threshold on a given day. This is a deliberate simplification of the originally proposed `wᵢ ∝ pᵢ^α` rule — see [Decisions.md → Allocation Strategy](Decisions.md#1-allocation-strategy-why-equal-weight-won).

### Allotment realism
Per-IPO allotment is approximated as `1 / max(1, NII_subscription_multiple)`. Oversubscribed issues yield a pro-rata fraction of the requested shares, which removes the unrealistic "100% allotment" assumption from earlier iterations.

### Daily portfolio return formula
```
contributionᵢ = weightᵢ × allotmentᵢ × listing_gain_%ᵢ
day_return    = Σ contributionᵢ
```

---

## 7. 📐 Business Performance metrics — definitions

| Metric                      | Definition                                                                     |
|-----------------------------|--------------------------------------------------------------------------------|
| Cumulative Return           | `(∏(1 + r_d/100) − 1) × 100` over all daily portfolio returns                  |
| Mean Daily Return           | `(1/N) · Σ(Σ wᵢ × allotmentᵢ × rᵢ)_day`                                        |
| Win Rate                    | Fraction of trade days with positive portfolio return                          |
| % Days Traded               | Fraction of trade days where Σwᵢ > 0                                           |
| Volatility                  | `std(portfolio_return_day)`. Needs ≥ 5 days.                                   |
| Sharpe-like                 | `mean_daily_return / volatility`. No risk-free adjustment. Needs ≥ 5 days.     |
| Avg Return per Calendar Day | `mean_daily_return × num_ipo_days / calendar_days` — adjusts for inactive days |

---

## 8. 📈 Backtesting & results

### Methodology
- 🔁 Walk-forward simulation across 2017–2025
- 📅 Daily granularity (one decision point per IPO listing day)
- 🎛️ Strategy parameters (`t_min`, allocation rule) tuned on CV folds, then applied to the unseen holdout window

### Headline metrics

#### 🧪 Holdout (live simulation on unseen 2025 data)
| Metric                    | Value       |
|---------------------------|-------------|
| Calendar days             | 366         |
| IPO trade days            | 75          |
| IPOs evaluated            | 108         |
| **Cumulative return**     | **+16.38%** |
| Avg return / trade day    | +0.20%      |
| Avg return / calendar day | +0.04%      |
| Win rate (trade days)     | 61.3%       |
| % trade days deployed     | 72.0%       |
| Volatility (trade days)   | 0.47%       |
| Sharpe-like               | +0.43       |

#### 📊 Full backtest (2017 → 2025)
| Metric                                                          | Value        |
|-----------------------------------------------------------------|--------------|
| Calendar days                                                   | 3,256        |
| IPO trade days                                                  | 358          |
| IPOs evaluated                                                  | 444          |
| **Cumulative return (Strategy)**                                | **+242.54%** |
| Cumulative return (Equal-weight allocation, no filter baseline) | **−54%**     |
| Avg return / trade day                                          | +0.35%       |
| Win rate (trade days)                                           | 62.0%        |
| % trade days deployed                                           | 73.5%        |
| Sharpe-like                                                     | +0.33        |

### Baseline comparison
The selection-filtered strategy compounds positively across the full window while the unfiltered "equal-weight on every IPO" baseline ends sharply negative — confirming the model's value is in **filtering**, not in fine-grained allocation.

### 🔍 Key insights from the backtest
| Probability bucket | Avg listing-day return |
|--------------------|------------------------|
| 0.3 – 0.4          | −3.2%                  |
| 0.4 – 0.5          | +4.1%                  |
| 0.5 – 0.6          | +14.6%                 |
| 0.6 – 0.7          | +24.0%                 |
| 0.7+               | +55.3%                 |

Two things to read from this:
1. 🎯 **Expected return crosses zero around p ≈ 0.4** — that's why the learned `t_min` lands there, not at the naive 0.5.
2. 📈 **Predicted probability monotonically tracks realized return** — the model produces a useful ranking signal, in addition to acting as a classifier.

---

## 9. 🔌 Prediction API

A FastAPI service exposes the trained model as an HTTP endpoint.

### Run locally
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Swagger UI: `http://127.0.0.1:8000/docs`

### `POST /predict`
**Request**
```json
{
  "ipos": [
    {
      "nii": 50.0,
      "qib": 30.0,
      "retail": 10.0,
      "total": 25.0,
      "year": 2025,
      "issue_amount": 500.0,
      "price_band_high": 500.0,
      "price_band_low": 475.0,
      "gmp": null
    }
  ]
}
```

**Response**
```json
{
  "allocations": [
    {"probability": 0.7321, "allocation_weight": 1.0}
  ]
}
```

The service returns one `(probability, allocation_weight)` pair per submitted IPO. Weights are computed assuming all submitted IPOs share the same listing day. IPOs below `t_min` receive `allocation_weight = 0`.

---

## 10. 📁 Repository structure

```
.
├── README.md                    # this file
├── Decisions.md                 # deeper "why" behind key technical choices
├── requirements.txt
├── configs/
│   ├── config.yml               # paths, model hyperparams, t_min, threshold
│   └── feature_config.py
├── data/                        # raw, filtered, aggregated CSVs
├── notebooks/                   # eda, models, backtesting, BSEscraper
├── src/
│   ├── data/                    # cleaning + aggregation
│   ├── features/                # engineering + selection
│   ├── models/                  # trainer, eval, experiment runner
│   ├── pipelines/               # data, model, inference, prediction pipelines
│   ├── portfolio/               # allocator, backtester
│   └── utils/
├── app/                         # FastAPI inference service
├── dashboard/                   # Streamlit app + artifact builder
├── artifacts/
│   ├── models/                  # trained, joblib-serialized prediction pipeline
│   └── dashboard/               # trades.csv, meta.json for the dashboard
└── logs/
```

---

## 11. 🚀 Getting started

```bash
# 1. Install
pip install -r requirements.txt

# 2. (Optional) Re-run the modeling notebook end-to-end
jupyter notebook notebooks/models.ipynb

# 3. (Optional) Re-run the backtest and regenerate dashboard artifacts
jupyter notebook notebooks/backtesting.ipynb
python -m dashboard.build_artifacts

# 4. Serve the model
uvicorn app.main:app --host 127.0.0.1 --port 8000

# 5. Launch the dashboard (in a separate shell)
streamlit run dashboard/app.py
```

---

## 12. ⚠️ Limitations & assumptions

### Modeling
- Dataset is small by ML standards (720 IPOs after cleaning).
- Temporal drift exists — listing-day returns trend higher in the recent regime.
- GMP missingness is non-random; some IPOs have no signal there (partially captured via `is_gmp_missing`).

### Strategy / execution
- **Allotment** is approximated from NII subscription multiple — actual allotment depends on category-specific lottery rules.
- **No transaction costs, taxes, or slippage** are modeled. The 5% gain buffer is the only cushion.
- **Trade execution at issue price** is assumed (i.e., assumes successful application + allotment + sell at listing-day open).
- **No capital-blocking model** — funds are assumed available even on consecutive-IPO days.
- **No microstructure modeling** — listing gain is the auction-discovered open price; actual fills can deviate.

### Out-of-distribution risk
A black-swan IPO whose features fall outside the training distribution can produce a confidently-wrong prediction. Backtesting suggests the strategy remains profitable on average over multi-year windows, but a single bad day can be material. A human-in-the-loop layer at decision time is recommended for live use.

---

## 13. 🔮 Future scope

- 📡 **Monitoring** — drift detection on input features and prediction distribution; automated alerts.
- 🔁 **Retraining pipeline** — scheduled retrains as new IPOs list, with model registry / versioning.
- ⏱️ **Live decisioning** — scheduled job that fetches open IPOs, runs predictions ~30 min before bidding close, and emits notifications (or trades).
- ✅ **Pre-flight check** — a dry run at issue open to validate the end-to-end inference path before the real prediction run.
- 🤝 **Broker integration** — order placement and exit strategy automation.
- 🛠️ **CI/CD** — model registry, artifact versioning, deployment automation.
- 🧮 **Multi-year fundamentals** — encode growth trends from prospectus financials (currently excluded due to schema inconsistency across filings; see [Decisions.md](Decisions.md)).

---

## 14. ❓ FAQ

**❔ What is listing gain?**
Listing-day open price minus issue price, expressed as a percentage of issue price.

**❔ Why a 5% threshold instead of >0?**
A minimum economically meaningful return: covers brokerage, slippage, taxes, and a risk buffer. It also makes the target closer to realizable profit than to theoretical underpricing.

**❔ If listing gain is positive, can I actually capture it by selling at market open?**
Mostly, yes — historically, the bulk of IPO underpricing is captured in the opening auction and early trading. The 5% buffer protects against execution noise. But this system models pricing efficiency, not intraday microstructure; precise fills depend on order flow and liquidity not represented here.

**❔ Why not a regression model?**
Tried it. R² ≈ 0 in CV, < 0 on holdout. Heavy-tailed, low-signal returns make exact-magnitude prediction unreliable on this dataset. Classification + ranking is far more stable. Full breakdown in [Decisions.md](Decisions.md#2-why-classification-instead-of-regression).

**❔ Why logistic regression over XGBoost / LightGBM?**
Small dataset (~700 IPOs), mostly-monotonic feature/outcome relationships, and a need for clean calibrated probabilities downstream. Tuned tree ensembles didn't clearly beat a well-regularized elastic-net logistic regression on time-series CV AUC, and showed higher fold-to-fold variance. Simpler model + smaller artifact + faster inference won.

**❔ Why not include 3-year company financials as features?**
Three reasons: (1) prospectus financial coverage is inconsistent across IPOs, (2) demand-side signals like subscription ratios already absorb most of the fundamental signal indirectly, and (3) IPO listing-day price action is more sentiment-driven than fundamentals-driven. A future iteration could fold in summarized growth metrics.

**❔ What if a black-swan IPO arrives?**
That's classic data drift. 100% loss avoidance isn't possible; the strategy is built to be profitable on average over multi-year windows. Mitigations: scheduled retraining, drift monitoring, and a human-in-the-loop approval step before live trades (feasible because IPO events are rare).

---

## 15. 🙏 Data sources & credits

- 🏢 IPO data: [Chittorgarh](https://www.chittorgarh.com/)
- 💹 Grey market premium: [Investorgain](https://www.investorgain.com/)
- 📈 Listing-day prices: NSE
- 📉 IPO start/end dates and price band backfill: BSE

---

## 16. 💡 Project learnings

- 🧭 **Scope first, model later.** Pinning down the business metric, the input/output contract, and verifying the dataset can plausibly support the target — that's the most leveraged hour of the project.
- 🧱 **Modeling is ~10% of the work.** Data, evaluation, and the decision layer around the model are the other 90%.
- 🏃 **Get a baseline working end-to-end before iterating.** Error analysis on a working baseline beats over-investing in any single stage.
- ✂️ **Simpler often wins.** A learned threshold + equal weighting beat a more complex `pᵢ^α` allocation rule once the data structure (one IPO/day on most days) was understood — and a regularized logistic regression beat tuned tree ensembles on this small dataset.
- 🎯 **Centre on business logic, work backwards.** _"What decision are we improving?"_ is a sharper compass than _"what model fits best?"_

---

## 📜 Disclaimer

**This project is for educational and research purposes only. It is not investment advice.** Past performance does not guarantee future results. The author accepts no responsibility for any financial loss arising from use of this system or its outputs. Consult a qualified financial advisor before making any investment decision.
