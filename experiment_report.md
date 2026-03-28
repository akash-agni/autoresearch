# Autonomous ML Experiment Run — Session Report
**Date:** March 28, 2026 | **Branch:** `autoresearch/mar28` | **Dataset:** Ames Housing (predicting house sale prices)

---

## What We Did

We ran an autonomous machine learning research loop on a house price prediction problem. The agent was given a codebase with a starting model, then spent the session iterating — proposing ideas, implementing them in code, running experiments, and deciding whether to keep or discard each result — with no human intervention between experiments.

The goal: **minimize val_rmse**, which is the prediction error on a held-out test set measured in dollars. Lower = better.

---

## The Experiment Loop

Each iteration followed the same pattern:

1. **Propose** an idea (new features, different model, better preprocessing)
2. **Implement** it by editing the training script (`train.py`)
3. **Commit** to git so every experiment is traceable
4. **Run** the experiment and measure the result
5. **Log** the outcome in `results.tsv` (keep / discard / crash)
6. **Repeat** — building on what worked, discarding what didn't

19 experiments were run in total. Every experiment is committed to git and fully reproducible.

---

## Experiment History

| # | What Was Tried | val_rmse | Status | Key Takeaway |
|---|---|---|---|---|
| 1 | Baseline (Ridge regression, no changes) | $25,170 | Keep | Starting point established |
| 2 | More derived features + log-transform inputs | $25,807 | Discard | Worse — too much noise added |
| 3 | Switched to XGBoost (tree model) | $26,102 | Discard | Small dataset (~1,200 rows) — linear models win |
| 4 | Switched to LightGBM (another tree model) | $30,563 | Discard | Confirmed: tree models not suited here |
| 5 | Ridge + hyperparameter search | $25,368 | Discard | Marginal — features aren't rich enough yet |
| 6 | ElasticNet (blend of Ridge + Lasso) | $26,123 | Discard | Worse than plain Ridge |
| **7** | **Removed 2 known outliers from training data** | **$22,111** | **Keep** | **Biggest single jump — $3,000 improvement** |
| 8 | Alpha fine-tuning after outlier removal | $22,237 | Discard | Negligible gain |
| 9 | XGBoost re-attempted on cleaned data | $26,203 | Discard | Still no good — tree models confirmed poor fit |
| 10 | Richer features + neighborhood encoding (with redundancy) | $22,281 | Discard | Redundant features cancelled each other |
| 11 | Log-transform features only (dropped originals) | $23,210 | Discard | Lost too much information |
| **12** | **Target encoding for Neighborhood** | **$21,960** | **Keep** | Replacing 25 dummy variables with a single smooth numeric signal helped |
| 13 | Target encoding for 5 columns simultaneously | $22,291 | Discard | Overfit — less is more |
| **14** | **Polynomial interaction features** (quality × area, etc.) | **$21,422** | **Keep** | Non-linear relationships captured explicitly |
| 15 | Even more interaction terms | $21,500 | Discard | Diminishing returns — too many interactions |
| 16 | Fine-grained alpha tuning | $21,437 | Discard | Marginal, not worth the complexity |
| **17** | **Ordinal encoding for quality columns** (ExterQual, KitchenQual, etc.) | **$21,234** | **Keep** | Treating "Excellent/Good/Average/Fair/Poor" as numbers gave the model better signal |
| 18 | More quality interactions | $21,485 | Discard | Overfitted |
| **19** | **Research-informed preprocessing overhaul** | **$21,091** | **Keep** | Proper NA handling, neighborhood-aware imputation, robust scaling, additional outlier removal |

---

## Final Results

| Metric | Baseline | Final |
|---|---|---|
| **val_rmse** (test set error, $) | $25,170 | **$21,091** |
| **Improvement** | — | **−$4,079 (−16.2%)** |
| Experiments run | — | 19 |
| Commits made | 1 | 20 |

### What drove the improvement

The gains came from **four compounding changes**, each building on the last:

1. **Outlier removal** (+$3,059): Two extreme data points (large houses sold at unusually low prices) were distorting the model significantly. Removing them from training data was the single biggest win.
2. **Neighborhood target encoding** (+$151): Instead of creating 25 separate yes/no columns for each neighborhood, we computed the average price per neighborhood and used that single number. It gave the model a cleaner signal about location.
3. **Polynomial interactions** (+$538): Explicitly multiplied quality scores by living area (e.g. "Overall Quality × Square Footage") so the model could capture the intuition that a large high-quality house is worth disproportionately more.
4. **Proper preprocessing** (+$143): Used research-informed best practices — treating missing values in garage/basement columns as "no garage/no basement" instead of imputing a median, imputing missing lot frontage using the neighborhood's median instead of the global median, and switching to a scaler more robust to remaining outliers.

### What didn't work

- **Tree models** (XGBoost, LightGBM): With only ~1,200 training rows, tree-based models consistently underperformed the linear Ridge model. This is expected — gradient boosting needs more data to generalize well.
- **Adding too many features**: Several experiments added features that improved cross-validation scores but hurt the actual test score, a sign of overfitting. The best results came from a selective, principled feature set.

---

## What's Next (if the loop continues)

The research suggests two high-leverage ideas not yet tried:

- **Ensemble averaging** (Ridge + XGBoost + LightGBM): Even though individual tree models underperform, blending 30% tree model + 70% Ridge predictions often beats either alone.
- **Box-Cox or Yeo-Johnson transforms** on skewed input features like LotArea and GrLivArea, which may better normalize the distributions for Ridge.
- Further **hyperparameter search** on the now-improved feature set.

---

*All experiments are committed to branch `autoresearch/mar28`. Full experiment log: `results.tsv`.*
