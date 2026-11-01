# Restaurant Liquidation Risk Modelling

A data science project that asks whether public data can flag UK hospitality businesses — restaurants, cafés, takeaways, pubs and bars — that are heading for failure.

We combined **Companies House** company records, **Food Standards Agency (FSA)** hygiene ratings, **Google Places** ratings and reviews, and **police crime data by postcode** for **24,748 businesses**, scored about **120,000 Google reviews** with a zero-shot language model, and compared several classification models on a heavily imbalanced target (about 5% of businesses in distress).

---

## Key findings

- **Filing behaviour is the strongest signal.** In the final dataset, 81% of distressed businesses were overdue with their Companies House confirmation statement (13% of active ones), and 65% were overdue with their accounts (6.5% of active ones).
- **That signal is close to the outcome itself.** Most distressed companies are ones Companies House has proposed to strike off, which usually follows missed filings. With the overdue flags, logistic regression reaches a test ROC-AUC of about 0.93; without them (the "stress test"), the best model reaches about 0.77. We treat the stress-test results as the honest measure of what the other features can predict.
- **Customer reviews barely separate the two groups.** Average Google rating was 4.27 for distressed businesses against 4.36 for active ones, and the seven review-sentiment scores differed by less than 0.03.
- **Older companies were more often in distress** (median age 64 months against 31 months), and **pubs, bars and nightclubs** had the highest distress rate (about 10%, double the average).

## Results (held-out 20% test set, 4,950 businesses, 245 distressed)

| Model | Features | Test ROC-AUC | Precision* | Recall* | F1* |
|---|---|---|---|---|---|
| Logistic regression | All, incl. overdue flags | 0.930 | 0.79 | 0.56 | 0.66 |
| Logistic regression, class-weighted | All, incl. overdue flags | 0.929 | 0.24 | 0.87 | 0.37 |
| Logistic regression | Stress test (no overdue flags) | 0.740 | 1.00 | 0.01 | 0.02 |
| Logistic regression, class-weighted | Stress test | 0.742 | 0.09 | 0.68 | 0.17 |
| Logistic regression + SMOTE | Stress test | 0.740 | 0.18 | 0.33 | 0.24 |
| Random forest + SMOTE + undersampling | Stress test | **0.773** | 0.46 | 0.10 | 0.16 |
| XGBoost + SMOTE | Stress test | 0.732 | 0.48 | 0.16 | 0.24 |
| Neural network (Keras) + SMOTE | Stress test | 0.709 | 0.17 | 0.31 | 0.22 |

\* For the distressed class, at the default 0.5 threshold. Full outputs, confusion matrices and ROC curves are in `Code/Modelling.ipynb`.

Without balancing, models predict "survives" for almost everyone (95% accuracy, near-zero recall). Class weighting, SMOTE and undersampling trade precision for recall in different ways; the tree models overfit noticeably (training AUC 0.97–1.00 against 0.73–0.77 on test).

## Extended analysis

`Code/Extended_Analysis.ipynb` builds on `Modelling.ipynb` with the same data and features. It reproduces the headline result (logistic regression, test AUC 0.930) and then adds:

| Addition | Result |
|---|---|
| Stratified 5-fold cross-validation, imputation and SMOTE inside each fold | Random forest AUC 0.770 ± 0.024; XGBoost 0.758 ± 0.026 |
| Threshold tuned on out-of-fold predictions | XGBoost F1 0.22 → 0.27 at threshold 0.24; distressed businesses caught 36 → 73 of 245 |
| SHAP (XGBoost, stress test) | Company age, "no accounts filed" status, Google rating and restaurant density in the postcode are the main drivers |

---

## Data sources

| Source | What it provided |
|---|---|
| [Companies House Basic Company Data](https://download.companieshouse.gov.uk/en_output.html) (1 Feb 2026 snapshot) | Status, SIC code, incorporation date, filing due dates, account category, mortgage charges, previous names |
| [Companies House API](https://developer.company-information.service.gov.uk/) | Filing history, used to recover previous registered addresses of non-active companies |
| [FSA Food Hygiene Rating Scheme](https://ratings.food.gov.uk/open-data) | Rating, hygiene, structural and confidence-in-management scores, business type |
| Google Places API | Overall rating, price level, up to 5 reviews per business |
| [data.police.uk](https://data.police.uk/) | Crime per postcode |

## Pipeline

Run the notebooks in `Code/` in this order (details in [`Code/README.md`](Code/README.md)):

1. `DataFileLoader.ipynb` — Companies House download, hospitality filter, postcode validation, fuzzy matching to FSA ratings
2. `rating_gathering.ipynb` — Google Places ratings and reviews
3. `Combiner.ipynb` and `Query_03_06_26.ipynb` — merge review batches, re-query missing ones
4. `Ultimate_Combiner.ipynb` — final review set joined to the master dataset
5. `Postcode_crime_lookup.ipynb` — crime per postcode
6. `Sentiment_Analysis.ipynb` — zero-shot sentiment scoring of reviews
7. `Modelling.ipynb` — feature engineering, imputation, modelling and evaluation
8. `Extended_Analysis.ipynb` — cross-validation, threshold tuning and SHAP

## Target definition

`Target Value = 1` ("distressed") covers Companies House status *Active – Proposal to Strike off* (1,194), *Liquidation* (29) and *Voluntary Arrangement* (2): 1,225 of 24,748 businesses (4.95%). `Target Value = 0` is *Active*.

## Limitations

- **Target and leakage.** Most positives are strike-off proposals, which Companies House typically issues after missed filings, so the overdue flags are close to the label. We therefore report stress-test results without them.
- **Different matching paths by class.** Non-active companies needed an extra Companies House API step to find their old address before matching to FSA data, so the two classes were not collected in exactly the same way.
- **Single snapshot.** Features and outcome come from the same point in time, so the model describes rather than forecasts. A forward-looking version would build features from one snapshot and measure outcomes 12 months later.
- **One train/test split** and the default 0.5 threshold in `Modelling.ipynb` (cross-validation and threshold tuning are in `Extended_Analysis.ipynb`).
- **Google matching** takes the top search result for name and postcode, and only businesses with a Google rating are in the final dataset.

## Possible extensions

- A survival model (e.g. Cox proportional hazards) to estimate *when* a business is likely to fail, not just *whether*
- Time-based validation using two Companies House snapshots
- Area-level deprivation (IMD) by postcode

## Setup

```bash
pip install pandas numpy requests openpyxl rapidfuzz tqdm scikit-learn imbalanced-learn xgboost tensorflow scikeras transformers datasets torch matplotlib seaborn shap
```

You need your own **Google Places API key** (`rating_gathering.ipynb`) and **Companies House API key** (`DataFileLoader.ipynb`). The sentiment, XGBoost and neural-network steps were run on GPU in Google Colab.

## Licence

See [LICENSE](LICENSE).
