![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

# Google Review Sentiment & Restaurant Liquidation Risk Modelling
## End-to-End Classification Pipeline — Companies House · Google Places API · ONS IMD 2019
### Companies House Open Data · Google Places API · ONS Open Geography Portal

> **Key Finding:** Accounts filing overdue status is the single strongest predictor of restaurant liquidation — 64.2% of liquidated businesses had overdue filings compared to just 6.5% of active businesses, a 10× difference. Counterintuitively, liquidated businesses are on average 15 months *older* than active ones, challenging the assumption that newly-formed restaurants are most at risk. Google Review sentiment scores across all 7 dimensions showed less than 0.05 separation between classes, confirming that operational compliance signals dominate over customer perception signals in predicting business failure.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Datasets Used](#datasets-used)
3. [Repository Structure](#repository-structure)
4. [Pipeline Execution Order](#pipeline-execution-order)
5. [Methodology](#methodology)
6. [Analysis and Findings](#analysis-and-findings)
7. [Conclusions](#conclusions)
8. [Limitations](#limitations)
9. [Technologies Used](#technologies-used)
10. [Setup & Installation](#setup--installation)
11. [Author](#author)

---

## Project Overview

This project builds an end-to-end machine learning pipeline to predict the risk of restaurant and hospitality business liquidation in England, using a multi-source feature set drawn from Companies House records, Google Reviews sentiment analysis, FSA food hygiene inspection scores, postcode-level crime statistics, and the ONS Index of Multiple Deprivation (IMD 2019).

The dataset spans 25,268 hospitality businesses filtered by SIC code from the Companies House bulk snapshot. Given a significant class imbalance — only 5% of businesses in the dataset are liquidated — the modelling phase employs SMOTE oversampling combined with ensemble methods (Random Forest, XGBoost) and a custom Keras Neural Network, with threshold tuning and SHAP-based feature explanation to produce interpretable, actionable predictions.

The project introduces several derived features and metrics not present in the raw datasets, including a filing overdue flag, company age in months, name-change frequency categorisation, a stop-to-prosecution efficiency ratio, and postcode-level IMD deprivation deciles — all of which deepen the predictive analysis beyond what the raw sources provide.

---

## Datasets Used

| Dataset | Source | Coverage | Key Fields |
|---|---|---|---|
| Companies House Bulk Snapshot | [download.companieshouse.gov.uk](https://download.companieshouse.gov.uk) | All registered companies in England & Wales | Company status, SIC code, incorporation date, filing dates, mortgage charges |
| Google Places API | Google Cloud Platform | Live at time of extraction | Overall rating, price level, up to 5 review texts per business |
| FSA Food Hygiene Ratings | [ratings.food.gov.uk](https://ratings.food.gov.uk) | All rated hospitality venues in England | Hygiene score, Structural score, Confidence in Management score, Rating value |
| Postcode Crime Statistics | [data.police.uk](https://data.police.uk) | England & Wales | Crime probability per postcode, number of restaurants per postcode |
| ONS Index of Multiple Deprivation 2019 | [gov.uk](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019) | All LSOAs in England (32,844 areas) | IMD decile, IMD score, Employment score, Income score |
| ONS Postcode–LSOA Lookup | [ONS Open Geography Portal](https://geoportal.statistics.gov.uk) | All postcodes in England & Wales | Postcode → LSOA mapping |

> The master dataset (`Merged_Final_Dataset_With_Sentiment.xlsx`) is not included in this repository due to file size. Follow the pipeline execution order below to reproduce it from source.

---

## Repository Structure

```
├── Code/
│   ├── DataFileLoader_Group.ipynb
│   ├── rating_gathering.ipynb
│   ├── Query_03_06_26.ipynb
│   ├── Postcode_crime_lookup.ipynb
│   ├── Combiner.ipynb
│   ├── Ultimate_Combiner.ipynb
│   ├── Sentiment_Analysis.ipynb
│   └── Group_Model.ipynb
├── Data/
│   ├── Final Dataset.xlsx               ← output of DataFileLoader
│   ├── Merged_Final_Dataset.xlsx        ← output of Ultimate_Combiner
│   └── Merged_Final_Dataset_With_Sentiment.xlsx  ← output of Sentiment_Analysis
└── README.md
```

---

## Pipeline Execution Order

To replicate the study or re-run the pipeline, notebooks should be executed in the following chronological order:

1. `DataFileLoader_Group.ipynb` — Download and filter Companies House snapshot; merge FSA hygiene ratings
2. `rating_gathering.ipynb` — Scrape Google Places ratings and review texts via API
3. `Query_03_06_26.ipynb` & `Combiner.ipynb` — Deduplicate and batch-merge intermediate review datasets
4. `Postcode_crime_lookup.ipynb` — Map postcode crime statistics and restaurant density features
5. `Ultimate_Combiner.ipynb` — Final consolidation of all sources + IMD 2019 enrichment
6. `Sentiment_Analysis.ipynb` — BART zero-shot sentiment classification + EDA
7. `Group_Model.ipynb` — Feature engineering, modelling, evaluation, SHAP analysis, conclusions

---

## Methodology

1. **Data extraction:** Companies House bulk CSV was downloaded programmatically and filtered to hospitality SIC codes (`56101`, `56102`, `56103`, `56210`, `56290`, `56301`, `56302`). Duplicate trading names were resolved by retaining the most recently incorporated entity per name.

2. **API enrichment:** Google Places API was queried for each business using name and postcode. Up to 5 review texts and an overall star rating were retrieved per business. Businesses returning `N/A` from the first pass were routed back for re-querying via `Combiner.ipynb`.

3. **Sentiment analysis:** Raw review texts were exploded from pipe-delimited strings into individual reviews and classified using `facebook/bart-large-mnli` zero-shot classification across 7 dimensions: Food and Beverage Quality, Customer Service, Delivery & Order Accuracy, Value for Money, Ambiance & Atmosphere, Cleanliness & Hygiene, and Facilities & Amenities. Scores were aggregated to business level by mean.

4. **IMD enrichment:** Business postcodes were mapped to LSOA codes via the ONS postcode lookup and joined to IMD 2019 scores and deciles, adding socioeconomic deprivation context at neighbourhood level.

5. **Feature engineering:** Derived features included filing overdue flag (accounts due date vs. reference date), company age in months, previous name count, SIC code count, and tenure category (frequent/moderate/stable). Categorical variables were one-hot encoded. KNN imputation (k=5, distance-weighted) was applied to numeric features.

6. **Class imbalance:** The target variable (liquidated = 1) represented only ~5% of records. SMOTE oversampling (`sampling_strategy=0.2`) combined with random undersampling (`sampling_strategy=0.5`) was applied to the training set only, with all resampling done inside cross-validation to prevent data leakage.

7. **Modelling:** Four model families were benchmarked: Logistic Regression (baseline + class-weighted), Random Forest + SMOTE/undersampling, XGBoost + SMOTE, and a custom Keras Neural Network + SMOTE. All models were evaluated on a held-out 20% test set using AUC, F1, Precision, Recall, and Average Precision.

8. **Threshold tuning:** The default 0.5 classification threshold was evaluated against the optimal F1 threshold via a full precision-recall sweep, addressing the sub-optimality of the default threshold under severe class imbalance.

9. **Interpretability:** SHAP (SHapley Additive exPlanations) values were computed for the XGBoost model to produce a ranked, directional explanation of feature contributions to individual predictions.

---

## Analysis and Findings

### 1. Class Imbalance

The dataset contains 25,268 businesses of which approximately 5% (≈1,266) are liquidated. This severe imbalance means a naive classifier predicting all-active achieves ~95% accuracy while catching zero liquidations. SMOTE and threshold tuning were essential to produce a model with meaningful recall for the minority class.

---

### 2. Accounts Filing Overdue — Strongest Predictor

**64.2%** of liquidated businesses had overdue accounts filings at the time of analysis, compared to just **6.5%** of active businesses — a **10× difference** and the single largest discriminating signal in the dataset.

This finding has a clear real-world interpretation: businesses approaching failure tend to stop meeting their statutory filing obligations before they formally enter liquidation. The overdue flag is an early-warning signal that precedes the formal legal event.

---

### 3. Business Type Risk

Liquidation rates vary substantially by business type when controlling for base rates:

- **Pubs and bars** have the highest liquidation rate at approximately **9.9%** — nearly double the dataset average of ~5%
- **Takeaways and fast food** sit at **4.3%**, below the sector mean
- **Restaurants** (sit-down) at **4.8%**, close to the mean

The pub and bar sector faces structural headwinds — higher fixed costs, changing drinking habits, and greater sensitivity to discretionary spending — that are reflected in its elevated failure rate relative to the broader hospitality sector.

---

### 4. Company Age — Counterintuitive Finding

Liquidated businesses are on average **15 months older** than active ones (68.6 months vs. 53.5 months at time of analysis). This challenges the common assumption that newly-formed restaurants are most vulnerable.

A plausible interpretation is survivorship bias in the active cohort — very recently incorporated businesses have not yet had sufficient time to accumulate the filing history, debt obligations, and operational pressures that precede liquidation. Failure risk in the restaurant sector may peak in the 3–7 year operating window rather than in the first year.

---

### 5. Hygiene Inspection Scores

Liquidated businesses scored modestly worse (higher penalty scores = poorer standards) across all three FSA inspection dimensions:

| Dimension | Active (mean) | Liquidated (mean) | Δ |
|---|---|---|---|
| Hygiene | lower | higher | positive |
| Structural | lower | higher | positive |
| Confidence in Management | lower | higher | positive |

While the differences are modest in absolute terms, they are consistent in direction and support the inclusion of hygiene scores as predictive features. Deteriorating hygiene standards may be a lagging indicator of broader operational and managerial decline.

---

### 6. IMD Deprivation Index

After joining the ONS IMD 2019 data at postcode level, liquidated and active businesses show a measurable difference in the socioeconomic deprivation of their operating areas. Businesses in more deprived neighbourhoods (lower IMD decile) face higher liquidation risk, consistent with lower consumer spending power and higher operating environment instability in those areas.

The IMD Crime domain score also partially overlaps with the postcode crime feature already in the dataset, allowing cross-validation of those signals.

---

### 7. Google Review Sentiment — Limited Discriminating Power

All 7 BART zero-shot sentiment dimensions showed less than **0.05 difference** in mean score between liquidated and active businesses. The Google overall star rating gap was similarly small: **4.27** for liquidated vs. **4.36** for active businesses.

This is a significant and honest finding: **customer-facing sentiment signals do not reliably predict business failure** in this dataset. Restaurants that ultimately liquidate receive broadly similar Google reviews to those that remain active. Financial compliance signals (filing overdue, company age, mortgage charges) carry substantially more predictive weight.

The one exception is **Cleanliness & Hygiene** sentiment, which is the only dimension that directionally aligns with the liquidation target, consistent with the FSA inspection score findings above.

---

### 8. Model Comparison

All four model families were evaluated on the held-out test set. Results are printed in the conclusions cell of `Group_Model.ipynb` after running. Key comparative observations:

- **XGBoost + SMOTE** achieves the highest Test AUC and Average Precision, benefiting from its ability to model non-linear feature interactions and its native handling of feature importance
- **Logistic Regression (weighted)** provides a strong, interpretable baseline and remains competitive on AUC despite its linearity assumption
- **Stress-test results** (models trained without the filing overdue flag) confirm that the remaining features — hygiene scores, company age, sentiment, crime, IMD — retain meaningful predictive signal even after removing the dominant predictor
- The **Neural Network** was trained on GPU in Colab and is preserved in the notebook but its output metrics are not reported here as they are runtime-dependent

---

### 9. Threshold Tuning

At the default classification threshold of 0.5, minority-class recall is suppressed by the class imbalance. Sweeping the decision threshold across the full precision-recall curve identifies an optimal F1 threshold substantially below 0.5, recovering a meaningful improvement in F1 score and recall while accepting a controlled increase in false positives.

The threshold analysis cell in `Group_Model.ipynb` prints:
- F1 at default threshold (0.50)
- F1 at optimal threshold
- Percentage improvement from threshold tuning
- Precision and recall at the optimal threshold, with a plain-language interpretation

---

### 10. SHAP Feature Importance

SHAP values for the XGBoost model confirm the EDA findings quantitatively and provide directional information:

- Filing overdue status, company age, and hygiene scores rank among the top predictors by mean |SHAP| value
- Google sentiment dimension scores rank lower, consistent with their weak separation between classes
- IMD decile and crime features provide additional signal, with higher deprivation (lower decile) pushing predictions toward liquidation
- The SHAP summary plot and bar chart are saved to `Data/` as `shap_summary.png` and `shap_bar.png`

---

## Conclusions

1. **Filing overdue status dominates.** A 10× difference in overdue filing rates between liquidated and active businesses makes this the single most actionable early-warning signal in the dataset. Monitoring statutory compliance is more predictive than monitoring customer reviews.

2. **Older businesses are more at risk, not newer ones.** Liquidated businesses are on average 15 months older. Failure risk in the restaurant sector appears to peak in the 3–7 year operating window, not at launch.

3. **Pubs and bars are highest risk.** At nearly 10% liquidation rate — double the sector average — the pub sector warrants distinct treatment in any risk model rather than being grouped with restaurants and takeaways.

4. **Sentiment signals are not reliable liquidation predictors.** All 7 review dimensions show <0.05 separation between classes. Customer perception and business viability are largely decoupled in this data. Financial compliance signals are more informative than NLP features for this task.

5. **SMOTE and threshold tuning are both necessary.** With only 5% positive class prevalence, neither SMOTE alone nor default threshold classification produces adequate recall. The combination — resampling during training and threshold optimisation at inference — meaningfully improves minority class detection.

6. **Socioeconomic context adds signal.** IMD deprivation decile, joined via postcode, provides additional predictive power beyond operational features alone. Restaurants in more deprived areas face higher failure risk, independent of their review scores or hygiene ratings.

7. **XGBoost + SMOTE is the strongest performer.** It outperforms logistic regression and random forest on AUC and Average Precision, and its SHAP-based interpretability makes it the recommended model for deployment.

---

## Limitations

- Google Reviews coverage is incomplete — not all businesses in the dataset had retrievable ratings, meaning sentiment features are missing for a subset of records. KNN imputation was applied but introduces uncertainty.
- The Companies House snapshot represents a single point in time. Businesses that dissolved before the snapshot date are not captured, introducing survivorship bias into the active cohort.
- IMD 2019 data may not fully reflect current deprivation conditions, particularly given post-COVID economic shifts in urban restaurant districts.
- The dataset is limited to hospitality SIC codes. Results should not be generalised to other business sectors without revalidation.
- The Neural Network component was trained on GPU in Google Colab and its performance metrics are not reproduced here. CPU replication is possible but substantially slower.
- Geographic coverage is England-dominant. Scottish postcodes (PA, FK prefixes) are present in the dataset but may not match the England-only IMD lookup, reducing IMD coverage for those records.

---

## Technologies Used

| Library / Tool | Purpose |
|---|---|
| `pandas`, `numpy` | Data loading, cleaning, merging, feature engineering |
| `requests` | Companies House download, Google Places API calls, IMD data retrieval |
| `Google Places API` | Rating and review text extraction per business |
| `transformers` (Hugging Face) | `facebook/bart-large-mnli` zero-shot sentiment classification |
| `scikit-learn` | Logistic Regression, Random Forest, KNN Imputation, MinMaxScaler, train/test split, evaluation metrics |
| `imbalanced-learn` | SMOTE oversampling, RandomUnderSampler |
| `xgboost` | Gradient boosted classification with GPU support |
| `TensorFlow` / `Keras` | Custom Neural Network with BatchNorm and Dropout |
| `shap` | SHAP values for XGBoost feature explanation |
| `matplotlib`, `seaborn` | EDA visualisations, model evaluation plots |
| `openpyxl` | Excel file read/write |

---

## Setup & Installation

Ensure you have the required dependencies installed before running the notebooks. It is recommended to use an isolated environment (like a Conda environment) or Google Colab with a GPU runtime for the Neural Network and XGBoost cells.

```bash
pip install pandas numpy requests scikit-learn imbalanced-learn tensorflow xgboost shap transformers datasets openpyxl matplotlib seaborn
```

Or save the following as `requirements.txt` and run `pip install -r requirements.txt`:

```
pandas
numpy
requests
scikit-learn
imbalanced-learn
tensorflow
xgboost
shap
transformers
datasets
openpyxl
matplotlib
seaborn
```

> **Note:** A valid Google Places API key is required in `rating_gathering.ipynb` for the data extraction steps to function. Set `API_KEY = "YOUR_GOOGLE_PLACES_API_KEY"` in the configuration cell before running.

> **Note:** The IMD 2019 and postcode–LSOA lookup files are downloaded automatically in `Ultimate_Combiner.ipynb` via public government URLs. No manual download is required.

> **GPU:** The XGBoost model uses `device='cuda'` and the Neural Network uses TensorFlow GPU. Both will fall back to CPU automatically if no GPU is available, but runtime will be significantly longer.

---



