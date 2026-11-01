# Code

Notebooks are listed in the order they were run. Saved outputs are kept so results can be read without re-running.

### 1. `DataFileLoader.ipynb` — build the business list and match hygiene ratings

- Downloads the Companies House bulk snapshot (5.68 million companies) and keeps seven hospitality SIC codes (56101, 56102, 56103, 56210, 56290, 56301, 56302): 246,706 companies.
- Cleans company names (removes quotes, brackets and suffixes like LTD) and validates UK postcodes with a rule-based checker.
- Downloads the FSA ratings file (about 600,000 premises) and, for each company, lists the FSA premises at the same postcode.
- **Active companies:** matched on the first line of the address, using company-name similarity (`rapidfuzz` token-set ratio) to choose between premises that share an address.
- **Non-active companies:** name similarity ≥ 90 first; otherwise the Companies House API filing history is used to recover the company's previous registered address (AD01 filings), which is then matched to FSA premises. Run in parallel (10 threads) within the API rate limit.
- Where several companies matched the same FSA premises, keeps the most recently incorporated one.
- Output: 25,268 matched businesses (5.0% distressed), merged with their FSA ratings → `Final Dataset`.

### 2. `rating_gathering.ipynb` — Google Places

For each business, a Text Search on name + postcode returns a `place_id`; Place Details then returns rating, price level and up to five reviews with their star ratings.

### 3. `Combiner.ipynb` and `Query_03_06_26.ipynb` — merge and re-query

Google results were collected in batches. These notebooks concatenate the batches, separate businesses without a rating, and send them back for a second query.

### 4. `Ultimate_Combiner.ipynb` — final review set

Combines all review batches (24,748 businesses with a Google rating), zero-pads company numbers to 8 characters and left-joins onto the master dataset → `Merged_Final_Dataset.xlsx`.

### 5. `Postcode_crime_lookup.ipynb` — crime

Maps a postcode-level crime figure onto each business.

### 6. `Sentiment_Analysis.ipynb` — review sentiment

Splits the pipe-separated reviews into 119,932 individual reviews and scores each with `facebook/bart-large-mnli` zero-shot classification (multi-label, GPU, batches of 64, progress saved every 10,000 rows) against seven labels: Food and Beverage Quality, Customer Service, Delivery & Order Accuracy, Value for Money, Ambiance & Atmosphere, Cleanliness & Hygiene, Facilities & Amenities. Scores are averaged per business.

### 7. `Modelling.ipynb` — features and models

- **Features:** company age in months, number of previous names, number of extra SIC codes, name-change tenure category, average review star rating, overdue flags for confirmation statement and accounts (reference date 9 March 2026), one-hot encoded categories, FSA rating mapped to numbers (Scottish Pass = 4, Pass and Eat Safe = 5, Improvement Required = 1).
- **Missing values:** MinMax scaling + KNN imputation (k = 5, distance-weighted).
- **Split:** stratified 80/20.
- **Models:** logistic regression (unweighted and class-weighted), random forest (unbalanced and with SMOTE + random undersampling), logistic regression with SMOTE, SMOTE + undersampling and SMOTE + Tomek links, XGBoost + SMOTE, Keras neural network + SMOTE.
- **Stress test:** models re-trained without the two overdue flags.
- **Evaluation:** train/test accuracy and ROC-AUC, overfit gap, classification report, confusion matrix and ROC curve; logistic regression coefficients and XGBoost gain-based feature importance.

### 8. `Extended_Analysis.ipynb` — extended analysis

Builds on `Modelling.ipynb`, on CPU, with the same data and feature engineering. It reproduces the logistic regression result (test AUC 0.930), then:

- runs stratified 5-fold cross-validation with imputation and SMOTE inside each fold (random forest 0.770 ± 0.024, XGBoost 0.758 ± 0.026),
- tunes the decision threshold on out-of-fold predictions (XGBoost F1 0.22 → 0.27 on the test set),
- explains the XGBoost model with SHAP.
