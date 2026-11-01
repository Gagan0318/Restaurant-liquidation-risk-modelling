# Data

| File | Rows | Description |
|---|---|---|
| `Final Dataset.csv` | 25,268 | Hospitality companies from Companies House matched to FSA hygiene ratings (output of `DataFileLoader.ipynb`) |
| `Merged_Final_Dataset_With_Sentiment.xlsx` | 24,748 | Final modelling table: the above plus Google ratings and reviews, crime per postcode and the seven review-sentiment scores |

**Target:** `Target Value` = 1 for *Active – Proposal to Strike off*, *Liquidation* or *Voluntary Arrangement*; 0 for *Active*.

Sources: Companies House (Open Government Licence), FSA Food Hygiene Rating Scheme open data, data.police.uk, and Google Places API (review text is subject to Google's terms).
