# Google Review Sentiment & Classification Pipeline

This repository contains the code and data processing pipeline for an end-to-end data science project. The workflow focuses on gathering business data (specifically restaurant and hospitality venues), extracting Google Reviews via API, processing text for sentiment analysis, and building classification models to handle heavily skewed target variables. 

## 📁 Repository Structure

The project is divided into several Jupyter Notebooks, categorized by their role in the pipeline:

### 1. Data Gathering & Extraction
* **`DataFileLoader_Group 1.ipynb`**: Handles the initial download and extraction of the base dataset (e.g., Companies House data) to establish the foundational business records.
* **`rating_gathering.ipynb`**: Connects to the Google Places API to scrape overall ratings, price levels, and individual text reviews for the gathered entities.
* **`Query_03_06_26.ipynb`**: Manages specific data queries and deduplication steps for intermediate datasets retrieved during the API extraction phase.

### 2. Data Merging & Feature Engineering
* **`Postcode_crime_lookup.ipynb`**: Integrates external datasets by mapping business postcodes to local crime probability statistics, engineering new geographical features.
* **`Combiner.ipynb`**: Merges intermediate data batches (e.g., concatenating dataframes of 10,000 records at a time) and handles missing values.
* **`Ultimate Combiner.ipynb`**: The final consolidation script that brings together the base records, the API-gathered reviews, and the postcode features into a single master dataset (`Merged_Final_Dataset.xlsx`).

### 3. Natural Language Processing (NLP)
* **`Sentiment_Analysis.ipynb`**: Processes the raw text from `Google_Review_Texts`.This notebook parses the review strings and calculates sentiment scores for 
"Food and Beverage Quality","Customer Service","Delivery & Order Accuracy",
"Value for Money", "Ambiance & Atmosphere", "Cleanliness & Hygiene", "Facilities & Amenities"
 generating the final structured dataset required for predictive modeling.

### 4. Predictive Modeling
* **`Group_Model 1.ipynb`**: These notebooks contain the core classification models. Given the significant class imbalance in the dataset, the modeling phase incorporates techniques like SMOTE (Synthetic Minority Over-sampling Technique) combined with deep learning architectures (Keras/TensorFlow Neural Networks) and ensemble methods like XGBoost to optimize the F1 score.

## 🚀 Pipeline Execution Order

To replicate the study or re-run the pipeline, notebooks should be executed in the following chronological order:
1. `DataFileLoader_Group 1.ipynb`
2. `rating_gathering.ipynb`
3. `Query_03_06_26.ipynb` & `Combiner.ipynb` 
4. `Postcode_crime_lookup.ipynb`
5. `Ultimate Combiner.ipynb`
6. `Sentiment_Analysis.ipynb`
7. `Group_Model 1.ipynb`

## 🛠️ Technologies Used
* **Data Manipulation:** `pandas`, `numpy`
* **APIs & Web:** `requests`, Google Places API
* **NLP:** Text processing and sentiment classification libraries
* **Machine Learning:** `scikit-learn`, `imbalanced-learn` (SMOTE), `xgboost`
* **Deep Learning:** `TensorFlow` / `Keras` (Custom NN models)

## ⚙️ Setup & Installation
Ensure you have the required dependencies installed before running the notebooks. It is recommended to use an isolated environment (like a Conda environment).

```bash
pip install pandas numpy requests scikit-learn imbalanced-learn tensorflow xgboost openpyxl

Note: You will need to provide your own Google API key in rating_gathering.ipynb for the data extraction steps to function correctly.
