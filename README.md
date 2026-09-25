# Complex Data Manipulation, Model Fitting, and Evaluation — Fraud Transaction Detection

A machine learning pipeline to detect potentially fraudulent credit card transactions, built on a large, messy multi-table dataset that required substantial cleaning, feature engineering, and imbalanced-class handling before model fitting.

## Contents

| File | Description |
|---|---|
| `Fraud Transaction Detection.ipynb` | Notebook containing data loading, cleaning, feature engineering, class-imbalance handling, model training, and evaluation. |
| `README.md` | This file. |

## Data

Three related CSVs from a Kaggle credit card transaction dataset (IBM synthetic data):

* **`credit_card_transactions-ibm_v2.csv`** — ~24.4 million transactions with 15 columns (user, card, date/time, amount, merchant details, error codes, and the `Is Fraud?` label).
* **`sd254_cards.csv`** — 6,146 card records (brand, type, credit limit, dark-web status, etc.).
* **`sd254_users.csv`** — 2,000 user/cardholder records (demographics, income, FICO score, etc.).

## How It Works

1. **Loading & inspection**: All three datasets are loaded and inspected column by column, checking the number of unique values to decide which columns need cleaning versus which are already usable.
2. **Feature cleaning**:
   * `Merchant State` has ~2.7M missing values, all traced to online orders (`Merchant City == 'ONLINE'`) — these are filled in as `'ONLINE'`.
   * `Errors?` (a column of comma-separated error strings, e.g. `"Bad PIN,Insufficient Balance"`) is encoded into a single numeric `Error_Code` using a custom binary-flag encoding scheme, where each distinct error type contributes a power-of-10 digit if present.
   * `Use Chip` (transaction type) is label-encoded into `Use_Chip`.
   * `Time` (stored as `"HH:MM"` strings) is converted to `Time_min`, minutes since midnight.
   * `Amount` (stored as a currency string, e.g. `"$134.09"`) is stripped of the dollar sign and converted to a float.
   * The target `Is Fraud?` (`Yes`/`No`) is label-encoded to 1/0.
3. **Feature selection**: An attempt to enrich the transaction data with card- and user-level attributes (e.g. whether the card has a chip, whether it's on the dark web, whether the transaction occurred outside the cardholder's home state) was dropped, since neither the `cards` nor `users` dataset has a key that reliably joins back to the transactions table. The final feature set is: `User`, `Card`, `Year`, `Month`, `Day`, `Merchant Name`, `Zip`, `MCC`, `Use_Chip`, `Error_Code`, `Time_min`, `Amount_$`.
4. **Train/test split**: 70/30 split (`random_state=0`), giving roughly 17.1M training rows and 7.3M test rows.
5. **Handling class imbalance**: Fraudulent transactions make up only ~0.12% of the data (~20,848 of ~17.07M in the training set). To address this, 5 balanced training sets are built by combining all fraud cases with 5 different random samples of ~21,000 non-fraud cases each (different seeds per sample).
6. **Model training & evaluation**: A `RandomForestClassifier` is trained on each of the 5 balanced samples and evaluated on: its own balanced training set, the full (imbalanced) training set, and the held-out test set.

## Results

| Model | Balanced Train Score | Full Train Score | Test Score |
|---|---|---|---|
| Model 1 | 100.00% | 96.26% | 96.26% |
| Model 2 | 100.00% | 96.31% | 96.30% |
| Model 3 | 100.00% | 96.34% | 96.33% |
| Model 4 | 100.00% | 96.29% | 96.28% |
| **Model 5** | **100.00%** | **96.44%** | **96.43%** |

Model 5 achieves the best generalization performance and is selected as the final model.

## Requirements

* Python 3
* `pandas`
* `numpy`
* `scikit-learn`

Install dependencies:
```
pip install pandas numpy scikit-learn
```

## Usage

1. Place `credit_card_transactions-ibm_v2.csv`, `sd254_cards.csv`, and `sd254_users.csv` in the project root.
2. Open and run the notebook:
   ```
   jupyter notebook "Fraud Transaction Detection.ipynb"
   ```
3. Run the cells in order to reproduce cleaning, feature engineering, balanced sampling, and model evaluation.

## Notes & Limitations

* Every training run achieves 100% accuracy on its own balanced sample, which suggests the Random Forest is overfitting to that specific sample rather than perfectly separating the classes in general — the full-train and test scores (~96%) are the more meaningful numbers.
* `cards` and `users` datasets could not be joined to the transaction data (no shared primary key), so potentially useful features (chip presence, dark-web status, out-of-state transactions) were not incorporated.
* No hyperparameter tuning was performed on the `RandomForestClassifier` — default settings are used throughout.
* Given the ~0.12% fraud rate, accuracy alone can be misleading; precision, recall, and F1 on the minority (fraud) class would give a fuller picture of model quality.
