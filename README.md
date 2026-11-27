# Buying Behavior Prediction

This project asks whether a visitor to an online shop buys during a visit. Phase 1 cleans the data, looks at the patterns, and prepares the columns. Phase 2 compares five models on that split and serves the chosen ones in a small website.

**Live website:** https://buyerbehaviorprediction-48lrfsoksjxq27wuoh28gc.streamlit.app/

## Team

| Name | Registration number | GitHub username |
| --- | --- | --- |
| Ghayas Shahid | 23L-0555 | ghayasshahid |
| Burhan Shahzad | 23L-0624 | burhan-shahzad-1 |
| Syed Shahzaib Hussain | 23L-0677 | Shahzhussain |

## Problem statement

The target is `Revenue`. It is 1 if the visitor bought and 0 if they did not. The task is to predict that choice from the other columns in the visit.

Phase 1 stops at the cleaned and prepared data. Phase 2 trains the models and the website. A guess of "nobody buys" is right 84.37% of the time and finds no buyers, so the models are ranked by precision, recall, and F1, not by accuracy.

## Dataset

Source: UCI [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).

The file in this repo is `data/raw/online_shoppers_intention.csv`. It is never edited.

- Raw file: 12,330 rows and 18 columns.
- After cleaning: 12,205 rows.
- Target `Revenue`: 1,908 buyers out of 12,205 rows, which is 15.63%.

Inputs, grouped by type:

- Page counts and time on those pages: `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`.
- Scores calculated from the visit: `BounceRates`, `ExitRates`, `PageValues`.
- `SpecialDay`: a score from 0 to 1.
- `Weekend`: 0 for a weekday, 1 for a weekend.
- Text labels: `Month`, `VisitorType`.
- Number codes stored as text: `OperatingSystems`, `Browser`, `Region`, `TrafficType`.

## Repo structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   └── online_shoppers_intention.csv
│   └── processed/
│       ├── cleaned_shoppers.csv
│       ├── train_clean.csv
│       ├── test_clean.csv
│       ├── X_train_processed.csv
│       ├── X_test_processed.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       └── input_schema.json
├── images/
│   └── 18 PNG files (listed below)
├── models/
│   ├── preprocessor.joblib
│   └── feature_names.json
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   └── 04_modeling.ipynb
├── phase2/
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluate.py
│   ├── plots.py
│   ├── train.py
│   ├── models/
│   │   ├── app_pipeline.joblib
│   │   └── full_pipeline.joblib
│   └── results/
│       ├── cv_metrics.csv
│       ├── tuned_metrics.csv
│       ├── summary.json
│       ├── error_slices.csv
│       ├── feature_importance.csv
│       └── 04_*.png
└── website/
    └── app.py
```

## Cleaning summary

Cleaning is in `notebooks/01_data_cleaning.ipynb`. The cleaned table is `data/processed/cleaned_shoppers.csv`.

| Step | What the notebook found | What we did |
| --- | --- | --- |
| Month names | 288 rows were labeled `June`. The other months were already 3 letters. January and April are absent. | Renamed `June` to `Jun`. |
| Duplicate rows | 125 extra rows matched another row exactly. All 125 were non-buyers. All 125 had zero time and 1–2 pages. There is no ID column, so they were not proven to be copies. | Removed those 125 rows. This is a judgment call. Rows went from 12,330 to 12,205. Buyers stayed 1,908, so the buyer share went from 15.47% to 15.63%. |
| Code columns | `OperatingSystems`, `Browser`, `Region`, and `TrafficType` are category codes, not amounts. | Stored them as text. |
| Zero time | On the cleaned table, page count above 0 and time equal to 0: Administrative 135, Informational 226, ProductRelated 592. | Kept every one of these rows. |
| Missing values | 0 empty cells. | Nothing to fill. |
| Outliers | Counts were measured on the 12,205 cleaned rows. No row was capped or deleted. | See the two tables below. |

Real outlier counts, from the usual quartile rule. These rows were kept.

| Column | Outlier rows | Share of rows | Buyers inside that group | Plan in cleaning |
| --- | ---: | ---: | ---: | --- |
| Administrative | 404 | 3.31% | 27.23% | Keep |
| Administrative_Duration | 1,149 | 9.41% | 24.72% | Log later. Not applied here. |
| ProductRelated | 1,007 | 8.25% | 28.70% | Log later. Not applied here. |
| ProductRelated_Duration | 951 | 7.79% | 30.39% | Log later. Not applied here. |
| BounceRates | 1,428 | 11.70% | 1.75% | Keep. Values already sit between 0 and 1. |
| ExitRates | 1,325 | 10.86% | 0.60% | Keep. Values already sit between 0 and 1. |

For four columns, Q1 and Q3 are both 0. The quartile rule then only flags nonzero values. Those counts are not outliers.

| Column | Share that is 0 | Nonzero rows flagged | Share of rows |
| --- | ---: | ---: | ---: |
| Informational | 78.44% | 2,631 | 21.56% |
| Informational_Duration | 80.29% | 2,405 | 19.71% |
| PageValues | 77.63% | 2,730 | 22.37% |
| SpecialDay | 89.77% | 1,249 | 10.23% |

## Preprocessing summary

Preprocessing is in `notebooks/03_preprocessing.ipynb`. The split happens before any fit. The scaler and the category encoder are fit on the training rows only, then applied to the test rows.

- Split: 80% train, 20% test, `random_state=42`, same buyer share in both parts.
- Train: 9,764 rows, 1,526 buyers, 15.63%.
- Test: 2,441 rows, 382 buyers, 15.65%.
- `log1p` on 10 columns: `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`.
- `Weekend` stays 0 or 1.
- After the log, `StandardScaler` puts those 10 columns on a similar scale.
- `Month`, `VisitorType`, `OperatingSystems`, `Browser`, `Region`, and `TrafficType` become yes/no columns. A value seen fewer than 20 times in training is grouped with other rare values (`min_frequency=20`).
- Output: 65 columns. Train shape `(9764, 65)`. Test shape `(2441, 65)`.

On the 9,764 training rows, the log did not help every column the same way:

- It helped `Administrative` (1.9861 to 0.5578), `Administrative_Duration` (5.6769 to 0.2397), `ProductRelated` (4.3898 to -0.0434), and `PageValues` (6.5473 to 1.80). `PageValues` is still skewed at 1.80.
- It helped only partly: `Informational` (2.01) and `Informational_Duration` (1.91).
- It did almost nothing on `BounceRates`, `ExitRates`, and `SpecialDay`. Their values sit between 0 and 1. `StandardScaler` is doing the work there.
- It overshot on `ProductRelated_Duration` (skew -1.43).
- Rows were kept. Nothing was capped or deleted.

Saved files:

- `data/processed/train_clean.csv` and `data/processed/test_clean.csv` (split, before the log and the scaler)
- `data/processed/X_train_processed.csv` and `data/processed/X_test_processed.csv`
- `data/processed/y_train.csv` and `data/processed/y_test.csv`
- `data/processed/input_schema.json`
- `models/preprocessor.joblib`
- `models/feature_names.json`

## Key findings

Numbers are from the Insights section of `notebooks/02_eda.ipynb`, on the cleaned 12,205 rows.

- 15.63% of visits bought (1,908 buyers). A guess of "nobody buys" is right 84.37% of the time, so accuracy is a poor score.
- November is linked with buying: 25.49% bought (760 buyers, 2,982 rows). November is 24.43% of all rows and 39.83% of all buyers.
- New visitors are linked with a higher buying rate than returning visitors: 24.93% (422 of 1,693) versus 14.09% (1,470 of 10,431).
- `PageValues` is linked with buying. The zero bucket buys at 3.91% (n = 9,475). The four nonzero quarters buy at 36.31%, 48.68%, 62.32%, and 78.04%. Still, 370 of 1,908 buyers (19.4%) have `PageValues` = 0.
- Across the nonzero buckets, `BounceRates` is linked with a falling buying rate: 23.55%, 17.10%, 8.88%, 2.46%. `ExitRates` falls the same way: 26.10%, 20.20%, 12.29%, 3.07%.
- A nonzero `SpecialDay` score appears only in February (79 rows) and May (1,170 rows). Inside May, a score of 0 is linked with a 13.43% buying rate (290 buyers, n = 2,159) and a nonzero score with 6.41% (75 buyers, n = 1,170). February has only 3 buyers, so its split is unreliable. The all-year gap (16.71% versus 6.16%) mixes in months that never have a special-day score.
- ProductRelated pages above 0 with time equal to 0: 7 buyers in 592 rows (1.18%). Administrative zero-time rows buy at 25.19% (n = 135) and Informational zero-time rows buy at 21.24% (n = 226). The three types do not agree.
- Two pairs of inputs are tightly linked with each other: `BounceRates` with `ExitRates` (Pearson 0.90) and `ProductRelated` with `ProductRelated_Duration` (Pearson 0.86). No third pair reaches 0.70.

## Limitations

- The file is one year of visits. There is no January and no April, and there is no visitor ID.
- February has 3 buyers in 181 rows, and June has 29 buyers in 285 rows. Both are under 30 buyers, so those two month rates are unreliable.
- `PageValues`, `BounceRates`, and `ExitRates` are calculated from the visit. A shopper does not type them.
- The data alone cannot prove a leak. A leak would be a value from the same visit that gives away whether that visit already ended in a purchase.

## Graphs

Every PNG in `images/`, from `ls images`.

| File | What it shows |
| --- | --- |
| `images/02_bounce_vs_exit.png` | `BounceRates` versus `ExitRates` on the cleaned table, with a line where the two are equal. |
| `images/02_bouncerates_buckets.png` | Buying rate by `BounceRates` bucket. |
| `images/02_boxplots_by_revenue.png` | Boxplots of the key numeric columns, split by whether the visitor bought. |
| `images/02_boxplots_numeric.png` | Boxplots of the numeric columns. |
| `images/02_buying_rate_month.png` | Buying rate by month. A gray bar has under 30 buyers. |
| `images/02_buying_rate_region.png` | Buying rate by region code. A gray bar would mean under 100 rows. |
| `images/02_buying_rate_special_day.png` | Buying rate for each `SpecialDay` score from 0 to 1. |
| `images/02_buying_rate_traffic_type.png` | Buying rate by traffic code. A gray bar has under 100 rows. |
| `images/02_buying_rate_visitor_type.png` | Buying rate by visitor type. A gray bar has under 100 rows. |
| `images/02_buying_rate_weekend.png` | Buying rate for weekday (0) and weekend (1). |
| `images/02_buying_rate_zero_time.png` | Buying rate when the page count is above 0 and the time is 0. |
| `images/02_correlation_heatmap.png` | How the numeric columns move together, including `Revenue`. |
| `images/02_correlation_with_revenue.png` | Pearson and Spearman link of each numeric column with `Revenue`. |
| `images/02_exitrates_buckets.png` | Buying rate by `ExitRates` bucket. |
| `images/02_histograms_numeric.png` | Histograms of the numeric columns, with the percent of zeros. |
| `images/02_pagevalues_buckets.png` | Buying rate by `PageValues` bucket. |
| `images/02_revenue_counts.png` | How many cleaned visits bought and how many did not. |
| `images/03_log_before_after.png` | The three most skewed training columns before and after `log1p`. |

## How to run

The notebooks were run with Python 3.10.8. They also run with Python 3.11. Use Python 3.10 to 3.13. `numpy==2.2.6` has no ready-made install package for Python 3.14.

From the repo root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_data_cleaning.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_preprocessing.ipynb
```

Run the Phase 1 notebooks in that order, from the `notebooks/` folder. Notebook 01 writes the cleaned table. Notebook 02 reads it and writes the charts. Notebook 03 reads it and writes the split, the prepared tables, and the saved preprocessor.

Phase 2 reads that split. From the repo root:

```bash
python -m phase2.train
python -m nbconvert --to notebook --execute --inplace notebooks/04_modeling.ipynb
python -m streamlit run website/app.py
```

`phase2.train` writes `phase2/results/` and the two pipelines in `phase2/models/`. The notebook loads those files and scores the test rows again. The website loads the same pipelines. Open the local URL Streamlit prints, enter a visit, and read the chance of purchase.

The website is deployed on Streamlit Community Cloud: https://buyerbehaviorprediction-48lrfsoksjxq27wuoh28gc.streamlit.app/

To deploy it again, point a new app at `website/app.py` on the `main` branch. Under "Advanced settings", choose Python 3.11 before you deploy. The default Python is newer, and `numpy==2.2.6` has no ready-made package for it, so the build compiles numpy and stalls. The Python version cannot be changed after deploying, so a wrong choice means deleting the app and deploying again. Do not put any key in the repo. This project does not need one.

A free app sleeps when nobody visits for a while. Open the link a few minutes before a demo, and click the button to wake it if one appears.

## Models

Each model is a scikit-learn pipeline. The pipeline logs the skewed counts, scales those columns, and one-hot encodes the categories with `min_frequency=20`. The preparation is fit on each training fold, not on the test rows. Class weights are set on the training rows so the 15.6% of buyers are not ignored.

| Model | Role |
| --- | --- |
| Majority class | Always predicts "no purchase". |
| Logistic regression | Linear baseline. `class_weight="balanced"`. |
| Decision tree | One tree, with a minimum leaf size so it cannot memorize single rows. |
| Random forest | 200 trees. This is the better model when the three visit rates are included. |
| Gradient boosting | Histogram gradient boosting. This is the model the website uses for a normal form. |

A small grid search picks the depth, the leaf size, the learning rate, or `C`. The score being maximized is F1. The probability cutoff is chosen from out-of-fold training scores.

`PageValues`, `BounceRates`, and `ExitRates` are calculated from the visit. Every model is trained once without them and once with them. The form uses the version without them. The version with them is an upper bound.

## Results

Numbers are on the 2,441 held-out test rows (382 buyers), after the model was chosen on the training folds.

| Setup | Model | Precision | Recall | F1 | ROC AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Majority class | Always "no" | 0 | 0 | 0 | 0.50 |
| Form columns only | Gradient boosting | 0.311 | 0.667 | 0.424 | 0.770 |
| Form columns plus the three visit rates | Random forest | 0.665 | 0.691 | 0.678 | 0.930 |

The shipped model flags 255 of the 382 buyers and also flags 566 visits that did not buy. Its cutoff is 0.53. At cutoff 0.5, F1 is 0.413, so the training-fold cutoff is a small improvement. Accuracy is 0.716, below the 0.844 of "nobody buys". The F1 is the number that matters.

The random forest that also sees the visit rates has 5-fold F1 0.681 and test F1 0.678. Its score on the training rows themselves is 0.916 because the forest is deep. The fold score and the test score agree.

On the shipped model, May recall is 0.532 (79 test buyers) and November recall is 0.851 (161 test buyers). New visitors are recalled at 0.825. Returning visitors are recalled at 0.614. Recall is similar when `PageValues` is 0 (0.694) and when it is above 0 (0.661). Precision is not: a flag on a zero page-value visit is right 10.3% of the time, and a flag on a positive page-value visit is right 61.0% of the time. Most false alarms are in the zero group, and the shipped model is not allowed to see that column.

Shuffling `ProductRelated_Duration` hurts test F1 the most, then `Month`, `ProductRelated`, `VisitorType`, and `TrafficType`. `Weekend`, `Region`, and `Browser` add nothing once the other columns are present.

### Phase 2 graphs

Every PNG in `phase2/results/`, written by `python -m phase2.train`.

| File | What it shows |
| --- | --- |
| `phase2/results/04_cv_comparison.png` | Precision, recall, and F1 of each model on the 5 training folds, using the form columns at cutoff 0.5. Gradient boosting and random forest tie on F1 (about 0.40). The majority class scores 0 on all three. |
| `phase2/results/04_feature_set_gap.png` | 5-fold F1 of each model without and with `PageValues`, `BounceRates`, and `ExitRates`. The three columns lift every model from about 0.37–0.40 to about 0.62–0.67. |
| `phase2/results/04_confusion_matrices.png` | Test-set counts for both models. The shipped model finds 255 buyers, misses 127, and raises 566 false alarms. The model with the visit rates finds 264, misses 118, and raises 133 false alarms. |
| `phase2/results/04_threshold_sweep.png` | Precision, recall, and F1 of the shipped model on out-of-fold training scores, for cutoffs from 0.15 to 0.85. F1 stays within 0.01 of its best (0.401) from 0.45 to 0.61. The chosen cutoff 0.53 sits on that flat part. |
| `phase2/results/04_roc_curves.png` | Test ROC curves. AUC is 0.770 for the shipped model and 0.930 with the visit rates. The dashed line is a model with no ranking skill. |
| `phase2/results/04_pr_curves.png` | Test precision-recall curves. At recall 0.6, precision is about 0.34 for the shipped model and about 0.72 with the visit rates. |
| `phase2/results/04_feature_importance.png` | Drop in test F1 when one input is shuffled, for the shipped model. `ProductRelated_Duration` matters most (0.099), then `Month` (0.048) and `ProductRelated` (0.043). `Weekend`, `Region`, and `Browser` are below 0, so they add nothing. |

## How the website works

The website (`website/app.py`) scores one visit at a time. A visit is one row of the dataset: 17 facts about a single session on the shop.

1. The user fills in the form. The fields are the dataset columns:
   - When and who: `Month`, `VisitorType`, `Weekend`, and `SpecialDay` (how close the visit was to a holiday, from 0 to 1).
   - Pages seen and seconds spent: administrative pages (account and login), informational pages (shipping, contact), and product pages.
   - Codes: `OperatingSystems`, `Browser`, `Region`, and `TrafficType`. The dataset gives numbers only, not names.
2. The optional analytics box holds `BounceRates`, `ExitRates`, and `PageValues`. The shop's analytics software calculates these, and a shopper cannot know them.
3. If the box is off, the gradient boosting model scores the visit from the 14 form columns. If the box is on, the random forest scores it from all 17 columns.
4. The page shows the chance of purchase and compares it with that model's cutoff (0.53 without analytics, 0.51 with analytics). At or above the cutoff, the visit is flagged as a likely purchase.
5. The two example buttons load real training visits, one buyer and one non-buyer, that both models score correctly. `phase2/train.py` stops with an error if a retrained model ever scores them on the wrong side of the cutoff.

The chance is how closely the visit matches past buyers. It is not a promise. On the test rows, the form-only model is right about 1 time in 3 when it flags a visit.

### Sample visits to try

Click "Example: typical non-buyer" first, then change only the fields listed. The scores below come from the saved models in `phase2/models/`.

| Visit | Month | Visitor type | Product pages / seconds | Admin pages / seconds | Bounce / exit / page value | Form only | With analytics |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Quick bouncer | Mar | unchanged | 1 / 5 | unchanged | 0.2 / 0.2 / 0 | 16.1%, no | 12.2%, no |
| Short browse | May | unchanged | 5 / 120 | unchanged | 0.02 / 0.05 / 0 | 20.0%, no | 9.0%, no |
| Window shopper | Mar | unchanged | 20 / 600 | unchanged | 0.01 / 0.03 / 0 | 52.5%, no | 10.4%, no |
| Engaged returning shopper | Dec | unchanged | 40 / 1500 | 3 / 80 | 0.005 / 0.02 / 5 | 59.9%, yes | 48.5%, no |
| New visitor in November | Nov | New_Visitor | 30 / 1000 | 2 / 60 | 0.005 / 0.02 / 15 | 82.4%, yes | 86.8%, yes |
| Holiday power shopper | Nov | New_Visitor | 80 / 3000 | 5 / 200 | 0 / 0.01 / 60 | 84.6%, yes | 83.4%, yes |

The holiday power shopper also has 2 informational pages and 60 informational seconds. The window shopper and the engaged returning shopper are close to the form-only cutoff. Once `PageValues` is known (0 and 5), the analytics model says no to both.

## Conclusion

The model a person can fill in finds about two thirds of buyers and is right about one time in three when it raises a flag. Use it to rank visits, not to decide them automatically. A shop that already stores `PageValues`, `BounceRates`, and `ExitRates` should use the random forest instead. F1 rises from 0.424 to 0.678.

A missed buyer still has no stated cost, February and June are too small to trust, and the three visit rates should be logged by the shop rather than typed into a form.
