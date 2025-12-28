# Heart Attack Classification Notebook Guide

This document explains **how the notebook** [heart_attack_classification.ipynb](heart_attack_classification.ipynb) works end-to-end.

## What the notebook produces

- Loads the dataset from `data/heart_attack_prediction_dataset.csv`
- Cleans/engineers features (notably parses `Blood Pressure` into numeric columns)
- Runs **early feature ranking** using:
  - **Mutual Information (MI)**
  - **ANOVA F-test** (`f_classif`)
- Runs EDA plots (target distribution, feature distributions, per-feature risk rates)
- Trains and compares **5 models** with a single reproducible preprocessing pipeline
- Runs a **tuning section** (cross-validated) and selects a `final_pipe`
- Evaluates with confusion matrix + ROC/PR curves
- Computes permutation feature importance on the final pipeline
- Saves the chosen pipeline to `heart_attack_risk_best_pipeline.joblib`

## Notebook structure (top to bottom)

### 1) Title + checklist
The first two cells are markdown:
- Project title
- An A→Z checklist that mirrors the notebook steps

Nothing executes here; it’s just orientation.

### 2) Imports and global plotting settings
The imports cell:
- Turns off warnings to keep output readable
- Imports numerical/data/plotting libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`
- Imports scikit-learn tools for:
  - splitting data (`train_test_split`)
  - preprocessing (`ColumnTransformer`, `Pipeline`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`)
  - modeling (Logistic Regression, RandomForest, GradientBoosting, HistGradientBoosting, KNN)
  - evaluation (accuracy/precision/recall/F1/ROC-AUC, confusion matrix, ROC/PR curves)
  - explainability (permutation importance)
- Imports tuning helpers (`StratifiedKFold`, `RandomizedSearchCV`, `loguniform`)

It also sets a Seaborn theme and default figure size.

### 3) Load the dataset
The data-loading cell:
- Sets `DATA_PATH = 'data/heart_attack_prediction_dataset.csv'`
- Reads it with `pd.read_csv`
- Prints shape and displays the head

This establishes the base dataframe `df`.

### 4) Quick data quality checks
The next cell:
- Displays missing values (top 15)
- Prints duplicate count
- Shows `describe(include='all')` to see distributions/dtypes for many columns

This helps decide what needs preprocessing.

### 5) Feature plan (markdown)
Explains the modeling intent:
- Target is `Heart Attack Risk` (binary)
- Drops `Patient ID` if present
- Parses `Blood Pressure` string like `158/88` into numeric systolic/diastolic columns

### 6) Feature engineering + X/y split
This is the first key transformation cell.

It:
1. Copies the dataframe to `df_model`
2. Drops `Patient ID` if present
3. Parses `Blood Pressure`:
   - splits on `/`
   - converts to numeric (invalid parses become `NaN`)
   - creates:
     - `BP_Systolic`
     - `BP_Diastolic`
   - drops the original `Blood Pressure` text column
4. Sets:
   - `TARGET = 'Heart Attack Risk'`
   - `X = df_model.drop(TARGET)`
   - `y = df_model[TARGET].astype(int)`
5. Detects feature types:
   - `numeric_features = X.select_dtypes(include=[np.number])`
   - `categorical_features = X.select_dtypes(exclude=[np.number])`

These lists are used everywhere later for consistent preprocessing.

### 7) Early Feature Ranking (MI + ANOVA F)
This section is intentionally placed *before modeling*.

#### What it does
It runs two **filter methods** that rank features without training a full predictive model:

1) **Mutual Information (MI)**
- Measures dependency between each feature and the target
- Can capture non-linear relationships

2) **ANOVA F-test** (`f_classif`)
- Measures how well a feature separates the two classes by comparing means
- More “linear separation” focused than MI

#### How it works
Because you have mixed types (numeric + categorical), the notebook first builds a lightweight transformer:
- Numeric: median imputation
- Categorical: most-frequent imputation + one-hot encoding

That produces a single matrix `X_rank` suitable for ranking.

It then:
- Computes MI scores: `mutual_info_classif(X_rank, y)`
- Computes ANOVA scores: `f_classif(X_rank, y)`
- Displays the top-ranked features and plots bar charts

#### How it updates the heatmap
After ranking, it derives `top_numeric_for_heatmap`:
- It takes the top-ranked items that came from numeric columns (feature names that start with `num__...`)
- It merges MI and ANOVA numeric picks

This list is later used to focus the correlation heatmap on the “most relevant numeric features” rather than every numeric column.

### 8) EDA (Exploratory Data Analysis)
The EDA section generates plots for:
- Target class distribution
- Numeric feature histograms
- Categorical count plots
- A correlation heatmap
- Boxplots of numeric features vs target
- Categorical “risk rate” plots (mean target per category)

#### Correlation heatmap
The heatmap cell:
- Uses `StandardScaler()` to create a standardized numeric matrix for plotting
- Uses `top_numeric_for_heatmap` if it exists (so it reflects MI/ANOVA ranking)
- Computes Pearson correlation and plots with Seaborn

Important concept: standardizing does **not** change correlation values; it just confirms consistent scaling and avoids surprises.

### 9) Modeling setup (shared preprocessing + helper functions)
The modeling section starts by splitting:
- `train_test_split(..., stratify=y, random_state=42)`

Then it defines a **single preprocessing pipeline** used for all models:
- Numeric pipeline:
  - median imputation
  - standard scaling
- Categorical pipeline:
  - most-frequent imputation
  - one-hot encoding
- Combined with `ColumnTransformer`

This is critical because:
- Every model is trained on the *same* preprocessing
- There is no “manual preprocessing” mismatch between models

It also defines helper functions:
- `evaluate(pipe, name)` returns accuracy/precision/recall/F1/ROC-AUC
- `show_curves(name, proba)` plots ROC and PR curves

### 10) Compare 5 models
The notebook trains exactly five classifiers, each wrapped as:

`Pipeline([('preprocess', preprocess), ('model', clf)])`

It stores:
- metrics for each
- fitted pipelines
- predictions/probabilities for later reuse

It then creates `results_df` sorted by ROC-AUC and F1.

### 11) Improve Accuracy (tuning + feature reduction)
This section is designed to be “proper” in the sense that:
- it uses cross-validation (`StratifiedKFold`)
- it tunes hyperparameters inside CV (`RandomizedSearchCV`)

It also prints a key baseline:
- **majority-class accuracy** on the test split

Two strategies are tried:
- A tuned Logistic Regression (sparse / regularized)
- A tuned HistGradientBoosting

It also includes a variant that removes the high-cardinality `Country` feature.

The section picks a `final_pipe` (and `final_name`, `final_preds`, `final_proba`) based on whether tuned performance meets/exceeds the earlier leaderboard.

### 12) Final evaluation
Using the chosen final model (`final_pipe`):
- Plots confusion matrix
- Plots ROC curve and Precision–Recall curve (if probabilities exist)

### 13) Feature importance (permutation)
Permutation importance is computed for the **final** model.

Two views are produced:
1) Raw-feature view:
- Permutes original input columns (`X_test` columns)
- Easiest to interpret at the “business feature” level

2) Expanded one-hot view:
- Transforms `X_test` through the preprocessor
- Permutes the transformed columns (one-hot expanded)
- More granular but can have many features

### 14) Save the model pipeline
Finally:
- Saves `final_pipe` to `heart_attack_risk_best_pipeline.joblib`

This artifact contains:
- the full preprocessing logic
- the model

So you can later load it and call `.predict()` / `.predict_proba()` on raw rows.

## How to run the notebook cleanly

1. Start at the top and run cells in order.
2. If you change feature engineering (e.g., adding/removing columns), re-run from the feature engineering cell onward.
3. If you change the ranking methods, re-run ranking and then the heatmap cell to refresh the “top numeric” selection.

## Notes on metrics (important)

Accuracy can be misleading when classes are imbalanced.
- The notebook prints the majority-class baseline so you can tell whether accuracy is meaningful.
- If you care about catching risk cases, prioritize `recall`, `f1`, and PR curves.

If you want, I can add a short section explaining how to pick a probability threshold to trade off false positives vs false negatives.
