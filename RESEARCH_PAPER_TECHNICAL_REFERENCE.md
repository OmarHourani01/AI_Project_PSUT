# Technical Reference for Research Paper

**Project:** Heart Attack Risk Classification (Binary Classification)

This document is intended to be your **technical reference** while writing the research paper. It mirrors the implementation decisions and workflow in the notebook **`AI_Project_The_Omars.ipynb`**, and it explains the modeling choices at both:

- **High level:** project framing + model pros/cons
- **Low level:** EDA steps, preprocessing rules, model hyperparameters, training protocol, tuning grids, evaluation metrics

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Formulation](#problem-formulation)
3. [Dataset](#dataset)
4. [Reproducibility & Environment](#reproducibility--environment)
5. [Data Preparation & Feature Engineering](#data-preparation--feature-engineering)
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
7. [Train/Test Split Strategy](#traintest-split-strategy)
8. [Preprocessing Pipeline (Leakage-Safe)](#preprocessing-pipeline-leakage-safe)
9. [Models Used (Base Learners)](#models-used-base-learners)
10. [Cross-Validation Protocol](#cross-validation-protocol)
11. [Hyperparameter Tuning (GridSearchCV)](#hyperparameter-tuning-gridsearchcv)
12. [Ensemble Learning](#ensemble-learning)
13. [Evaluation Metrics & Thresholding](#evaluation-metrics--thresholding)
14. [Interpretation & Discussion Guidance](#interpretation--discussion-guidance)
15. [Limitations & Risk of Misinterpretation](#limitations--risk-of-misinterpretation)
16. [Suggested Paper Structure (Technical Sections)](#suggested-paper-structure-technical-sections)

---

## Project Overview

**Goal:** Predict whether an individual is at **heart attack risk** (binary label) using demographic, lifestyle, and clinical measurements.

**Approach summary:**

- Load `data/heart_attack_prediction_dataset.csv`
- Standardize column names and engineer features (notably parse blood pressure)
- Perform EDA (class balance, univariate distributions, category frequencies, correlations)
- Split into train/test with stratification
- Build a **single preprocessing pipeline** (numeric + categorical) and reuse it across models
- Compare multiple model families using stratified CV (ROC-AUC)
- Tune each model with `GridSearchCV`
- Evaluate tuned models on held-out test set
- Train and evaluate ensemble methods (soft voting + stacking)

**Core idea for paper framing:**

- This is a **benchmarking study** across diverse model families.
- The pipeline is designed to be **reproducible** and **leakage-resistant** by embedding all transforms in scikit-learn `Pipeline` / `ColumnTransformer`.

---

## Problem Formulation

- **Task type:** supervised binary classification
- **Input:** a feature vector $\mathbf{x} \in \mathbb{R}^d$ with mixed numeric/categorical fields
- **Output:** predicted class $\hat{y} \in \{0,1\}$ and/or a risk score $\hat{p} = P(y=1\mid \mathbf{x})$

**Target column handling:**

- If `target` does not exist and the dataset contains `Heart Attack Risk`, the notebook renames `Heart Attack Risk` to `target`.
- Target interpretation throughout the analysis:
  - `0` = no heart attack risk
  - `1` = heart attack risk

---

## Dataset

**File:** `data/heart_attack_prediction_dataset.csv`

**Expected raw fields (conceptual):**

- Demographics: Age, Sex, Income, Country/Continent/Hemisphere
- Clinical/biometric: Cholesterol, Blood Pressure, Heart Rate, BMI, Triglycerides
- Lifestyle: Smoking, Alcohol Consumption, Exercise Hours/Week, Physical Activity Days/Week, Sleep Hours/Day, Sedentary Hours/Day
- Medical history: Diabetes, Family History, Previous Heart Problems, Medication Use
- Other: Diet, Stress Level

**Identifier field:**

- `Patient ID` / `patient_id` is dropped (high-cardinality identifier; not meaningful for learning generalizable patterns).

---

## Reproducibility & Environment

**Random seed used:**

- `RANDOM_STATE = 42`

This seed is applied to:

- `train_test_split(..., random_state=42, stratify=y)`
- `StratifiedKFold(..., shuffle=True, random_state=42)`
- random-state-capable models (Logistic Regression, SVC, Random Forest, XGBoost, MLP, stacking final estimator)

**Dependencies (as used by the notebook):**

- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`
- `xgboost`

Note: your `requirements.txt` pins many additional packages, but the notebook’s core workflow relies primarily on the packages above.

---

## Data Preparation & Feature Engineering

This section describes the *exact* cleaning/engineering logic used.

### 1) Column standardization / renaming

The notebook applies a renaming map (when columns exist) to convert human-readable column names into snake_case equivalents.

Examples:

- `"Blood Pressure" → "blood_pressure"`
- `"Exercise Hours Per Week" → "exercise_hours_per_week"`
- `"Heart Attack Risk" → "target"` (if `target` not already present)

Why this matters for a paper:

- It makes feature engineering consistent and reduces risk of mistakes due to spaces/special characters.

### 2) Blood pressure parsing

If `blood_pressure` exists and is formatted like `"120/80"`, the notebook creates:

- `systolic_bp` = numeric left side
- `diastolic_bp` = numeric right side

Implementation details:

- Converts to string, splits once on `/`.
- Uses `pd.to_numeric(..., errors="coerce")` to handle malformed values (become NaN).
- Drops the original `blood_pressure` column.

Justification:

- Many models require numeric inputs; string blood pressure must be converted.
- Splitting retains clinically meaningful interpretation (systolic vs diastolic).

### 3) Dropping ID column

- Drops `patient_id` (or `Patient ID` if still present).

Justification:

- Identifiers are high-cardinality and can cause overfitting / leakage-like behavior without improving generalization.

### 4) Encoding sex

If `sex` exists and is object/string:

- Applies `LabelEncoder()` on `sex`.

Notes:

- This is safe for binary categories like `Male/Female`.
- In general, label encoding multi-class categories can impose an artificial ordering, but here the notebook only applies it to `sex`.

---

## Exploratory Data Analysis (EDA)

EDA is performed after feature engineering and before training.

### 1) Target distribution

- Uses `seaborn.countplot` to visualize class counts for `target`.
- Prints absolute counts and percentages.

Purpose:

- Detects class imbalance (important for interpreting accuracy and for choosing ROC-AUC).

### 2) Feature type identification

- `numeric_cols` = all numeric columns (excluding `target`)
- `cat_cols` = all non-numeric columns (excluding `target`)

This is used to:

- Decide which columns are scaled vs one-hot encoded
- Decide which plots to generate

### 3) Descriptive statistics

- `df[numeric_cols].describe()` provides mean/std/min/max/quartiles.

Purpose:

- Understand scale, outliers, plausible ranges, and potential preprocessing needs.

### 4) Numeric feature histograms

- For each numeric feature, plots histogram with KDE:
  - `sns.histplot(..., kde=True, bins=30)`

Purpose:

- Identify skewness, multimodality, outliers.

### 5) Correlation heatmap (numeric)

- Computes correlation matrix using `df[numeric_cols + ['target']].corr(numeric_only=True)`
- Plots via `plt.imshow`

Interpretation guidance:

- Correlation is linear association only.
- Correlation with `target` can hint at predictive signal, but does not guarantee causal relevance.

### 6) Boxplots for numeric features

- For each numeric feature:
  - `sns.boxplot(x=df[col])`

Purpose:

- Highlight outliers and distribution spread.

### 7) Skewness

- Computes `df[numeric_cols].skew()`

Purpose:

- Quantifies asymmetry to motivate transformations if needed.

### 8) Categorical feature exploration

- For each categorical column:
  - prints `value_counts()`

Additionally:

- Generates count plots for all categorical features (grid layout).
- Generates a dedicated count plot for `country` (often high-cardinality).

### 9) Risk-rate by category

For each categorical feature `col`:

- Computes `df.groupby(col)['target'].mean()` and plots as bar chart.

Important explanation for the paper:

- Because `target` is binary, the mean equals the **empirical risk rate**: the proportion of positive cases in each category.

---

## Train/Test Split Strategy

The notebook uses:

```python
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)
```

Key points for writing:

- **Hold-out test set:** 20% reserved for final evaluation.
- **Stratification:** preserves class proportions across splits.
- **Random state:** ensures reproducible split.

---

## Preprocessing Pipeline (Leakage-Safe)

All preprocessing is done inside a scikit-learn pipeline.

### Numeric preprocessing

Pipeline:

1. `SimpleImputer(strategy='median')`
2. `StandardScaler()`

Justification:

- Median imputation is robust to outliers.
- Standard scaling benefits models sensitive to feature scale (LogReg, SVM, MLP).

### Categorical preprocessing

Pipeline:

1. `SimpleImputer(strategy='most_frequent')`
2. `OneHotEncoder(handle_unknown='ignore')`

Justification:

- Imputation ensures stability if missing values appear due to future data changes.
- One-hot encoding avoids ordinal assumptions.
- `handle_unknown='ignore'` prevents runtime failure on unseen categories.

### Combined transformer

- `ColumnTransformer([('num', numeric_pipeline, numeric_cols), ('cat', categorical_pipeline, cat_cols)], remainder='drop')`

Leakage note (important for methodology section):

- Because preprocessing is inside the pipeline, **each cross-validation fold fits preprocessing only on the training fold**.

---

## Models Used (Base Learners)

This project evaluates five model families:

1. Logistic Regression
2. Support Vector Machine (SVC)
3. Random Forest
4. XGBoost (Gradient Boosted Trees)
5. Multi-layer Perceptron (Neural Network)

All are wrapped as:

```python
Pipeline([('preprocess', preprocess), ('model', MODEL)])
```

### 1) Logistic Regression

Implementation:

- `LogisticRegression(max_iter=4000, random_state=42, solver='lbfgs')`

Advantages:

- Strong baseline, fast to train.
- Interpretable in terms of linear feature effects (especially after one-hot).

Disadvantages:

- Linear decision boundary (unless you engineer non-linear features).
- Can underfit complex relationships.

### 2) SVM (Support Vector Classifier)

Implementation:

- `SVC(probability=False, class_weight='balanced', random_state=42)`

Advantages:

- Effective in high-dimensional spaces.
- Can model non-linear boundaries with kernels.

Disadvantages / caveats:

- Training can be slow on large datasets.
- If `probability=False`, the classifier does **not** expose `predict_proba()`. Some downstream evaluation/ensembles require probabilities.

### 3) Random Forest

Implementation:

- `RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced', n_jobs=-1)`

Advantages:

- Handles non-linearities and interactions.
- Robust baseline for tabular data.
- Less sensitive to scaling.

Disadvantages:

- Less interpretable than linear models.
- Can be memory-heavy; may not be as strong as boosted trees.

### 4) XGBoost

Implementation:

- `XGBClassifier(`
  - `n_estimators=800`
  - `learning_rate=0.03`
  - `max_depth=4`
  - `subsample=0.85`
  - `colsample_bytree=0.85`
  - `reg_lambda=1.0`
  - `min_child_weight=1.0`
  - `objective='binary:logistic'`
  - `eval_metric='logloss'`
  - `random_state=42`
  - `n_jobs=-1`
  - `tree_method='hist'`
  `)`

Advantages:

- Often state-of-the-art for structured/tabular datasets.
- Captures complex interactions and non-linearities.

Disadvantages:

- More hyperparameters; easier to overfit without tuning.
- Feature importance can be misleading if not handled carefully.

### 5) Neural Network (MLP)

Implementation:

- `MLPClassifier(`
  - `hidden_layer_sizes=(128, 64)`
  - `activation='relu'`
  - `alpha=0.0005`
  - `learning_rate_init=0.001`
  - `max_iter=400`
  - `random_state=42`
  `)`

Advantages:

- Flexible function approximator.
- Can learn non-linear decision surfaces.

Disadvantages:

- Requires scaling; sensitive to hyperparameters.
- Can be unstable (local minima, convergence variability).
- Harder to interpret.

---

## Cross-Validation Protocol

The model comparison uses:

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- `cross_val_score(..., scoring='roc_auc', n_jobs=-1)`

Why ROC-AUC:

- Threshold-independent ranking quality.
- Often more informative than accuracy when classes are imbalanced.

Leakage safety:

- Pipelines ensure preprocessing and model are fitted within each fold.

---

## Hyperparameter Tuning (GridSearchCV)

Tuning is performed separately per model:

- `GridSearchCV(estimator=pipe, param_grid=..., scoring='roc_auc', cv=5-fold stratified, n_jobs=-1, refit=True)`

Key methodology note:

- `refit=True` means the returned `best_estimator_` is retrained on the **full training set** after selecting best hyperparameters.
- The test set remains untouched until final evaluation.

### Parameter grids (exact)

#### Logistic Regression

- `model__C`: `[0.01, 0.1, 1, 10]`
- `model__penalty`: `['l2']`
- `model__solver`: `['lbfgs']`

#### SVM

- `model__C`: `[0.1, 1, 10]`
- `model__kernel`: `['linear', 'rbf']`
- `model__gamma`: `['scale', 'auto']`

#### Random Forest

- `model__n_estimators`: `[300, 500]`
- `model__max_depth`: `[None, 5, 10]`
- `model__min_samples_split`: `[2, 5]`
- `model__min_samples_leaf`: `[1, 2]`

#### XGBoost

- `model__n_estimators`: `[300, 600, 900]`
- `model__max_depth`: `[3, 4, 6]`
- `model__learning_rate`: `[0.01, 0.03, 0.1]`
- `model__subsample`: `[0.8, 1.0]`
- `model__colsample_bytree`: `[0.8, 1.0]`
- `model__reg_lambda`: `[0.5, 1.0, 2.0]`

#### Neural Network (MLP)

- `model__hidden_layer_sizes`: `[(64,), (128, 64), (128, 128)]`
- `model__alpha`: `[0.0001, 0.0005, 0.001]`
- `model__learning_rate_init`: `[0.0005, 0.001, 0.005]`

---

## Ensemble Learning

Two ensemble strategies are evaluated:

### 1) Soft Voting

- `VotingClassifier(estimators=[lr, svm, rf, xgb, nn], voting='soft', n_jobs=-1)`

Interpretation:

- Averages predicted probabilities from base learners.
- Can reduce variance and improve stability.

Caveat:

- Soft voting requires `predict_proba()` from each estimator.

### 2) Stacking

- `StackingClassifier(`
  - `estimators=[lr, svm, rf, xgb, nn]`
  - `final_estimator=LogisticRegression(max_iter=4000, random_state=42)`
  - `stack_method='predict_proba'`
  - `cv=StratifiedKFold(5, shuffle=True, random_state=42)`
  - `n_jobs=-1`
  `)`

Interpretation:

- Learns a meta-model that combines base model outputs.
- Often improves performance when base learners make complementary errors.

Caveat:

- If any base estimator lacks `predict_proba()`, stacking with `stack_method='predict_proba'` may fail.

---

## Evaluation Metrics & Thresholding

The evaluation helper uses:

- `predict_proba(X_test)[:, 1]` to get probabilities
- Threshold at `0.5` to get class predictions:

$$\hat{y} = \mathbb{1}[\hat{p} \ge 0.5]$$

Metrics computed:

- **ROC-AUC**: area under ROC curve
- **Accuracy**: $(TP+TN)/(TP+TN+FP+FN)$
- **Precision**: $TP/(TP+FP)$
- **Recall**: $TP/(TP+FN)$
- **F1**: $2 \cdot (\text{precision} \cdot \text{recall})/(\text{precision}+\text{recall})$

Visuals:

- Confusion matrix
- ROC curve

Paper guidance:

- If the dataset is imbalanced, emphasize **ROC-AUC**, **Recall**, **F1**, and/or precision–recall analysis.
- In medical-risk settings, the cost of false negatives (missing at-risk individuals) may be higher than false positives. This motivates reporting recall and discussing threshold selection.

---

## Interpretation & Discussion Guidance

### When all models perform similarly (low ROC-AUC)

If diverse model families (linear, kernel, bagging trees, boosting, neural nets) all show similarly weak performance, a credible explanation is:

- limited predictive signal in available features, or
- noisy/weakly-defined target, or
- features not causally tied to the outcome.

This is a valid research outcome: “model capacity did not overcome dataset limitations.”

### What to say about preprocessing

- Pipeline-based preprocessing supports fair comparison.
- Scaling affects LogReg/SVM/MLP strongly; tree-based models are less sensitive.

---

## Limitations & Risk of Misinterpretation

Include a limitations section in the paper. Common points consistent with this workflow:

- **Target definition uncertainty:** “risk” labels may be proxies rather than clinically confirmed outcomes.
- **Feature quality:** measurement noise, self-reported lifestyle fields, or coarse categories can limit performance.
- **Potential confounding:** correlations do not imply causation.
- **Geography leakage risk:** `country` may encode socioeconomic or healthcare system patterns that do not generalize.
- **Probability calibration:** raw probabilities from some models may be poorly calibrated unless calibrated explicitly.

---

## Suggested Paper Structure (Technical Sections)

You can map this project cleanly into standard ML paper sections.

1. **Introduction**
   - Motivation: early detection / risk stratification
   - ML framing: supervised classification

2. **Related Work** (optional)
   - Tabular risk prediction; model families

3. **Dataset & Features**
   - Source and size
   - Target definition (`target`)
   - Feature groups (demographic, clinical, lifestyle)

4. **Methodology**
   - Preprocessing pipeline (numeric vs categorical)
   - Train/test split
   - Cross-validation protocol
   - Models compared + rationale

5. **Experiments**
   - Baseline model comparison (CV ROC-AUC)
   - Hyperparameter tuning grids
   - Ensemble methods

6. **Results**
   - Table with metrics on test set (ROC-AUC, accuracy, precision, recall, F1)
   - Confusion matrices + ROC curves for best models

7. **Discussion**
   - Interpret results and limitations
   - Explain why performance may be bounded by data

8. **Conclusion**
   - Summary and future work (feature enrichment, calibration, threshold optimization)

---

## Appendix (Optional): “Exact Implementation Notes”

If you want an appendix that reviewers love, include:

- Random seed values and where applied
- Exact hyperparameter grids
- Exact preprocessing steps
- Exact evaluation threshold and metrics

This document already contains those details.
