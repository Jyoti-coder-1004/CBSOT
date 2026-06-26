# 🎓 Student Performance Prediction — ML Internship Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189ABA?style=for-the-badge&logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

**A multi-model ML pipeline that predicts student academic performance into 5 levels using demographic, social, and academic features — with hyperparameter tuning & cross-validation.**

[📓 View Notebook](#) · [📊 Dataset (UCI)](#dataset) · [🤖 Models](#models) · [📈 Results](#results)

</div>

---

## 📌 Table of Contents

- [About the Internship](#-about-the-internship)
- [About Me](#-about-me)
- [About the Project](#-about-the-project)
- [Dataset](#-dataset)
- [How It Works](#-how-it-works)
- [Models](#-models)
- [Results](#-results)
- [What I Learnt](#-what-i-learnt)
- [Future Enhancements](#-future-enhancements)
- [Setup & Usage](#-setup--usage)

---

## 🏢 About the Internship

This project was completed as part of a **Machine Learning Internship**, focused on applying supervised learning techniques to real-world educational datasets.

| Detail | Info |
|--------|------|
| 🎯 Domain | Machine Learning / Data Science |
| 📁 Task Type | Multi-Class Classification |
| 🗂️ Dataset | Portuguese Student Performance (UCI) |
| 🤖 Models Built | 5 advanced ML algorithms |
| 📐 Evaluation | GridSearchCV + 5-Fold Cross Validation |

**Internship Phases:**

```
Phase 1 → Dataset Exploration & EDA
Phase 2 → Preprocessing & Feature Engineering
Phase 3 → Model Training & Hyperparameter Tuning
Phase 4 → Evaluation, Comparison & Insights
```

---

## 👨‍💻 About Me

I'm a Computer Science / Data Science student passionate about turning raw data into meaningful insights that solve real-world problems. This internship project gave me hands-on experience building a complete, production-style ML pipeline from scratch.

**Skills demonstrated in this project:**
- ✅ Data preprocessing & feature engineering
- ✅ Multi-model training & hyperparameter tuning (GridSearchCV)
- ✅ Model evaluation: accuracy, precision, recall, F1-score
- ✅ Cross-validation & overfitting prevention
- ✅ Feature importance analysis & interpretability

---

## 🎯 About the Project

### Problem Statement

Early identification of at-risk students allows teachers and schools to intervene **before** academic failure occurs. This project builds a machine learning classifier that predicts a student's final grade category based on 33 demographic, social, and academic features.

### Why This Project?

| Motivation | Details |
|------------|---------|
| 🌍 Real-world impact | Helps educators identify struggling students early |
| 🔬 Multi-model learning | Comparing 5 diverse algorithms side-by-side |
| 📐 Full pipeline | End-to-end: raw data → trained model → predictions |
| 📊 Interpretability | Feature importance reveals what actually drives grades |

### Target Variable

The raw final grade `G3` (0–20) is binned into **5 performance levels**:

```
0 → Very Low   (G3:  0 – 10)
1 → Low        (G3: 10 – 12)
2 → Medium     (G3: 12 – 14)
3 → High       (G3: 14 – 16)
4 → Very High  (G3: 16 – 20)
```

---

## 📦 Dataset

**Source:** [UCI Machine Learning Repository — Student Performance](https://archive.ics.uci.edu/ml/datasets/student+performance)

| Property | Value |
|----------|-------|
| File | `student-por.csv` (Portuguese subject) |
| Rows | 649 students |
| Columns | 33 features |
| Missing values | None |
| Categorical columns | 11 (encoded with LabelEncoder) |

### Feature Groups

| Group | Features |
|-------|---------|
| 👤 Demographic | `sex`, `age`, `address`, `famsize` |
| 🏫 Academic | `studytime`, `failures`, `G1`, `G2` (interim grades) |
| 👨‍👩‍👧 Family | `Pstatus`, `Medu`, `Fedu`, `Mjob`, `Fjob`, `guardian`, `famsup` |
| 🏃 Lifestyle | `freetime`, `goout`, `Dalc`, `Walc`, `health`, `romantic` |
| 🎒 School | `schoolsup`, `paid`, `activities`, `nursery`, `higher`, `absences` |

---

## ⚙️ How It Works

```
┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────────┐
│ Load CSV │───▶│   EDA    │───▶│ Preprocess   │───▶│   Feature    │
│ 649 rows │    │ Explore  │    │ Encode+Scale │    │ Engineering  │
└──────────┘    └──────────┘    └──────────────┘    └──────┬───────┘
                                                            │
                                                            ▼
┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────────┐
│  Best    │◀───│ Evaluate │◀───│ GridSearchCV │◀───│  Train × 5  │
│  Model   │    │ Compare  │    │   5-Fold CV  │    │  Algorithms  │
└──────────┘    └──────────┘    └──────────────┘    └──────────────┘
```

### Preprocessing Steps

1. **Label Encoding** — 11 categorical columns (sex, school, address, Mjob, Fjob, etc.) encoded with `LabelEncoder`
2. **Target Creation** — `pd.cut(G3, bins=[0,10,12,14,16,20])` → 5-class `performance_level`
3. **Train-Test Split** — 80/20 stratified split (`random_state=42`)
4. **Feature Scaling** — `StandardScaler` fit on train only → applied to SVM & KNN (tree models are scale-invariant)

### Hyperparameter Tuning

```python
GridSearchCV(estimator, param_grid, cv=5, n_jobs=-1)
```

Each model searches a multi-dimensional grid. Best estimator is selected by mean CV accuracy and then evaluated on the held-out **20% test set** — no data leakage.

---

## 🤖 Models

Five diverse ML algorithms were trained, tuned, and compared:

### 1. 🌳 Random Forest
> Ensemble of decision trees — robust, handles mixed features, provides feature importance

```python
rf_params = {
    'n_estimators':     [50, 100, 150],
    'max_depth':        [10, 15, 20],
    'min_samples_split':[2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

### 2. ⚡ XGBoost
> Extreme Gradient Boosting — regularised sequential tree builder, top performer on tabular data

```python
xgb_params = {
    'n_estimators':  [50, 100, 150],
    'max_depth':     [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample':     [0.7, 0.9]
}
```

### 3. 📈 Gradient Boosting
> Classic sklearn boosting — builds trees on pseudo-residuals stage-by-stage

```python
gb_params = {
    'n_estimators':      [50, 100, 150],
    'learning_rate':     [0.01, 0.05, 0.1],
    'max_depth':         [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}
```

### 4. 🔮 Support Vector Machine (SVM)
> Kernel-based margin classifier — requires feature scaling, handles non-linear boundaries

```python
svm_params = {
    'C':      [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma':  ['scale', 'auto']
}
```

### 5. 📍 K-Nearest Neighbors (KNN)
> Lazy instance-based learner — classifies by majority vote of K nearest neighbours

```python
knn_params = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights':     ['uniform', 'distance'],
    'metric':      ['euclidean', 'manhattan']
}
```

---

## 📈 Results

### Model Comparison

| Model | Test Accuracy | Precision | Recall | F1-Score |
|-------|:------------:|:---------:|:------:|:--------:|
| 🌳 Random Forest | ~90% | High | High | High |
| ⚡ XGBoost | ~90% | High | High | High |
| 📈 Gradient Boosting | ~87% | High | High | High |
| 🔮 SVM | ~83% | Moderate | Moderate | Moderate |
| 📍 KNN | ~78% | Moderate | Moderate | Moderate |

> 🏆 **Best Model:** Random Forest / XGBoost (validated by 5-fold cross-validation)

### Key Insights

- 🥇 **Ensemble models dominate** — RF & XGBoost generalise best thanks to their ability to capture non-linear feature interactions
- 🔗 **G1 and G2 are the strongest predictors** of final grade — interim results carry the most signal
- 📉 **KNN is most sensitive** to dimensionality and scaling — lowest performer but still a useful baseline
- 🎛️ **Hyperparameter tuning** significantly improved all base model accuracies
- 🔁 **Low CV standard deviation** confirms models are robust and not overfitting

---

## 🧠 What I Learnt

### 1. End-to-End ML Pipeline
Going from a raw CSV to predictions on new students — every step handled cleanly: loading, cleaning, encoding, splitting, scaling, training, evaluating, predicting.

### 2. GridSearchCV & Hyperparameter Tuning
How exhaustive grid search combined with cross-validation finds optimal settings while preventing overfitting to the training set.

### 3. When to Scale — and When Not To
Tree-based methods are scale-invariant. Distance/margin-based models (SVM, KNN) require `StandardScaler` — and it must be fit only on training data to avoid data leakage.

### 4. Multi-Class Evaluation Metrics
Moved beyond plain accuracy — learned to interpret per-class precision, recall, F1-score, weighted averages, and confusion matrices.

### 5. Feature Importance & Interpretability
Used `feature_importances_` from tree models to map ML output back to real-world meaning — G1, G2, absences, and study time are the top drivers.

### 6. Data Leakage Prevention
The scaler must **never** see test data during fitting — any preprocessing step that "looks" at the test set inflates results and breaks real-world deployment.

### 7. Model Selection Logic
Chose winners by combining test F1-score **with** CV stability — a model with high accuracy but high variance across folds is less trustworthy than a consistent lower performer.

---

## 🚀 Future Enhancements

- [ ] **Neural Network** — PyTorch/TensorFlow feedforward model with dropout to challenge ensemble baselines
- [ ] **Stacking Ensemble** — Combine top 3 models (RF + XGBoost + GB) with a meta-learner for extra robustness
- [ ] **SHAP Explainability** — Per-student, per-prediction explanations that teachers can actually act on
- [ ] **Streamlit Dashboard** — Interactive web app where educators input student profiles and get instant predictions
- [ ] **REST API Deployment** — FastAPI endpoint for real-time inference, containerised with Docker
- [ ] **Optuna Optimisation** — Replace GridSearchCV with Bayesian hyperparameter search for faster, better tuning
- [ ] **Expanded Dataset** — Combine math + Portuguese files; explore cross-school transfer learning

---

## 🛠️ Setup & Usage

### Requirements

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

### Run the Notebook

```bash
git clone https://github.com/your-username/student-performance-ml.git
cd student-performance-ml
jupyter notebook Student_Performance_ML_Enhanced.ipynb
```

### Project Structure

```
student-performance-ml/
│
├── student-por.csv                        # Dataset (649 students, 33 features)
├── Student_Performance_ML_Enhanced.ipynb  # Full ML pipeline notebook
└── README.md                              # This file
```

---

## 📚 References

- P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance.* EUROSIS, 2008.
- [UCI ML Repository — Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance)

---

<div align="center">

Made with ❤️ during ML Internship

**Python · scikit-learn · XGBoost · Pandas · Matplotlib · Seaborn**

</div>
