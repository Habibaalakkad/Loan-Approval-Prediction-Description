# Loan Approval Prediction

This project builds and evaluates machine learning models to predict whether a loan application will be approved or rejected. It is based on the **Loan Approval Prediction dataset** from Kaggle. The workflow covers preprocessing, handling missing data, dealing with class imbalance, and comparing multiple classification models.

---

## 📂 Dataset

* Source: Loan Approval Prediction Dataset (Kaggle)
* Size: ~4,200 rows, 13 columns
* Target column: `loan_status` (`Approved` / `Rejected`)
* Features include applicant details (income, education, dependents, CIBIL score) and asset information.

---

## ⚙️ Project Workflow

### 1. Data Preprocessing

* Handled **missing values**:

  * Numeric → replaced with median
  * Categorical → replaced with mode
* Encoded categorical features using **LabelEncoder**
* Scaled numerical features with **StandardScaler**

### 2. Handling Class Imbalance

* Used **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset.
* Ensures models are not biased toward the majority class.

### 3. Model Training

Trained and evaluated multiple classifiers for comparison:

* **Logistic Regression** (baseline linear model)
* **Decision Tree Classifier** (non-linear model)
* **Random Forest Classifier** (ensemble model, tuned with 200 trees)

### 4. Model Evaluation

* Evaluation metrics: **Precision, Recall, F1-score, Accuracy**
* Visualizations:

  * Confusion Matrix (Heatmap)
  * ROC Curves & AUC scores
  * Feature Importance (Random Forest)
  * Classification Reports

---

## 📊 Results

Example (Random Forest results):

* Accuracy: ~99%
* Precision, Recall, and F1-score all above **0.98**

Comparison across models:

* Logistic Regression → Strong baseline
* Decision Tree → Better recall, interpretable model
* Random Forest → Best overall performance

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **Pandas, NumPy** for data preprocessing
* **Scikit-learn** for ML models & evaluation
* **Imbalanced-learn (SMOTE)** for oversampling
* **Matplotlib, Seaborn** for visualization

---

## 🚀 How to Run

1. Clone this repo:

   ```bash
   git clone https://github.com/yourusername/loan-approval-prediction.git
   cd loan-approval-prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:

   ```bash
   python loan_prediction.py
   ```

---

## 🔮 Future Improvements

* Hyperparameter tuning with GridSearchCV
* Add more advanced models (XGBoost, LightGBM)
* Deploy as a **Flask/Django web app** for real-time predictions

---

## 📌 Author

Developed by **[Your Name]** as part of the **Elevvo Pathway** Machine Learning track.

