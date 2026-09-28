# 🩺 Disease Prediction using Machine Learning

A machine learning project that predicts whether a tumour is **malignant or benign** from cell measurements, using a **Random Forest** classifier. Includes model comparison, hyperparameter tuning, evaluation charts, and a **Streamlit web app**.

## Highlights
- Dataset: Breast Cancer Wisconsin (Diagnostic), 569 patient records, 30 features (built into scikit-learn)
- Compared **Logistic Regression, Decision Tree and Random Forest** using 5-fold cross-validation
- Tuned Random Forest with **GridSearchCV**, optimising **recall** (missing a disease is costlier than a false alarm)
- Evaluated with accuracy, precision, recall, F1 and a confusion matrix
- Interactive web app for live predictions

## Results
| Metric (test set) | Score |
|---|---|
| Accuracy | 97% |
| Precision (malignant) | 100% |
| Recall (malignant) | 93% |
| F1-score (malignant) | 96% |

5-fold cross-validation accuracy: Logistic Regression 97.4%, Random Forest 96.5%, Decision Tree 92.3%.
Logistic Regression scored slightly higher in cross-validation. Random Forest was chosen for its tunability and feature importance insights. Exact numbers are saved in `reports/metrics.json`.

| Confusion Matrix | Feature Importance |
|---|---|
| ![cm](reports/confusion_matrix.png) | ![fi](reports/feature_importance.png) |

## Tech Stack
Python, pandas, NumPy, scikit-learn, Matplotlib, Streamlit, Joblib

## Project Structure
```
disease-prediction-ml/
├── app.py              # Streamlit web app
├── train_model.py      # Training + evaluation pipeline
├── requirements.txt
├── data/               # Dataset CSV (auto-generated)
├── models/             # Saved model
└── reports/            # Metrics and charts
```

## How to Run
```bash
git clone https://github.com/<your-username>/disease-prediction-ml.git
cd disease-prediction-ml
pip install -r requirements.txt
python train_model.py        # trains the model and creates reports
streamlit run app.py         # launches the web app
```

## What I Learned
- Why accuracy alone can mislead in medical data, and why recall matters
- Using stratified splits and cross-validation to avoid overfitting
- Reading feature importance to understand key risk indicators
- Turning a trained model into a usable web app

## Future Improvements
- Add a Power BI dashboard of the dataset insights
- Try other diseases (diabetes, heart disease) with UCI data
- Deploy the app on Streamlit Community Cloud

## Disclaimer
Educational project only. Not a medical diagnostic tool.
