# Telecom Customer Churn Prediction & Feature Explainability Engine

Data Science portfolio framework for predictive modeling and operational customer segmentation using the public IBM Telco Customer Churn dataset. The repository contains a self-contained Jupyter notebook that ingests the official dataset, performs preprocessing and feature engineering, trains a Random Forest classifier, exports diagnostics (plots & metrics), and generates an operational customer segmentation manifest.

## Key Features
- Automated dataset extraction from the official IBM Telco CSV.
- Data cleaning & preprocessing (handles blank TotalCharges values).
- One-hot encoding of categorical features.
- Imbalance-aware Random Forest classifier (class_weight='balanced_subsample').
- Model evaluation (accuracy, precision/recall/f1, confusion matrix).
- Diagnostic visual assets: confusion matrix and top feature importance chart (Gini).
- Operational segmentation: exports `customer_segments_manifest.csv` (Loyal / At Risk / Dormant / Standard Active).

## Repository structure
- data/  
  - (created by notebook) dataset saved as `Telco-Customer-Churn.csv`
- documents/  
  - (project documentation, if any)
- outputs/  
  - (created by notebook) `churn_confusion_matrix.png`, `churn_feature_importance.png`, `customer_segments_manifest.csv`
- src/
  - churn_notebook.ipynb — primary notebook implementing the pipeline

## Notebook snapshot
The main pipeline lives in `src/churn_notebook.ipynb`. Highlights from the notebook:
- Downloads dataset: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- Cleans `TotalCharges`, maps `Churn` to binary target, one-hot encodes categorical variables.
- Trains RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced_subsample').
- Example reported test accuracy (from run in notebook): ~76.8% and a balanced classification report.
- Saves visualizations and `customer_segments_manifest.csv` to `outputs/`.

## Requirements
- Python 3.8+ (notebook metadata shows Python 3.11)
- Recommended packages:
  - pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter (or jupyterlab)
- Example install:
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```
(Or create a `requirements.txt` with these packages and run `pip install -r requirements.txt`.)

## How to run
1. Clone the repository and change to its root:
```bash
git clone https://github.com/ishanpote/telecom-churn-intelligence.git
cd telecom-churn-intelligence
```
2. Start Jupyter and open the notebook:
```bash
jupyter notebook src/churn_notebook.ipynb
# or
jupyter lab src/churn_notebook.ipynb
```
3. Run the notebook cells (recommended to run top-to-bottom). The notebook will:
   - Download and save the dataset to `data/Telco-Customer-Churn.csv`.
   - Create `outputs/` and write:
     - `churn_confusion_matrix.png`
     - `churn_feature_importance.png`
     - `customer_segments_manifest.csv`

4. (Optional) Run headless to execute the notebook and capture outputs:
```bash
jupyter nbconvert --to notebook --execute src/churn_notebook.ipynb --output executed_notebook.ipynb
```

Important: run the notebook from the repository root so the notebook's relative paths (`../data`, `../outputs`) resolve correctly.

## Notes & Reproducibility
- The notebook sets `random_state=42` for deterministic splitting and model training where applicable.
- Kernel metadata: Python 3; nbformat 4.
- Consider adding an explicit `requirements.txt` and a `runtime.txt` if you need to reproduce exact package versions.

## Customization ideas
- Hyperparameter tuning with cross-validation (GridSearchCV or RandomizedSearchCV).
- Try other models (XGBoost, LightGBM) and compare ROC/AUC.
- Add SHAP or LIME for local explainability (per-account explanations).
- Add automated model logging (MLflow) and serialization for deployment.

## Contributing
- Open an issue or PR with improvements, versioned requirements, or pipeline hardening.
- Add a LICENSE file if you want to make the project publicly reusable.

## License
No license file is present in this repository. Add a LICENSE to clarify reuse terms (MIT, Apache 2.0, etc.).

## Contact
Repository owner: ishanpote (https://github.com/ishanpote)
