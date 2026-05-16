# Feature Learning Kernel Machines for Tabular Data: xRFM Evaluation

This project evaluates **xRFM** on tabular machine learning datasets and compares it against two strong baseline models: **Random Forest** and **XGBoost**. The goal is to test how well xRFM performs on both classification and regression tasks, while also comparing model accuracy, AUC, RMSE, training time, inference time, scalability, and interpretability.

## Project Overview

The notebook builds a full machine learning experimentation pipeline. It loads multiple tabular datasets, prepares the data, trains and tunes models, evaluates performance on a held-out test set, and records the results. The project also includes interpretability analysis to understand which features are most important for model predictions.

## Models Compared

The project compares three machine learning models:

- **Random Forest**
- **XGBoost**
- **xRFM**

Each model is tested using different hyperparameter settings. The best version of each model is selected based on validation performance before being evaluated on the final test set.

## Datasets

The experiments use five tabular datasets:

| Dataset | Task Type | Target Variable |
|---|---|---|
| Airline | Classification | satisfaction |
| E-commerce | Classification | Reached.on.Time_Y.N |
| Real Estate | Regression | price |
| Student | Regression | Final_Score |
| Superconduct | Regression | critical_temp |

## Methodology

The project follows these main steps:

1. Load each dataset and define the target variable.
2. Remove unnecessary columns such as IDs or unnamed index columns.
3. Split the data into training, validation, and test sets.
4. Preprocess numerical and categorical features.
5. Train Random Forest, XGBoost, and xRFM models.
6. Tune hyperparameters using validation performance.
7. Evaluate the best models on the test set.
8. Compare performance, training time, and inference time.
9. Run interpretability analysis on the e-commerce dataset.
10. Run scalability experiments using different training set sizes.

## Preprocessing

The preprocessing pipeline handles both numerical and categorical data:

- Missing numerical values are filled using the median.
- Numerical features are scaled using standardisation.
- Missing categorical values are filled using the most frequent value.
- Categorical features are encoded using one-hot encoding.
- Classification labels are encoded using `LabelEncoder`.

## Evaluation Metrics

Different metrics are used depending on the task type:

### Classification
- Accuracy
- AUC

### Regression
- RMSE

The project also records:

- Training time
- Inference time per sample

## Interpretability Analysis

The project includes an interpretability experiment on the e-commerce dataset. It compares different feature importance methods, including:

- PCA-based importance
- Mutual information
- Permutation importance
- xRFM AGOP/GOP diagonal importance

This helps compare the models not only by performance, but also by how well their behaviour can be understood.

## Scalability Experiment

A learning curve experiment is also included to test how the models perform as the training set size increases. The project compares:

- Test AUC vs training size
- Training time vs training size

This helps show how each model scales with more data.

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- xRFM
- Matplotlib
- Google Colab
- Jupyter Notebook

## Project Structure

```text
.
├── GroupAssignment.ipynb
├── data/
│   ├── airline.csv
│   ├── ecommerce.csv
│   ├── realestate.csv
│   ├── student.csv
│   └── superconduct.csv
├── results_tuned_all.csv
├── results_tuned_airline.csv
├── results_tuned_ecommerce.csv
├── results_tuned_realestate.csv
├── results_tuned_student.csv
├── results_tuned_superconduct.csv
├── learning_curve_ecommerce_auc.png
├── learning_curve_ecommerce_time.png
└── interpretability_ecommerce.png
```

## How to Run

1. Clone this repository.

```bash
git clone <your-repository-url>
```

2. Install the required libraries.

```bash
pip install pandas numpy scikit-learn xgboost matplotlib xrfm
```

3. Open the notebook.

```bash
jupyter notebook GroupAssignment.ipynb
```

4. Make sure the datasets are placed inside the `data/` folder.

5. Run the notebook cells in order.

## Key Skills Demonstrated

This project demonstrates skills in:

- Machine learning experimentation
- Data preprocessing
- Classification and regression modelling
- Hyperparameter tuning
- Model evaluation
- Runtime comparison
- Feature importance and interpretability
- Python-based data science workflow
- Working with tabular datasets

## Summary

Overall, this project compares xRFM with XGBoost and Random Forest across multiple tabular datasets. It focuses on both performance and interpretability, making it useful for understanding how different machine learning models behave on real-world structured data.
