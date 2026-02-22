# Customer Churn Prediction

## Project Overview

This project builds an end-to-end machine learning pipeline to predict customer churn and translate predictions into actionable business strategy.

The objective is not only to build an accurate model, but to optimize decision thresholds, calibrate probabilities, and simulate a targeted retention campaign.


## Business Problem

Customer churn directly impacts revenue. Instead of treating all customers equally, businesses need to:

- Identify high-risk customers
- Prioritize retention efforts
- Optimize marketing spend
- Maximize churn capture within budget constraints

This project demonstrates how predictive modeling supports that decision-making process.

## Dataset

The dataset includes customer demographics, service usage, account information, and contract details.

Target variable:
- `Churn` (1 = customer churned, 0 = retained)


## Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Churn distribution analysis
- Feature relationships with churn
- Identification of high-risk patterns

### 2. Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling (for Logistic Regression)
- Train-test split

### 3. Baseline Model: Logistic Regression
- Trained as an interpretable baseline
- Evaluated using ROC-AUC, Recall, Precision

### 4. Advanced Model: XGBoost
- Hyperparameter tuning using cross-validation
- Improved ranking performance over baseline

### 5. Threshold Optimization
Instead of using the default 0.5 threshold, the decision threshold was optimized to target ~80% recall.

This aligns the model with business objectives:
- Capture majority of churners
- Control false positives

### 6. Probability Calibration
Calibrated probabilities using `CalibratedClassifierCV` to improve probability reliability.

This ensures predicted probabilities reflect true churn likelihood — critical for risk ranking and business targeting.

### 7. Model Comparison

| Model | ROC-AUC | Recall | Precision |
|-------|---------|--------|-----------|
| Logistic Regression | 0.835 | 0.797 | 0.491 |
| XGBoost (Tuned) | 0.841 | 0.813 | 0.503 |
| XGBoost (Calibrated, Re-optimized) | 0.842 | 0.802 | 0.521 |

Final model selected: **Calibrated XGBoost (Re-optimized)**

## Targeted Retention Strategy Simulation

To simulate a real-world campaign:

- Customers were ranked by predicted churn probability
- Top 20% highest-risk customers were selected

### Results

- Total churners in test set: 374
- Churners captured in top 20%: 186
- Capture rate: 49.7%
- Random 20% selection would capture: 75 churners
- Model delivers ~2.5x lift over random targeting

This demonstrates strong business impact.

## Key Insights

- Customers with month-to-month contracts are at higher churn risk
- Short tenure strongly correlates with churn
- Payment method influences churn likelihood
- Model ranking ability (ROC-AUC 0.842) enables effective prioritization

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn

## What Makes This Project Strong

This project goes beyond basic accuracy reporting by including:

- Hyperparameter tuning
- Threshold optimization
- Probability calibration
- Business-aligned evaluation
- Lift analysis and targeting simulation

It bridges machine learning and business strategy.

## How to Run

1. Clone the repository
2. Install required libraries
3. Run the notebook from top to bottom

The notebook is fully reproducible.

## Future Improvements

- Cost-sensitive learning
- SHAP-based interpretability
- Deployment-ready pipeline
- Real-time scoring simulation

