
# Employee Attrition Prediction using Machine Learning

**Goal**: Predicting employee turnover and understanding the key drivers behind it using SHAP explainability


## Business Impact

* Helps HR teams identify at-risk employees early
* Supports retention planning:

  * Reduce overtime for specific roles
  * Invest in improving satisfaction and engagement
  * Focus retention efforts on younger employees
* Potential to **simulate attrition costs** and optimize workforce strategy

## About the Dataset

This project uses a fictional dataset created by IBM data scientists to simulate real-world HR data. The purpose is to uncover what factors contribute to employee attrition (i.e., leaving the company), and to help businesses proactively reduce turnover.

**Source**: [IBM HR Analytics Attrition Dataset (Kaggle)](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
**Rows**: 1470
**Target variable**: `Attrition` (Yes/No)

### Key Features:

* **Demographics**: Age, Gender, MaritalStatus, Education
* **Job Info**: Department, JobRole, YearsAtCompany, JobSatisfaction
* **Performance**: PerformanceRating, OverTime, MonthlyIncome
* **Label**: `Attrition` — whether the employee left or stayed


## Dashboard to visualize some features

![](/Assets/2025-06-22.png)
*dynamic dashboard can be found [here](https://public.tableau.com/app/profile/amr.alesseily/viz/EmployeeAttritionMLPrediction/Dashboard1?publish=yes) for more interactivity and info in the tooltip*

## Exploratory Data Analysis

* Checked class imbalance: \~16% of employees left (`Attrition=Yes`)
* Explored patterns by marital status, satisfaction levels, income, and overtime
* Removed irrelevant or duplicate columns (e.g., `EmployeeCount`, `Over18`)
* Identified clear correlations between attrition and:

  * Overtime work
  * Younger age
  * Low satisfaction
  * High distance from home


## Preprocessing & Feature Engineering

* **Label Mapping**: `Attrition`, `Gender`, `OverTime`, and `Over18` were mapped to binary values
* **Ordinal Encoding**: `BusinessTravel`
* **One-Hot Encoding**: Applied to `Department`, `EducationField`, `JobRole`, `MaritalStatus`
* **Scaling**: StandardScaler applied to continuous numerical features
* **Balancing**: Used `RandomOverSampler` to handle class imbalance in training data



## Modeling

Trained and evaluated six classification models:

| Model                  | Accuracy | Key Observations                            |
| ---------------------- | -------- | ------------------------------------------- |
| Logistic Regression    | 76%      | Good recall on attrition class (66%)        |
| K-Nearest Neighbors    | 67%      | Weak precision/recall for attrition         |
| Decision Tree          | 74%      | Low performance on class 1                  |
| **Random Forest**      | 85%      | High accuracy, weak recall for attrition    |
| Support Vector Machine | 60%      | High recall but very low precision          |
| **XGBoost**            | **86%**  | Best overall performance, selected for SHAP |



## Model Explainability with SHAP

Used SHAP (SHapley Additive exPlanations) to understand **why** the model predicts attrition.

### Key Drivers of Attrition:

| Feature                     | Insight                                                             |
| --------------------------- | ------------------------------------------------------------------- |
| **OverTime**                | Strongest predictor of attrition — frequent overtime increases risk |
| **Age**                     | Younger employees more likely to leave                              |
| **EnvironmentSatisfaction** | Dissatisfied employees more likely to leave                         |
| **DistanceFromHome**        | Longer commutes correlate with higher attrition                     |
| **JobSatisfaction**         | Low job satisfaction strongly tied to attrition                     |

**SHAP Summary Plot**:

![](/Assets/0d483497-fd04-427d-bcce-89bc9d992eb6.png)

* Shows direction and strength of impact for each feature
* Confirms that the model aligns with real-world HR logic


| 🔝 **Top Positive Drivers** (Increase Attrition Risk)                | 🔽 **Top Negative Drivers** (Reduce Attrition Risk)              |
| -------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **OverTime** – Working overtime increases risk.                      | **Older Age** – Older employees are more stable.                 |
| **Low Job Satisfaction** – Unhappy in role.                          | **Higher Monthly Income** – More financial comfort.              |
| **Low Environment Satisfaction** – Poor work environment.            | **Higher Stock Option Level** – Incentivized to stay.            |
| **More Companies Worked For** – Suggests instability.                | **Higher Job Involvement** – More engaged employees.             |
| **Low Relationship Satisfaction** – Poor manager/peer relationships. | **Longer Years With Current Manager** – Manager stability helps. |


**SHAP Dependence Plot**:

* Interaction between `DistanceFromHome` and `OverTime`:

  * Employees living close by are often assigned overtime
  * Those far away + overtime = higher attrition

---


# Web App: Employee Attrition Predictor (for HR decision making)

To make the solution actionable for HR teams, I developed a **Streamlit web app** that predicts attrition risk for individual employees based on input features like age, job satisfaction, overtime, income, and more.

**Features of the App:**

* Dynamic form for entering employee details
* Real-time prediction of attrition risk (Yes/No)
* Probability score showing model confidence
* Built-in explanations of the key risk and retention factors (based on SHAP findings)

**ML Backend:**

* Model: XGBoost (best performance, 86% accuracy)
* Preprocessing: One-hot encoding, scaling with `StandardScaler`
* Class balancing handled with `RandomOverSampler`

**Why It Matters:**

* Empowers HR teams to proactively assess and mitigate employee churn
* Encourages data-driven retention strategies
* Bridges the gap between machine learning output and business decision-making

**Screenshot:**
![](/Assets/streamlit_app.png)
*you can find the interactive tool [here](https://orchaid-ml-employee-attrit-scriptsemployee-attrition-app-wgyizd.streamlit.app/)*

**▶ To Run Locally:**

```bash
pip install streamlit pandas numpy scikit-learn joblib xgboost
streamlit run Employee_Attrition_App.py
```





