import streamlit as st
import joblib
import pandas as pd
import numpy as np

model= joblib.load('XGBoost_model.pkl')
scaler= joblib.load('scaler.pkl')

st.set_page_config(page_title='Employee Attrition Predictor', layout='wide')


model_columns = [
    'Age', 'BusinessTravel', 'DailyRate', 'DistanceFromHome', 'Education',
    'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
    'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
    'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
    'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager', 'Department_Human Resources',
    'Department_Research & Development', 'Department_Sales',
    'EducationField_Human Resources', 'EducationField_Life Sciences',
    'EducationField_Marketing', 'EducationField_Medical',
    'EducationField_Other', 'EducationField_Technical Degree',
    'JobRole_Healthcare Representative', 'JobRole_Human Resources',
    'JobRole_Laboratory Technician', 'JobRole_Manager',
    'JobRole_Manufacturing Director', 'JobRole_Research Director',
    'JobRole_Research Scientist', 'JobRole_Sales Executive',
    'JobRole_Sales Representative', 'MaritalStatus_Divorced',
    'MaritalStatus_Married', 'MaritalStatus_Single'
]

# Numeric features to scale
features_to_scale = [
    'Age', 'DailyRate', 'DistanceFromHome', 'MonthlyIncome', 'MonthlyRate',
    'NumCompaniesWorked', 'TotalWorkingYears', 'PercentSalaryHike',
    'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager'
]



# --- Collect Input from User ---
st.markdown("### Enter Employee Information")

col1, col2, col3 = st.columns(3)

# --- Column 1 
with col1:
    age = st.slider('Age', 18, 60)
    daily_rate = st.number_input('Daily Rate', 100, 1500)
    distance = st.slider('Distance From Home', 1, 30)
    monthly_income = st.number_input('Monthly Income', 1000, 20000)
    monthly_rate = st.number_input('Monthly Rate', 1000, 20000)
    total_years = st.slider('Total Working Years', 0, 40)
    years_at_company = st.slider('Years at Company', 0, 40)

# --- Column 2 
with col2:
    education = st.slider('Education Level (1-5)', 1, 5)
    job_level = st.slider('Job Level (1-5)', 1, 5)
    job_involvement = st.slider('Job Involvement (1-4)', 1, 4)
    performance_rating = st.slider('Performance Rating (1-4)', 1, 4)
    stock_option = st.slider('Stock Option Level (0-3)', 0, 3)
    worklife_balance = st.slider('WorkLife Balance (1-4)', 1, 4)
    training_times = st.slider('Training Times Last Year', 0, 10)
    salary_hike = st.slider('Percent Salary Hike', 0, 25)

# --- Column 3 
with col3:
    years_in_role = st.slider('Years in Current Role', 0, 20)
    years_since_promo = st.slider('Years Since Last Promotion', 0, 15)
    years_with_manager = st.slider('Years With Current Manager', 0, 20)
    hourly_rate = st.number_input('Hourly Rate', 30, 150)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    business_travel = st.selectbox('Business Travel', ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
    department = st.selectbox('Department', ['Sales', 'Research & Development', 'Human Resources'])
    education_field = st.selectbox('Education Field', [
        'Life Sciences', 'Medical', 'Marketing', 'Technical Degree',
        'Human Resources', 'Other'
    ])
    job_role = st.selectbox('Job Role', [
        'Sales Executive', 'Research Scientist', 'Laboratory Technician',
        'Manufacturing Director', 'Healthcare Representative', 'Manager',
        'Sales Representative', 'Research Director', 'Human Resources'
    ])
    marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])

# --- Dedicated Column for your selected 5 important features ---
st.markdown("### Most Important Factors")
important_col = st.columns(1)[0]
with important_col:
    overtime = st.selectbox('OverTime', ['Yes', 'No'])
    job_satisfaction = st.slider('Job Satisfaction (1-4)', 1, 4)
    environment_satisfaction = st.slider('Environment Satisfaction (1-4)', 1, 4)
    num_companies = st.slider('Num Companies Worked', 0, 10)
    relationship_satisfaction = st.slider('Relationship Satisfaction (1-4)', 1, 4)




# --- raw input dict --- (for the dataframe)
raw_input = {
    'Age': age,
    'BusinessTravel': business_travel,
    'DailyRate': daily_rate,
    'DistanceFromHome': distance,
    'Education': education,
    'EnvironmentSatisfaction': environment_satisfaction,
    'Gender': gender,
    'HourlyRate': hourly_rate,
    'JobInvolvement': job_involvement,
    'JobLevel': job_level,
    'JobSatisfaction': job_satisfaction,
    'MonthlyIncome': monthly_income,
    'MonthlyRate': monthly_rate,
    'NumCompaniesWorked': num_companies,
    'OverTime': 1 if overtime == 'Yes' else 0,
    'PercentSalaryHike': salary_hike,
    'PerformanceRating': performance_rating,
    'RelationshipSatisfaction': relationship_satisfaction,
    'StockOptionLevel': stock_option,
    'TotalWorkingYears': total_years,
    'TrainingTimesLastYear': training_times,
    'WorkLifeBalance': worklife_balance,
    'YearsAtCompany': years_at_company,
    'YearsInCurrentRole': years_in_role,
    'YearsSinceLastPromotion': years_since_promo,
    'YearsWithCurrManager': years_with_manager,
    'Department': department,
    'EducationField': education_field,
    'JobRole': job_role,
    'MaritalStatus': marital_status
}


# --- convert to DataFrame ---
input_df = pd.DataFrame([raw_input])

# One-hot encode for expected features
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=model_columns, fill_value=0)  # Reindex to match training columns


# --- Apply scaler only to scaled columns ---

# --- Apply scaler only to the numeric columns that were scaled during training ---

# Apply scaling and assign it back
input_df[features_to_scale] = scaler.transform(input_df[features_to_scale])


# --- Predict ---
prediction = model.predict(input_df)
prob = model.predict_proba(input_df)[0][1]

st.markdown("---")
st.subheader("Prediction Result")
st.write(f'Attrition Risk: **{"Yes" if prediction[0] == 1 else "NO"}**')
st.caption(f"Model confidence: {prob:.2%}")

with st.expander(" Key Drivers of Employee Attrition (SHAP Summary)", expanded=False):
    st.markdown("""
    ### 🔝 Top Risk Factors (Increase Attrition Risk)
    - **OverTime** — Working overtime consistently increases risk.
    - **Low Job Satisfaction** — Dissatisfied employees are more likely to leave.
    - **Low Environment Satisfaction** — Poor workplace environment drives attrition.
    - **Many Companies Worked For** — Indicates potential job hopping or dissatisfaction.
    - **Low Relationship Satisfaction** — Weak coworker/manager relationships contribute to risk.

    ### 🔽 Top Retention Factors (Reduce Attrition Risk)
    - **Older Age** — Older employees tend to stay longer.
    - **Higher Monthly Income** — Financial satisfaction encourages retention.
    - **More Stock Options** — Incentives reduce turnover.
    - **High Job Involvement** — Engaged employees stay longer.
    - **Years With Current Manager** — Consistent leadership supports retention.
    """)

