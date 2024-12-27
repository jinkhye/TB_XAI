import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import shap
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Treatment Duration Prediction with XAI",
    page_icon="🕒",
)

# Title of the page
st.title("TB Treatment Duration Prediction with Explainable AI 🕒💊")

# Introduction Section with Collapsible Panels
with st.expander("About this Page"):
    st.markdown("""
    Welcome to the **Treatment Duration Prediction** page! Here, we use advanced machine learning models to predict the duration of TB treatment for patients, ensuring personalized and effective care management.
    """)

with st.expander("Model Used 🌳"):
    st.markdown("""
    The prediction is powered by a **Random Forest Regressor**, a robust and versatile machine learning algorithm known for its accuracy and ability to handle complex datasets.
    """)

with st.expander("Model Performance Metrics 📈"):
    st.markdown("""
    Our machine learning model has been evaluated using the following metrics:

    - **Mean Absolute Error (MAE):** 50.45 days
    - **Root Mean Squared Error (RMSE):** 67.92 days
    - **R-squared (R²):** 0.51

    These metrics indicate that the model provides reasonably accurate predictions while accounting for variability in treatment durations.
    """)

with st.expander("How It Works 🔍"):
    st.markdown("""
    - **Input Data:** Enter patient-specific information such as demographics, medical history, and clinical parameters.
    - **Model Prediction:** The model processes the input data and predicts the expected treatment duration in days.
    - **Explainable AI (XAI):** Understand the factors driving the prediction with interactive explanations to ensure transparency and trust.
    """)

with st.expander("Why is this Important? 🌟"):
    st.markdown("""
    - **Personalized Care:** Provides tailored treatment plans for patients.
    - **Resource Optimization:** Helps healthcare providers allocate resources effectively.
    - **Improved Outcomes:** Early identification of potential challenges can improve treatment success rates.

    Use the sidebar to input patient details and view predictions. Let's optimize TB treatment durations together! 💡
    """)

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/TB Cleaned 2.csv")

data = load_data()

# Outlier Removal Function
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_cleaned = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_cleaned

# Remove outliers for Treatment Duration
data_cleaned = remove_outliers(data, 'Treatment_duration')

data_cleaned = data_cleaned.rename(
    columns={
        "Bronchoalveolar_Lavage_AFB_Direct_Smear": "BAL_AFB_Smear",
        "Bronchoalveolar_Lavage_MTB_Culture": "BAL_MTB_Culture",
        "Bronchoalveolar_Lavage_MTB_Culture_SensitivitySensitive": "BAL_MTB_Sensitivity",
        "Cerebrospinal_Fluid_AFB_Direct_Smear": "CSF_AFB_Smear",
        "Cerebrospinal_Fluid_MTB_Culture": "CSF_MTB_Culture",
    }
)

# Pearson Correlation Function
def calculate_pearson_correlation(dataframe, target_variable):
    results = []
    for column in dataframe.drop(columns=[target_variable], errors='ignore').columns:
        # Skip constant columns
        if dataframe[column].nunique() == 1:
            continue
        corr_value, p_value = pearsonr(dataframe[column], dataframe[target_variable])
        results.append({'Feature': column, 'PearsonR': corr_value, 'PValue': p_value})
    correlation_df = pd.DataFrame(results)
    significant_features = correlation_df[correlation_df['PValue'] < 0.05]['Feature'].tolist()
    return significant_features

# Get significant features using Pearson correlation
pearson_features = calculate_pearson_correlation(data_cleaned, 'Treatment_duration')

# Filter dataset for Pearson-selected features
data_pearson = data_cleaned[pearson_features + ['Treatment_duration']]

# Preprocessing
X = data_pearson.drop(columns=['Treatment_duration'], errors='ignore')
y = data_pearson['Treatment_duration']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=382,
    max_depth=10,
    min_samples_split=8,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate Model
y_test_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

# SHAP Explanation
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test, check_additivity=False)

# Mappings for Categorical Features
# Updated Mappings for Categorical Features
mappings = {
    "Treatment_Outcome": {
        1: "Completed",
        2: "Defaulted",
        4: "Transfer out",
        5: "Died",
        6: "Change diagnosis",
        7: "Others",
    },
    "M_akurit": {
        0: "No",
        1: "Yes",
    },
    "Gender": {
        1: "Male",
        2: "Female",
    },
    "Nationality": {
        1: "Malaysian",
        2: "Others",
    },
    "Occupation_category": {
        1: "Managers",
        2: "Professionals",
        3: "Technicians",
        4: "Clerical support workers",
        5: "Service and sales",
        6: "Skilled agricultural",
        7: "Craft and related trades",
        8: "Plant and machine operators",
        9: "Elementary occupations",
        10: "Armed forces",
        11: "Unemployed",
        12: "Student",
        13: "Not known",
    },
    "BCG_Scar": {
        0: "No Scar",
        1: "Scar Present",
        2: "Not Applicable",
    },
    "BCG_scar_2": {
        1: "Yes",
        2: "No",
    },
    "Use_of_Illicit_Drugs": {
        0: "Never",
        1: "Past",
        2: "Current",
        3: "Not Applicable"
    },
    "Chronic_Kidney_Disease": {
        0: "No",
        1: "Yes",
    },
    "eGFR_group": {
        0: "Not applicable",
        1: "Mildly Decreased",
        2: "Moderately Decreased",
    },
    "Liver_Disease": {
        0: "No",
        1: "YesA",
        2: "YesB",
        3: "YesC",
    },
    "Cancer": {
        0: "No",
        1: "Yes",
    },
    "TST": {
        0: "Not Done",
        1: "Done",
    },
    "Sputum_AFB_Direct_Smear": {
        0: "Not Done",
        1: "Positive",
        2: "Negative",
        3: "Borderline",
    },
    "Sputum_MTB_Culture": {
        0: "Not Done",
        1: "Positive",
        2: "Negative",
        3: "Contamination",
        4: "NA",
    },
    "Sputum_MTB_Culture_Sensitivity": {
        0: "Not Done",
        1: "Sensitive to EHRZ",
        2: "Monodrug resistant",
        3: "Polydrug resistant",
        4: "Rifampicin resistant",
        5: "Multidrug resistant",
        6: "Contamination",
        7: "NA",
    },
    "BAL_AFB_Smear": {
        1: "Positive",
        2: "Negative",
        3: "Not Done",
        4: "NA",
    },
    "BAL_MTB_Culture": {
        1: "Positive",
        2: "Negative",
        3: "Not Done",
        4: "NA",
    },
    "BAL_MTB_Sensitivity": {
        0: "Not Done",
        1: "Sensitive to EHRZ",
        2: "Monodrug resistant",
        3: "Polydrug resistant",
        4: "Rifampicin resistant",
        5: "Multidrug resistant",
        6: "Contamination",
        7: "NA",
    },
    "CSF_AFB_Smear": {
        1: "Not Done",
        2: "Negative",
        3: "NA"
    },
    "CSF_MTB_Culture": {
        1: "Not Done",
        2: "Negative",
        3: "Positive",
        4: "NA",
    },
    "Biopsy_use": {
        1: "Yes",
        2: "No",
        3: "NA",
    },
    "Biopsy_Site": {
        1: "Lung",
        2: "Lymph Node",
        3: "Other",
    },
    "AFB_Stain": {
        0: "NA",
        1: "Positive",
        2: "Negative",
    },
    "Granulomatous_Inflammation": {
        0: "NA",
        1: "Yes",
        2: "No",
    },
    "Caseating_Necrosis": {
        0: "No lesion",
        1: "Minimal",
        2: "Moderate",
        3: "Advanced",
        4: "Not done",
    },
    "Chest_XRay_CXR_Findings": {
        1: "No lesion",
        2: "Minimal",
        3: "Moderate and far advanced",
        4: "Not done",
    },
    "CXR_2": {
        1: "No lesion",
        2: "Minimal",
        3: "Moderate and far advanced",
        4: "Not done",
    },
    "Symptoms_LOA": {
        0: "No",
        1: "Yes",
    },
    "Symptoms_Fever": {
        0: "No",
        1: "Yes",
    },
    "Sputum_smear_clean": {
        0: "NA",
        1: "Positive",
        2: "Negative",
    },
    "Intensive_Phase_Regime": {
        0: "None",
        1: "Akurit4 Only",
        9: "All",
    },
    "Intensive_akurit4": {
        0: "No",
        1: "Yes",
    },
    "Intensive_rifampicin": {
        0: "No",
        1: "Yes",
    },
    "Maintenance_Phase_Regime": {
        0: "None",
        1: "Yes",
    },
    "M_ethambutol": {
        0: "No",
        1: "Yes",
    },
    "M_levofloxacin": {
        0: "No",
        1: "Yes",
    },
    "Where_was_the_treatment_started": {
        1: "Inpatient",
        2: "Outpatient",
        3: "NA",
    },
    "Treatment_status": {
        0: "NA",
        1: "Favourable",
        2: "Unfavourable",
        3: "Ongoing treatment",
        4: "Transfer out",
    },
}

# Feature selection and interactive input with defaults
numeric_features = ['Age', 'Year_of_birth', 'Weightkg', 'Number_of_comorbids', 'Number_of_immunosuppressant']
all_features = numeric_features + list(mappings.keys())

# Allow user to select features to input
selected_features = st.multiselect(
    "Select features to input:", all_features, default=all_features
)

# Interactive Prediction in the Sidebar with Feature Selection
st.sidebar.write("### Input Patient Data")
user_input = {}

# Add numeric input fields for selected numeric features
for feature in numeric_features:
    if feature in X.columns:  # Ensure the feature exists in the model
        if feature in selected_features:
            user_input[feature] = st.sidebar.number_input(
                f"Enter value for {feature}", value=int(X[feature].mean()))
        else:
            user_input[feature] = X[feature].mean()  # Default to mean

# Add dropdowns for selected categorical features using mappings
for feature, mapping in mappings.items():
    if feature in X.columns:  # Ensure the feature exists in the model
        if feature in selected_features:
            user_input[feature] = st.sidebar.selectbox(
                f"Select value for {feature}", options=mapping.values()
            )
        else:
            # Default to mode (most frequent category)
            mode_value = X[feature].mode()[0]
            user_input[feature] = mode_value

# Convert dropdown choices back to numeric for prediction
for feature, mapping in mappings.items():
    if feature in user_input:
        reverse_mapping = {v: int(k) for k, v in mapping.items()}  # Ensure keys are Python integers
        user_input[feature] = reverse_mapping.get(user_input[feature], user_input[feature])  # Default to input value if key is missing

# Convert user input into DataFrame and align columns with X_train
user_input_df = pd.DataFrame([user_input])
user_input_df = user_input_df[X.columns]  # Ensure the same column order as X_train

# Debugging step: Display user input for verification
st.write("### User Input Data")
st.write(user_input_df)

# Prediction and SHAP Explanation
if st.sidebar.button("Predict and Explain"):
    # Predict the treatment duration for user input
    prediction = model.predict(user_input_df)[0]
    st.markdown(f"""
        ## 🎉 Prediction Result
        ### 🕒 Predicted Treatment Duration: **{prediction:.2f} days**
    """)
    # Generate SHAP values for the user input
    shap_values_input = explainer(user_input_df, check_additivity=False)

    # SHAP Waterfall Plot for User Input
    st.write("#### SHAP Explanation for Input Data (Waterfall Plot)")
    fig_waterfall = plt.figure()
    shap.waterfall_plot(shap_values_input[0], max_display=10)
    st.pyplot(fig_waterfall)

    st.write("### SHAP Values for Input Features")
    shap_df = pd.DataFrame({
        "Feature": user_input_df.columns,
        "SHAP Value": shap_values_input.values[0]
    }).sort_values(by="SHAP Value", key=abs, ascending=False)

    st.table(shap_df)




# st.write("#### SHAP Summary Plot")
# fig_summary = plt.figure()
# shap.summary_plot(shap_values, X_test, show=False, plot_size=[12,8])
# st.pyplot(fig_summary)

# # Extract mean absolute SHAP values for each feature
# shap_values_mean = np.abs(shap_values.values).mean(axis=0)
# shap_feature_importance = pd.DataFrame({
#     'Feature': X_test.columns,
#     'Mean_SHAP_Value': shap_values_mean
# }).sort_values(by='Mean_SHAP_Value', ascending=False)

# # Plot with Seaborn and add data labels
# fig, ax = plt.subplots(figsize=(10, 8))
# sns.barplot(
#     x='Mean_SHAP_Value', 
#     y='Feature', 
#     data=shap_feature_importance, 
#     palette='viridis', 
#     ax=ax
# )

# # Add data labels
# for i, value in enumerate(shap_feature_importance['Mean_SHAP_Value']):
#     ax.text(value, i, f"{value:.2f}", va='center', ha='left', fontsize=10)

# ax.set_title("SHAP Feature Importance (with Data Labels)", fontsize=16)
# ax.set_xlabel("Mean Absolute SHAP Value", fontsize=14)
# ax.set_ylabel("Feature", fontsize=14)

# st.pyplot(fig)