import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import shap
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Treatment Duration Prediction with XAI",
    page_icon="🕒",
    layout="wide",
)

# Title of the page
st.title("TB Treatment Duration Prediction with Explainable AI 🕒💊")

st.markdown("""
Welcome to the **Treatment Duration Prediction** page! Here, we use advanced machine learning models to predict the duration of TB treatment for patients, ensuring personalized and effective care management.

---
            
### How It Works 🔍
- **Input Data:** Enter patient-specific information such as demographics, medical history, and clinical parameters.
- **Model Prediction:** The model processes the input data and predicts the expected treatment duration in days.
- **Explainable AI (XAI):** Understand the factors driving the prediction with interactive explanations to ensure transparency and trust.

**Choose a prediction model at the sidebar to experiment with different approaches.**
            
---
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

# Constants for model parameters
SVR_PARAMS = {'C': 2.1586224623950696, 'epsilon': 0.5720233245417916, 'kernel': 'linear'}
RF_PARAMS = {'max_depth': 10, 'min_samples_split': 9, 'n_estimators': 393}
GBR_PARAMS = {'learning_rate': 0.012970917280826846, 'max_depth': 3, 'n_estimators': 443}

# Model Selection
st.sidebar.header("🧠 Select a Model")
model_choice = st.sidebar.selectbox(
    "Choose a model:",
    ["Support Vector Regressor", "Random Forest", "Gradient Boosting Regressor"],
    index=1  # Set the default selection to "Random Forest"
)

# Add to Sidebar
st.sidebar.header("Model Details and Insights")

# Display model-specific details dynamically
with st.sidebar.expander("Model Used 🌟"):
    if model_choice == "Support Vector Regressor":
        st.markdown(f"""
        The prediction is powered by a **Support Vector Regressor (SVR)**, a model known for its ability to capture complex relationships in data. This model has been tuned with the following parameters:
        
        - **Kernel:** {SVR_PARAMS['kernel']}
        - **C (Regularization):** {SVR_PARAMS['C']}
        - **Epsilon:** {SVR_PARAMS['epsilon']}
        """)
    elif model_choice == "Random Forest":
        st.markdown(f"""
        The prediction is powered by a **Random Forest Regressor**, a robust and versatile machine learning algorithm known for its accuracy and ability to handle complex datasets. This model has been tuned with the following parameters:
        
        - **Number of Estimators (n_estimators):** {RF_PARAMS['n_estimators']}
        - **Maximum Depth (max_depth):** {RF_PARAMS['max_depth']}
        - **Minimum Samples Split (min_samples_split):** {RF_PARAMS['min_samples_split']}
        """)
    elif model_choice == "Gradient Boosting Regressor":
        st.markdown(f"""
        The prediction is powered by a **Gradient Boosting Regressor**, an advanced ensemble model that builds sequential trees to minimize errors. This model has been tuned with the following parameters:
        
        - **Learning Rate:** {GBR_PARAMS['learning_rate']}
        - **Number of Estimators (n_estimators):** {GBR_PARAMS['n_estimators']}
        - **Maximum Depth (max_depth):** {GBR_PARAMS['max_depth']}
        """)

# Model performance metrics
SVR_METRICS = {'MAE': 45.2470, 'RMSE': 65.4471, 'R2': 0.5433}
RF_METRICS = {'MAE': 50.4543, 'RMSE': 67.9234, 'R2': 0.5081}
GBR_METRICS = {'MAE': 49.1775, 'RMSE': 67.6759, 'R2': 0.5116}

# Model performance metrics section
with st.sidebar.expander("Model Performance Metrics 📈"):
    if model_choice == "Support Vector Regressor":
        st.markdown(f"""
        **Support Vector Regressor (SVR)** performance on the test set:
        
        - **Mean Absolute Error (MAE):** {SVR_METRICS['MAE']} days
        - **Root Mean Squared Error (RMSE):** {SVR_METRICS['RMSE']} days
        - **R-squared (R²):** {SVR_METRICS['R2']}
        
        These results indicate that SVR performs well in predicting treatment durations, with a good balance between error and model fit.
        """)
    elif model_choice == "Random Forest":
        st.markdown(f"""
        **Random Forest Regressor (RF)** performance on the test set:
        
        - **Mean Absolute Error (MAE):** {RF_METRICS['MAE']} days
        - **Root Mean Squared Error (RMSE):** {RF_METRICS['RMSE']} days
        - **R-squared (R²):** {RF_METRICS['R2']}
        
        Random Forest is a reliable model with reasonably accurate predictions for treatment durations.
        """)
    elif model_choice == "Gradient Boosting Regressor":
        st.markdown(f"""
        **Gradient Boosting Regressor (GBR)** performance on the test set:
        
        - **Mean Absolute Error (MAE):** {GBR_METRICS['MAE']} days
        - **Root Mean Squared Error (RMSE):** {GBR_METRICS['RMSE']} days
        - **R-squared (R²):** {GBR_METRICS['R2']}
        
        Gradient Boosting Regressor demonstrates a good trade-off between predictive accuracy and generalization.
        """)

# Why is this Important Section
with st.sidebar.expander("Why is this Important? 🌟"):
    st.markdown("""
    - **Personalized Care:** Provides tailored treatment plans for patients.
    - **Resource Optimization:** Helps healthcare providers allocate resources effectively.
    - **Improved Outcomes:** Early identification of potential challenges can improve treatment success rates.

    Use the input section on the main page to enter patient details and view predictions. Let's optimize TB treatment durations together! 💡
    """)


# Initialize scaler for SVR
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

if model_choice == "Support Vector Regressor":
    # Define numerical features for scaling (used in SVR)
    numeric_features = [
        "Age", "Year_of_birth", "Number_of_comorbids"
    ]

    # Scale numerical features for SVR
    scaler = StandardScaler()
    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

    # Initialize and train SVR
    model = SVR(C=SVR_PARAMS['C'], epsilon=SVR_PARAMS['epsilon'], kernel=SVR_PARAMS['kernel'])
    model.fit(X_train_scaled, y_train)

elif model_choice == "Random Forest":
    # Initialize and train Random Forest
    model = RandomForestRegressor(
        n_estimators=RF_PARAMS['n_estimators'],
        max_depth=RF_PARAMS['max_depth'],
        min_samples_split=RF_PARAMS['min_samples_split'],
        random_state=42
    )
    model.fit(X_train, y_train)

elif model_choice == "Gradient Boosting Regressor":
    # Initialize and train Gradient Boosting Regressor
    model = GradientBoostingRegressor(
        learning_rate=GBR_PARAMS['learning_rate'],
        n_estimators=GBR_PARAMS['n_estimators'],
        max_depth=GBR_PARAMS['max_depth'],
        random_state=42
    )
    model.fit(X_train, y_train)

# Mappings for Categorical Features
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

# Input Patient Data with Logical Grouping
st.markdown("## Input Patient Data")

# Create an empty dictionary to hold user inputs
user_input = {}

# Reverse mappings for converting string selections back to numerical values
reverse_mappings = {key: {v: k for k, v in value.items()} for key, value in mappings.items()}

# File Upload Option
uploaded_file = st.file_uploader(
    "Upload a file to autofill patient data (CSV or Excel format)", 
    type=["csv", "xlsx"],
    help="Upload a CSV or Excel file with patient data to populate the fields automatically."
)

def align_columns(input_df, reference_columns):
    aligned_df = pd.DataFrame(index=input_df.index, columns=reference_columns)
    for col in reference_columns:
        if col in input_df.columns:
            aligned_df[col] = input_df[col]
        else:
            aligned_df[col] = 0  # Fill missing columns with 0
    return aligned_df

uploaded_data = None
missing_features = []  # To track missing features
if uploaded_file is not None:
    try:
        # Load data
        if uploaded_file.name.endswith('.csv'):
            uploaded_data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            uploaded_data = pd.read_excel(uploaded_file)

        st.write("Uploaded data preview:")
        st.write(uploaded_data.head())

        # Align the user input DataFrame with X columns
        user_input_df = align_columns(uploaded_data, X.columns)

        # Check for missing features
        missing_features = [col for col in X.columns if col not in uploaded_data.columns]
        if missing_features:
            st.warning(f"The uploaded file is missing the following columns: {missing_features}")
            
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
else:
    # Continue with manual input
    st.markdown("### Demographics")

    user_input['Age'] = st.number_input(
        "Age", 
        help="The age of the patient.", 
        value=30
    )

    user_input['Year_of_birth'] = st.number_input(
        "Year of Birth", 
        help="The year the patient was born.", 
        value=1990
    )

    user_input['Nationality'] = reverse_mappings['Nationality'][
        st.selectbox(
            "Nationality",
            list(mappings['Nationality'].values()),
            help="The nationality of the patient."
        )
    ]

    user_input['Occupation_category'] = reverse_mappings['Occupation_category'][
        st.selectbox(
            "Occupation Category",
            list(mappings['Occupation_category'].values()),
            help="The patient's occupation category."
        )
    ]

    # --- Clinical History ---
    st.markdown("### Clinical History")

    user_input['Chronic_Kidney_Disease'] = reverse_mappings['Chronic_Kidney_Disease'][
        st.selectbox(
            "Chronic Kidney Disease",
            list(mappings['Chronic_Kidney_Disease'].values()),
            help="Indicates if the patient has chronic kidney disease."
        )
    ]

    user_input['Liver_Disease'] = reverse_mappings['Liver_Disease'][
        st.selectbox(
            "Liver Disease",
            list(mappings['Liver_Disease'].values()),
            help="Indicates the type of liver disease, if any."
        )
    ]

    user_input['Cancer'] = reverse_mappings['Cancer'][
        st.selectbox(
            "Cancer",
            list(mappings['Cancer'].values()),
            help="Indicates if the patient has a history of cancer."
        )
    ]

    user_input['Number_of_comorbids'] = st.number_input(
        "Number of Comorbidities",
        help="The number of other medical conditions the patient has.",
        value=0
    )

    user_input['Use_of_Illicit_Drugs'] = reverse_mappings['Use_of_Illicit_Drugs'][
        st.selectbox(
            "Use of Illicit Drugs",
            list(mappings['Use_of_Illicit_Drugs'].values()),
            help="The patient's history of illicit drug use."
        )
    ]

    # --- Diagnostics ---
    st.markdown("### Diagnostics")

    user_input['TST'] = reverse_mappings['TST'][
        st.selectbox(
            "TST (Tuberculin Skin Test)",
            list(mappings['TST'].values()),
            help="Indicates if the Tuberculin Skin Test was performed."
        )
    ]

    user_input['Sputum_AFB_Direct_Smear'] = reverse_mappings['Sputum_AFB_Direct_Smear'][
        st.selectbox(
            "Sputum AFB Direct Smear",
            list(mappings['Sputum_AFB_Direct_Smear'].values()),
            help="The result of the Sputum AFB Direct Smear test."
        )
    ]

    user_input['Sputum_MTB_Culture'] = reverse_mappings['Sputum_MTB_Culture'][
        st.selectbox(
            "Sputum MTB Culture",
            list(mappings['Sputum_MTB_Culture'].values()),
            help="The result of the Sputum MTB Culture test."
        )
    ]

    user_input['Sputum_MTB_Culture_Sensitivity'] = reverse_mappings['Sputum_MTB_Culture_Sensitivity'][
        st.selectbox(
            "Sputum MTB Culture Sensitivity",
            list(mappings['Sputum_MTB_Culture_Sensitivity'].values()),
            help="The sensitivity results of the Sputum MTB Culture test."
        )
    ]

    user_input['BAL_AFB_Smear'] = reverse_mappings['BAL_AFB_Smear'][
        st.selectbox(
            "BAL AFB Smear",
            list(mappings['BAL_AFB_Smear'].values()),
            help="The result of the Bronchoalveolar Lavage (BAL) AFB smear test."
        )
    ]

    user_input['BAL_MTB_Culture'] = reverse_mappings['BAL_MTB_Culture'][
        st.selectbox(
            "BAL MTB Culture",
            list(mappings['BAL_MTB_Culture'].values()),
            help="The result of the BAL MTB Culture test."
        )
    ]

    user_input['BAL_MTB_Sensitivity'] = reverse_mappings['BAL_MTB_Sensitivity'][
        st.selectbox(
            "BAL MTB Sensitivity",
            list(mappings['BAL_MTB_Sensitivity'].values()),
            help="The sensitivity results of the BAL MTB Culture test."
        )
    ]

    user_input['CSF_AFB_Smear'] = reverse_mappings['CSF_AFB_Smear'][
        st.selectbox(
            "CSF AFB Smear",
            list(mappings['CSF_AFB_Smear'].values()),
            help="The result of the Cerebrospinal Fluid (CSF) AFB smear test."
        )
    ]

    user_input['CSF_MTB_Culture'] = reverse_mappings['CSF_MTB_Culture'][
        st.selectbox(
            "CSF MTB Culture",
            list(mappings['CSF_MTB_Culture'].values()),
            help="The result of the CSF MTB Culture test."
        )
    ]

    user_input['Granulomatous_Inflammation'] = reverse_mappings['Granulomatous_Inflammation'][
        st.selectbox(
            "Granulomatous Inflammation",
            list(mappings['Granulomatous_Inflammation'].values()),
            help="Indicates the presence of granulomatous inflammation."
        )
    ]

    user_input['Caseating_Necrosis'] = reverse_mappings['Caseating_Necrosis'][
        st.selectbox(
            "Caseating Necrosis",
            list(mappings['Caseating_Necrosis'].values()),
            help="The level of caseating necrosis observed in the patient."
        )
    ]

    user_input['Chest_XRay_CXR_Findings'] = reverse_mappings['Chest_XRay_CXR_Findings'][
        st.selectbox(
            "Chest X-Ray Findings",
            list(mappings['Chest_XRay_CXR_Findings'].values()),
            help="Findings from the patient's chest X-ray."
        )
    ]

    user_input['CXR_2'] = reverse_mappings['CXR_2'][
        st.selectbox(
            "Chest X-Ray 2",
            list(mappings['CXR_2'].values()),
            help="Additional findings from a follow-up chest X-ray."
        )
    ]

    user_input['BCG_Scar'] = reverse_mappings['BCG_Scar'][
        st.selectbox(
            "BCG Scar",
            list(mappings['BCG_Scar'].values()),
            help="Indicates the presence of a BCG vaccination scar."
        )
    ]

    user_input['BCG_scar_2'] = reverse_mappings['BCG_scar_2'][
        st.selectbox(
            "BCG Scar 2",
            list(mappings['BCG_scar_2'].values()),
            help="An additional indicator for the BCG scar presence."
        )
    ]

    user_input['eGFR_group'] = reverse_mappings['eGFR_group'][
        st.selectbox(
            "eGFR Group",
            list(mappings['eGFR_group'].values()),
            help="The estimated glomerular filtration rate (eGFR) group for kidney function."
        )
    ]

    user_input['Sputum_smear_clean'] = reverse_mappings['Sputum_smear_clean'][
        st.selectbox(
            "Sputum Smear Clean",
            list(mappings['Sputum_smear_clean'].values()),
            help="The result of the clean sputum smear test."
        )
    ]

    user_input['Biopsy_use'] = reverse_mappings['Biopsy_use'][
        st.selectbox(
            "Biopsy Use",
            list(mappings['Biopsy_use'].values()),
            help="Indicates whether a biopsy was performed for diagnostic purposes."
        )
    ]

    user_input['Biopsy_Site'] = reverse_mappings['Biopsy_Site'][
        st.selectbox(
            "Biopsy Site",
            list(mappings['Biopsy_Site'].values()),
            help="The site from which tissue was collected for biopsy."
        )
    ]

    user_input['AFB_Stain'] = reverse_mappings['AFB_Stain'][
        st.selectbox(
            "AFB Stain",
            list(mappings['AFB_Stain'].values()),
            help="The result of the Acid-Fast Bacilli (AFB) stain test, used to identify the presence of TB bacteria."
        )
    ]

    # --- Symptoms ---
    st.markdown("### Symptoms and Observations")

    user_input['Symptoms_LOA'] = reverse_mappings['Symptoms_LOA'][
        st.selectbox(
            "Loss of Appetite (LOA)",
            list(mappings['Symptoms_LOA'].values()),
            help="Indicates if the patient has experienced a loss of appetite."
        )
    ]

    user_input['Symptoms_Fever'] = reverse_mappings['Symptoms_Fever'][
        st.selectbox(
            "Fever",
            list(mappings['Symptoms_Fever'].values()),
            help="Indicates if the patient has experienced a fever."
        )
    ]

    # --- Treatment Details ---
    st.markdown("### Treatment Details")

    user_input['Treatment_Outcome'] = reverse_mappings['Treatment_Outcome'][
        st.selectbox(
            "Treatment Outcome",
            list(mappings['Treatment_Outcome'].values()),
            help="The outcome of the patient's previous treatment."
        )
    ]

    user_input['Treatment_status'] = reverse_mappings['Treatment_status'][
        st.selectbox(
            "Treatment Status",
            list(mappings['Treatment_status'].values()),
            help="The current status of the patient's treatment."
        )
    ]

    user_input['Where_was_the_treatment_started'] = reverse_mappings['Where_was_the_treatment_started'][
        st.selectbox(
            "Where was the Treatment Started?",
            list(mappings['Where_was_the_treatment_started'].values()),
            help="Indicates whether the treatment started as an inpatient or outpatient."
        )
    ]

    user_input['Intensive_Phase_Regime'] = reverse_mappings['Intensive_Phase_Regime'][
        st.selectbox(
            "Intensive Phase Regime",
            list(mappings['Intensive_Phase_Regime'].values()),
            help="The type of intensive phase treatment regime."
        )
    ]

    user_input['Intensive_akurit4'] = reverse_mappings['Intensive_akurit4'][
        st.selectbox(
            "Intensive Akurit4 (Intensive Phase)",
            list(mappings['Intensive_akurit4'].values()),
            help="Indicates if the patient is on Akurit4 in the intensive phase."
        )
    ]

    user_input['Intensive_rifampicin'] = reverse_mappings['Intensive_rifampicin'][
        st.selectbox(
            "Rifampicin (Intensive Phase)",
            list(mappings['Intensive_rifampicin'].values()),
            help="Indicates if the patient is taking Rifampicin in the intensive phase."
        )
    ]

    user_input['Maintenance_Phase_Regime'] = reverse_mappings['Maintenance_Phase_Regime'][
        st.selectbox(
            "Maintenance Phase Regime",
            list(mappings['Maintenance_Phase_Regime'].values()),
            help="Indicates if the patient is in the maintenance phase of treatment."
        )
    ]

    user_input['M_akurit'] = reverse_mappings['M_akurit'][
        st.selectbox(
            "Akurit (Maintenance Phase)",
            list(mappings['M_akurit'].values()),
            help="Indicates if the patient is taking Akurit medication in the maintenance phase."
        )
    ]

    user_input['M_ethambutol'] = reverse_mappings['M_ethambutol'][
        st.selectbox(
            "Ethambutol (Maintenance Phase)",
            list(mappings['M_ethambutol'].values()),
            help="Indicates if the patient is taking Ethambutol in the maintenance phase."
        )
    ]

    user_input['M_levofloxacin'] = reverse_mappings['M_levofloxacin'][
        st.selectbox(
            "Levofloxacin (Maintenance Phase)",
            list(mappings['M_levofloxacin'].values()),
            help="Indicates if the patient is taking Levofloxacin in the maintenance phase."
        )
    ]

    # Convert user input into DataFrame
    user_input_df = align_columns(pd.DataFrame([user_input]), X.columns)

if model_choice == "Support Vector Regressor":
    user_input_df[numeric_features] = scaler.transform(user_input_df[numeric_features])

# Ensure that the DataFrame is properly aligned
if set(user_input_df.columns) != set(X.columns):
    st.error("The input data does not match the expected structure. Please verify your inputs.")
else:
    # Prediction and SHAP Explanation
    if st.button("Predict and Explain"):
        # Predict the treatment duration for user input
        prediction = model.predict(user_input_df)[0]
        st.markdown(f"""
            ## 🎉 Prediction Result
            ### 🕒 Predicted Treatment Duration: **{prediction:.2f} days**
            ---
        """)
        
        # Generate SHAP values for the user input
        # SHAP Explanation
        explainer = shap.Explainer(model, X_train_scaled if model_choice == "Support Vector Regressor" else X_train)
        shap_values_input = explainer(user_input_df)

        # SHAP Waterfall Plot for User Input
        st.write("#### SHAP Explanation for Input Data (Waterfall Plot)")
        fig_waterfall = plt.figure()
        shap.waterfall_plot(shap_values_input[0], max_display=10)
        st.pyplot(fig_waterfall)

        # Extract SHAP values and features
        shap_df = pd.DataFrame({
            "Feature": user_input_df.columns,
            "SHAP Value": shap_values_input.values[0]
        }).sort_values(by="SHAP Value", key=abs, ascending=False)

        # Generate Text-Based Explanation
        top_features = shap_df.head(5)
        explanation_text = "### What does this mean?\n"
        explanation_text += (
            f"The model predicts a treatment duration of **{prediction:.2f} days**. "
            "Here are the top factors influencing this prediction:\n\n"
        )

        for index, row in top_features.iterrows():
            feature = row["Feature"]
            shap_value = row["SHAP Value"]
            direction = "increased" if shap_value > 0 else "reduced"
            days = abs(shap_value)
            explanation_text += f"- **{feature}**: {direction.title()} duration by **{days:.2f} days**.\n"

        # Highlight the most influential feature
        top_feature = top_features.iloc[0]
        top_feature_name = top_feature["Feature"]
        top_feature_impact = abs(top_feature["SHAP Value"])
        explanation_text += (
            f"\nThe most significant factor is **{top_feature_name}**, "
            f"affecting the duration by **{top_feature_impact:.2f} days**.\n"
        )

        st.markdown(explanation_text)

        st.markdown("""---""")

        # Display SHAP values as a table with a clear header
        st.write("### Full SHAP Contributions for All Features")
        st.write(
            """
            Below is the full list of how each feature contributed to the predicted treatment duration:

            - Each **SHAP value** represents the **number of days** the feature contributes to increasing or decreasing the predicted treatment duration.
            - For example:
                - A SHAP value of **+40** means the feature **increases** the predicted duration by 40 days.
                - A SHAP value of **-20** means the feature **reduces** the predicted duration by 20 days.
            - Features with larger SHAP values (positive or negative) have a greater influence on the prediction.

            This interpretation helps identify which factors are adding or subtracting from the treatment duration and by how much.
            """
        )
        st.table(shap_df.assign(index='').set_index('index'))

