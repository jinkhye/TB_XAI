import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Treatment Outcome Prediction through XAI",
    page_icon="🔍",
    layout="wide",
)

# Title of the page
st.title("TB Treatment Outcome Prediction through Explainable AI 🔍💊")

st.markdown("""
Welcome to the **Treatment Outcome Prediction** page! Here, we use advanced machine learning models to predict the outcome of TB treatment for patients, ensuring personalized and effective care management.

---
            
### How It Works 🔍
- **Input Data:** Enter patient-specific information such as demographics, medical history, and clinical parameters.
- **Model Prediction:** The model processes the input data and predicts the expected treatment outcome (e.g., Successful or Unsuccessful).
- **Explainable AI (XAI):** Understand the factors driving the prediction with interactive explanations to ensure transparency and trust.

**Choose a prediction model at the sidebar to experiment with different approaches.**

---
""")


# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/TB Cleaned 2.csv")

data = load_data()

# Map Treatment Outcome to Binary Classes
data['Binary_Treatment_Outcome'] = data['Treatment_Outcome'].map({
    1: 1,  # Successful
    2: 0,  # Unsuccessful
    4: 0,  # Unsuccessful
    5: 0   # Unsuccessful
}).fillna(0)

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

# Retain only relevant features
selected_features = ['Weightkg', 'Treatment_duration', 'Maintenance_Phase_Regime', 'M_akurit', 'Treatment_status']
X = data_cleaned[selected_features]
y = data_cleaned['Binary_Treatment_Outcome']

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Constants for model parameters
LR_PARAMS = {'C': 7.951759613930136, 'max_iter': 1050, 'penalty': 'l1', 'solver': 'liblinear'}
RF_PARAMS = {'bootstrap': False, 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 445}
XG_PARAMS = {'colsample_bytree': 0.8405197135484546, 'gamma': 0.11875323564623497, 'learning_rate': 0.09004457857440497, 'max_depth': 6, 'min_child_weight': 1, 'n_estimators': 508, 'reg_alpha': 0.43429956409473014, 'reg_lambda': 0.22359583851945264, 'subsample': 0.9816112697203057}

# Model Selection
# st.sidebar.header("🧠 Select a Model")
# model_choice = st.sidebar.selectbox(
#     "Choose a model:",
#     ["Logistic Regression", "Random Forest", "XGBoost"],
#     index=0
# )

model_choice = "Logistic Regression"

# Display Model Details Dynamically
st.sidebar.header("Model Details and Insights")

if model_choice == "Logistic Regression2":
    with st.sidebar.expander("Model Used 🔧"):
        st.markdown(f"""
        The prediction is powered by a **Logistic Regression** model with the following parameters:
        
        - **C:** {LR_PARAMS['C']}
        - **Penalty:** {LR_PARAMS['penalty']}
        - **Solver:** {LR_PARAMS['solver']}
        - **Max Iterations:** {LR_PARAMS['max_iter']}
        """)

    with st.sidebar.expander("Model Performance Metrics 📈"):
        st.markdown("""
        Our machine learning model has been evaluated using the following metrics:

        - **Accuracy:** 0.9440
        - **Precision:** 0.9870
        - **Recall:** 0.9268
        - **F1-Score:** 0.9560
        - **AUC (Area Under the Curve):** 0.9807
        """)

    model = LogisticRegression(**LR_PARAMS, random_state=42)

if model_choice == "Random Forest":
    with st.sidebar.expander("Model Used 🔧"):
        st.markdown(f"""
        The prediction is powered by a **Random Forest** model with the following parameters:
        
        - **Bootstrap:** {RF_PARAMS['bootstrap']}
        - **Max Depth:** {RF_PARAMS['max_depth']}
        - **Max Features:** {RF_PARAMS['max_features']}
        - **Min Samples Leaf:** {RF_PARAMS['min_samples_leaf']}
        - **Min Samples Split:** {RF_PARAMS['min_samples_split']}
        - **Number of Estimators:** {RF_PARAMS['n_estimators']}
        """)

    with st.sidebar.expander("Model Performance Metrics 📈"):
        st.markdown("""
        Our machine learning model has been evaluated using the following metrics:

        - **Accuracy:** 0.9840
        - **Precision:** 0.9877
        - **Recall:** 1.0000
        - **F1-Score:** 0.9756
        - **AUC:** 1.0000
        """)

    model = RandomForestClassifier(**RF_PARAMS, random_state=42)

# XGBoost
if model_choice == "Logistic Regression":
    with st.sidebar.expander("Model Used 🔧"):
        st.markdown(f"""
        The prediction is powered by a **XGBoost** model with the following parameters:
        
        - **Learning Rate:** {XG_PARAMS['learning_rate']}
        - **Max Depth:** {XG_PARAMS['max_depth']}
        - **Min Child Weight:** {XG_PARAMS['min_child_weight']}
        - **Number of Estimators:** {XG_PARAMS['n_estimators']}
        - **Colsample Bytree:** {XG_PARAMS['colsample_bytree']}
        - **Subsample:** {XG_PARAMS['subsample']}
        - **Gamma:** {XG_PARAMS['gamma']}
        - **Reg Alpha:** {XG_PARAMS['reg_alpha']}
        - **Reg Lambda:** {XG_PARAMS['reg_lambda']}
        """)

    with st.sidebar.expander("Model Performance Metrics 📈"):
        st.markdown("""
        Our machine learning model has been evaluated using the following metrics:

        - **Accuracy:** 0.9920
        - **Precision:** 0.9939
        - **Recall:** 1.0000
        - **F1-Score:** 0.9878
        - **AUC:** 1.0000
        """)

    # model = XGBClassifier(**XG_PARAMS, random_state=42)
    model = LogisticRegression(**LR_PARAMS, random_state=42)

with st.sidebar.expander("Why is this Important? 🌟"):
    st.markdown("""
    - **Improved Decision-Making:** Provides actionable insights for clinicians to adjust treatment plans as needed.
    - **Resource Optimization:** Helps healthcare providers focus resources on high-risk patients.
    - **Better Outcomes:** Facilitates early identification of potential challenges, improving treatment success rates.

    Use the input section on the main page to enter patient details and view predictions. Let's optimize TB treatment outcomes together! 💡
    """)

model.fit(X_train, y_train)

# Add dropdowns for categorical features using mappings
mappings = {
    "Maintenance_Phase_Regime": {
        0: "None",
        1: "Yes",
    },
    "M_akurit": {
        0: "No",
        1: "Yes",
    },
    "Treatment_status": {
        0: "NA",
        1: "Favourable",
        2: "Unfavourable",
        3: "Ongoing treatment",
        4: "Transfer out",
    },
}

# Interactive Prediction
st.markdown("## Input Patient Data")

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
    user_input['Weightkg'] = st.number_input(
        "Weight (kg)",
        help="The weight of the patient in kilograms.",
        value=50.0
    )

    user_input['Treatment_duration'] = st.number_input(
        "Treatment Duration (days)",
        help="The duration of the treatment in days.",
        value=180
    )

    user_input['Treatment_status'] = reverse_mappings['Treatment_status'][
        st.selectbox(
            "Treatment Status",
            list(mappings['Treatment_status'].values()),
            help="The current status of the patient's treatment."
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

    # Convert user input into DataFrame and align columns with selected features
    user_input_df = pd.DataFrame([user_input])
    user_input_df = user_input_df[selected_features]  # Align with selected features

if user_input_df.empty:
    st.error("Aligned input data is empty. Please check the uploaded file or manual inputs.")

# Prediction and SHAP Explanation for Treatment Outcome
if st.button("Predict and Explain"):
    # Predict the treatment outcome for user input
    prediction = model.predict(user_input_df)[0]
    prediction_label = "Successful" if prediction == 1 else "Unsuccessful"
    st.markdown(f"""
        <div style="text-align: center; padding: 20px; border: 2px solid #4CAF50; border-radius: 10px; background-color: #F0FFF0;">
            <h2 style="color: #4CAF50;">🎯 Predicted Treatment Outcome</h2>
            <h1 style="color: #000;">{prediction_label}</h1>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""---""")

    explainer = shap.Explainer(model, X_train)

    # Generate SHAP values for the user input
    shap_values_input = explainer(user_input_df)

    # Determine base value and SHAP values for plotting
    if isinstance(model, LogisticRegression):
        base_value = explainer.expected_value
        shap_values_single_output = shap_values_input.values[0]
    elif isinstance(model, (XGBClassifier, RandomForestClassifier)):
        # Handle multi-output SHAP values (e.g., shape (5, 2))
        if len(shap_values_input.values.shape) == 3:  # Multi-class case
            base_value = explainer.expected_value[1]  # Positive class base value
            shap_values_single_output = shap_values_input.values[0, :, 1]  # First sample, positive class
        else:  # Binary classification or single output
            base_value = explainer.expected_value
            shap_values_single_output = shap_values_input.values[0, :]  # First sample
    else:
        base_value = explainer.expected_value[0]
        shap_values_single_output = shap_values_input.values[0, :]

    # Proceed with SHAP waterfall plot and DataFrame creation
    st.write("#### SHAP Explanation for Input Data (Waterfall Plot)")
    fig_waterfall = plt.figure()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_single_output,
            base_values=base_value,
            data=user_input_df.iloc[0].values,
            feature_names=user_input_df.columns
        ),
        max_display=10
    )
    st.pyplot(fig_waterfall)



    # Extract SHAP values and features for DataFrame creation
    shap_df = pd.DataFrame({
        "Feature": user_input_df.columns,
        "SHAP Value": shap_values_single_output
    }).sort_values(by="SHAP Value", key=abs, ascending=False)

    # Generate Text-Based Explanation
    explanation_text = "### What does this mean?\n"

    # Calculate the predicted value
    f_x = base_value + shap_values_single_output.sum()

    explanation_text += (
        f"The model predicts the treatment outcome as **{f_x:.2f} ({prediction_label})**.\n\n"
        "- **Above 0** → Predicted outcome is **successful**.\n"
        "- **Below 0** → Predicted outcome is **unsuccessful**.\n\n"
        "### Feature Contributions:\n"
    )

    top_features = shap_df.head(5)
    explanation_text += "\n\nHere are the top factors influencing this prediction:\n\n"

    for index, row in top_features.iterrows():
        feature = row["Feature"]
        shap_value = row["SHAP Value"]
        direction = "positive (towards success)" if shap_value > 0 else "negative (towards failure)"
        impact = abs(shap_value)
        explanation_text += f"- **{feature}**: {impact:.2f} units, with a {direction} contribution.\n"

    # Highlight the most influential feature
    top_feature = top_features.iloc[0]
    top_feature_name = top_feature["Feature"]
    top_feature_impact = abs(top_feature["SHAP Value"])
    explanation_text += (
        f"\nThe most significant factor is **{top_feature_name}**, "
        f"contributing **{top_feature_impact:.2f} units** to the predicted outcome.\n"
    )

    st.markdown(explanation_text)

    st.markdown("""
        ### Interpreting the Prediction

        - **SHAP values** represent how much each feature contributes to the prediction:
            - A **positive SHAP value** indicates that the feature contributes toward a **successful outcome**.
            - A **negative SHAP value** indicates that the feature contributes toward an **unsuccessful outcome**.

        - The value is measured in **relative units of impact** on the likelihood of success. For example:
            - A SHAP value of **+2.5** means the feature increases the likelihood of a successful outcome by 2.5 units.
            - A SHAP value of **-1.8** means the feature decreases the likelihood of success by 1.8 units.

        This approach provides a transparent view of how different patient factors influence the prediction.

        ---
    """)

    # Display SHAP values as a table with a clear header
    st.write("### Full SHAP Contributions for All Features")
    st.write(
        """
        Below is the full list of how each feature contributed to the predicted treatment outcome:
        """
    )
    st.table(shap_df)


