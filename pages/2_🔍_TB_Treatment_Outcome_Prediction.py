import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Treatment Outcome Prediction with XAI",
    page_icon="🔍",
)

# Title of the page
st.title("TB Treatment Outcome Prediction with Explainable AI 🔍💊")

# Introduction Section with Collapsible Panels
with st.expander("About this Page"):
    st.markdown("""
    Welcome to the **Treatment Outcome Prediction** page! Here, we use advanced machine learning models to predict the outcome of TB treatment for patients, ensuring personalized and effective care management.
    """)

with st.expander("Model Used 🌟"):
    st.markdown("""
    The prediction is powered by an **XGBoost Classifier**, a highly efficient and flexible machine learning algorithm known for its exceptional performance on structured datasets and its ability to handle both classification and regression tasks.
    """)

with st.expander("Model Performance Metrics 📈"):
    st.markdown("""
    Our machine learning model has been evaluated using the following metrics:

    - **Accuracy:** 1.0
    - **Precision:** 1.0
    - **Recall:** 1.0
    - **F1-Score:** 1.0
    - **AUC (Area Under the Curve):** 1.0

    These metrics indicate that the model provides perfect predictions for the dataset, showcasing its reliability and effectiveness in predicting treatment outcomes.
    """)

with st.expander("How It Works 🔍"):
    st.markdown("""
    - **Input Data:** Enter patient-specific information such as demographics, medical history, and clinical parameters.
    - **Model Prediction:** The model processes the input data and predicts the expected treatment outcome (e.g., Successful or Unsuccessful).
    - **Explainable AI (XAI):** Understand the factors driving the prediction with interactive explanations to ensure transparency and trust.
    """)

with st.expander("Why is this Important? 🌟"):
    st.markdown("""
    - **Improved Decision-Making:** Provides actionable insights for clinicians to adjust treatment plans as needed.
    - **Resource Optimization:** Helps healthcare providers focus resources on high-risk patients.
    - **Better Outcomes:** Facilitates early identification of potential challenges, improving treatment success rates.

    Use the sidebar to input patient details and view predictions. Let's enhance TB treatment outcomes together! 💡
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

# Retain only relevant features
selected_features = ['Weightkg', 'Treatment_duration', 'Maintenance_Phase_Regime', 'M_akurit', 'Treatment_status']
X = data[selected_features]
y = data['Binary_Treatment_Outcome']

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train XGBoost Model
model = XGBClassifier(
    colsample_bytree=0.6559, gamma=0.26, learning_rate=0.1193,
    max_depth=8, min_child_weight=2, n_estimators=829,
    reg_alpha=0.2249, reg_lambda=0.3952, subsample=0.9633,
    random_state=42
)
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

# Interactive Prediction in the Sidebar
st.sidebar.write("### Input Patient Data")
user_input = {}

# Add numeric input fields for features
numeric_features = ['Weightkg', 'Treatment_duration']
for feature in numeric_features:
    user_input[feature] = st.sidebar.number_input(
        f"Enter value for {feature}", value=float(X[feature].mean()), step=1.0
    )

# Add dropdowns for categorical features
for feature, mapping in mappings.items():
    user_input[feature] = st.sidebar.selectbox(
        f"Select value for {feature}", options=list(mapping.values())
    )

# Convert dropdown choices back to numeric for prediction
for feature, mapping in mappings.items():
    reverse_mapping = {v: k for k, v in mapping.items()}
    user_input[feature] = reverse_mapping[user_input[feature]]

# Convert user input into DataFrame and align columns with selected features
user_input_df = pd.DataFrame([user_input])
user_input_df = user_input_df[selected_features]  # Align with selected features

# Prediction and Explanation
if st.sidebar.button("Predict and Explain"):
    # Predict the treatment outcome for user input
    prediction = model.predict(user_input_df)[0]
    prediction_label = "Successful" if prediction == 1 else "Unsuccessful"
    st.markdown(f"""
        <div style="text-align: center; padding: 20px; border: 2px solid #4CAF50; border-radius: 10px; background-color: #F0FFF0;">
            <h2 style="color: #4CAF50;">🎯 Predicted Treatment Outcome</h2>
            <h1 style="color: #000;">{prediction_label}</h1>
        </div>
    """, unsafe_allow_html=True)

    # SHAP Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(user_input_df)

    # SHAP Waterfall Plot
    st.write("### SHAP Explanation for Prediction (Waterfall Plot)")
    shap.waterfall_plot(shap_values[0])
    st.pyplot(plt.gcf())  # Display the SHAP Waterfall Plot

    # SHAP Value Table
    st.write("### SHAP Values for Input Features")
    shap_df = pd.DataFrame({
        "Feature": user_input_df.columns,
        "SHAP Value": shap_values.values[0],
    }).sort_values(by="SHAP Value", key=abs, ascending=False)

    st.table(shap_df)

