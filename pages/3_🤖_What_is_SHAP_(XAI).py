import streamlit as st

# Title and Intro
st.title("🔍 SHAP (Explainable AI) for Medical Predictions")

st.markdown("""
In healthcare, understanding **why** a prediction is made is just as important as the prediction itself. 
SHAP (SHapley Additive exPlanations) enhances **transparency** and **trust** by showing how each feature (e.g., age, test results) influences the outcome. 
This helps medical professionals make **informed decisions** and **validate AI predictions**.

---
""")

# Section: What is SHAP
st.image("images/shap_logo.png", width=450)

st.header("📖 What is SHAP?")
st.markdown("""
SHAP explains AI model predictions by assigning **contribution values (SHAP values)** to each feature.

### Example:
- **Positive SHAP Value**: Increases the prediction (e.g., +10 days to treatment duration)
- **Negative SHAP Value**: Decreases the prediction (e.g., -5 days to treatment duration)

---
""")

# Section: Medical Context Example
st.header("🩺 Medical Example of SHAP")
st.markdown("""
**Prediction Task**: Predict treatment duration for a patient.

- **Base Value (average prediction)**: 120 days
- **Feature Contributions**:
    - **Age**: +20 days
    - **Liver Disease**: +15 days
    - **Number of Comorbidities**: +5 days
    - **Early Diagnosis**: -10 days

**Final Prediction**:
120 (base) + 20 (Age) + 15 (Liver Disease) + 5 (Comorbidities) - 10 (Early Diagnosis) = **150 days**

---
""")

# Section: How SHAP Works
st.header("🔬 How Does SHAP Work?")
st.markdown("""
SHAP values are calculated by:

1. **Analyzing Feature Combinations**:
    - Consider predictions with and without each feature.
    - Example: Compare the model's prediction with "Liver Disease" included vs. excluded.

2. **Measuring Feature Impact**:
    - Calculate the difference in predictions when a feature is added or removed.
    - Example: Adding "Early Diagnosis" reduces treatment duration by 10 days.

3. **Averaging Impact Across All Combinations**:
    - Final SHAP values show the **average contribution** of each feature.

**Takeaway**: SHAP provides a clear and reliable measure of how much each feature contributes to the model's prediction.

---
""")

# Section: Visual Explanation (Waterfall Plot)
st.header("📊 Visualizing SHAP with a Waterfall Plot")
st.markdown("""
The **Waterfall Plot** below shows how SHAP values adjust the prediction step-by-step:

1. **Base Value**: Starting point of the prediction (average for all patients).
2. **Feature Contributions**: Red (positive) values increase the prediction; blue (negative) values decrease it.
3. **Final Prediction**: Sum of all contributions and the base value.
""")

# Display SHAP Plot
st.image("images/shap.png", caption="SHAP Waterfall Plot Example")

st.markdown("""
### **How to Read This Plot**:
- **Base Value**: The average prediction, E[f(X)] from the model (e.g., 1.562 in this example).
- **Feature Bars**:
    - Each bar represents a feature's contribution to the prediction.
    - **Red Bars**: Push the prediction **higher**.
    - **Blue Bars**: Push the prediction **lower**.
- **Final Prediction**: The total after adding/subtracting all feature contributions with base value (e.g., 3.513 in this example).

### **Insights for Medical Staff**:
- **Key Drivers**: Identify the top features influencing the prediction (e.g., `Treatment_status` in this case).
- **Actionable Factors**: Features like `M_akurit` or `Maintenance_Phase_Regime` may indicate areas for intervention or further investigation.

---
""")

# Section: Key Benefits for Healthcare
st.header("💡 Why is SHAP Important for Healthcare?")
st.markdown("""
SHAP empowers medical professionals by providing:

1. **Transparency**:
    - Understand how patient-specific factors drive predictions.
    - Validate AI outputs with clear, interpretable insights.

2. **Trust**:
    - Build confidence in AI-based tools by explaining the "why" behind predictions.

3. **Actionable Insights**:
    - Prioritize factors that significantly impact outcomes (e.g., early diagnosis reduces treatment duration).
    - Adjust treatment plans based on feature importance.

4. **Personalized Care**:
    - Tailor decisions to individual patients by understanding their unique drivers of risk or recovery.
""")

# Section: Summary
st.info("""
💡 **SHAP in Action**: 
SHAP transforms black-box AI predictions into actionable, interpretable insights, ensuring AI becomes a trusted partner in improving patient care.
""")
