import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import shap

# Set up the app page
st.set_page_config(page_title="TB Treatment Prediction", page_icon="🫁")
st.title("🫁 TB Treatment Duration and Outcome Prediction with Explainable AI")
st.write(
    """
    This app predicts tuberculosis (TB) treatment duration and outcomes using advanced machine learning models. It incorporates explainable AI techniques to provide insights into the factors influencing predictions, enhancing transparency and decision-making for healthcare professionals.
    """
)

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

st.write("### Data After Outlier Removal")
st.write(f"Original Data: {data.shape[0]} rows")
st.write(f"Data After Cleaning: {data_cleaned.shape[0]} rows")
st.write(data_cleaned.describe())

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

st.write("### Model Evaluation")
st.write(f"**Test MAE:** {test_mae:.2f}")
st.write(f"**Test RMSE:** {test_rmse:.2f}")
st.write(f"**Test R²:** {test_r2:.2f}")

# SHAP Explanation
st.write("### Explainable AI with SHAP")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test, check_additivity=False)

st.write("#### SHAP Summary Plot")
fig_summary = plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig_summary)

st.write("#### SHAP Bar Plot (Feature Importance)")
fig_bar = plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
st.pyplot(fig_bar)

# Interactive Prediction
st.write("### Make Predictions")

# Define all numeric and categorical features dynamically based on X
numeric_features = ['Age', 'Year_of_birth', 'Weightkg', 'Number_of_comorbids', 'Number_immunosuppressant']
numeric_features = [feature for feature in numeric_features if feature in X.columns]  # Only include existing features
categorical_features = [col for col in X.columns if col not in numeric_features]

user_input = {}

# Add numeric input fields for features that exist in X
for feature in numeric_features:
    user_input[feature] = st.number_input(f"Enter value for {feature}", value=int(X[feature].mean()), step=1)

# Add categorical dropdowns for features that exist in X
for feature in categorical_features:
    unique_values = X[feature].unique()
    user_input[feature] = st.selectbox(f"Select value for {feature}", options=unique_values)

# Convert user input into DataFrame and align columns with X_train
user_input_df = pd.DataFrame([user_input])

# Align columns with the training data
user_input_df = user_input_df[X.columns]  # Ensure the same column order as X_train

if st.button("Predict"):
    prediction = model.predict(user_input_df)[0]
    st.write(f"### Predicted Treatment Duration: {prediction:.2f} days")