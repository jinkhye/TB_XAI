import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
    layout="wide",
)

# Set the title of the app
st.title("TB Treatment Duration and Outcome Prediction with Explainable AI 🩺")

# Introduction Section
st.markdown("""
## Welcome 👋

This platform empowers users with **predictive insights** for **TB Treatment Duration** and **Treatment Outcomes** using cutting-edge **Machine Learning** models, integrated with **Explainable AI (XAI)** for transparency and reliability.

---

### How to Use the Platform:
""")

st.image("images/sidebar.png")

st.markdown(""" 1. Navigate through the **sidebar** to explore the features:
   - **TB Treatment Duration Prediction**: Estimate the duration of TB treatment with precision.
   - **TB Treatment Outcome Prediction**: Predict the treatment success or challenges ahead.
   - **What is SHAP (XAI)**: Learn how to interpret SHAP visualizations.

2. Select the prediction task of your choice to begin!
""")         

st.markdown("""
---
### What is Tuberculosis (TB)? 🤔
Tuberculosis (TB) is a **serious airborne infectious disease** caused by the bacterium *Mycobacterium tuberculosis*. 
While it primarily affects the **lungs**, it can also impact other organs like the **kidneys**, **brain**, and **spine**. 

### Global Impact of TB:
- In 2022, **10.6 million** people were infected globally, leading to **1.3 million deaths**.
- In 2023, **Malaysia** alone recorded **26,781 cases**, a **5.47% increase** from the previous year.

TB is both preventable and treatable, but challenges such as **drug resistance** and **treatment non-compliance** persist.
""")

# Placeholder for TB Illustration Image
st.image("images/tb.png", caption="Illustration of Mycobacterium tuberculosis")

# How TB Spreads Section
st.markdown("""
---
## How Does TB Spread? 🌬️

TB spreads through **airborne droplets** expelled when an infected person **coughs**, **sneezes**, or **talks**. 
Close and prolonged exposure to an infected individual increases the risk of transmission.

### Key Risk Factors:
- Living in crowded or poorly ventilated spaces.
- Co-existing conditions like **HIV** or **diabetes**.
- Lack of access to timely healthcare.
""")

# Placeholder for Transmission Diagram
st.image("images/spread.png", caption="TB Method of Transmission")

# Challenges in TB Treatment Section
st.markdown("""
---
## Challenges in TB Treatment 🏥

The standard treatment for TB includes a **6-month regimen**:
1. **2-month intensive phase** with multiple antibiotics.
2. **4-month continuation phase** for eradicating remaining bacteria.

However, **not all patients respond the same** to this approach, leading to:
- **Drug resistance** requiring complex and toxic regimens.
- **Treatment failures** due to non-compliance or comorbidities.
            
#### Need for Personalized Care:
By leveraging **AI models**, healthcare providers can tailor treatment plans to individual patient needs, improving outcomes.
""")

# Placeholder for Treatment Challenges Image
st.image("images/treatment.png", width=600)

# Importance of Explainable AI Section
st.markdown("""
---
## Importance of Explainable AI in TB Treatment 🔍

### What is Explainable AI (XAI)?
XAI enhances the transparency of **machine learning models** by explaining how **specific factors influence predictions**. This ensures:
- Trust among clinicians and patients.
- Informed decisions for better treatment strategies.

For example, **SHAP (SHapley Additive exPlanations)** can help identify which patient features (e.g., age, comorbidities) impact their treatment duration or outcome predictions.

### Learn More About SHAP:
To delve deeper into **SHAP interpretations and calculations**, navigate to the **sidebar** and select **"What is SHAP (XAI)"**. This section provides an overview of SHAP’s functionality and its role in making machine learning models explainable.
""")

# Placeholder for SHAP sidebar
st.image("images/sidebar2.png")


# Conclusion Section
st.markdown("""
---
## Let's Fight TB Together! 💪

By combining **advanced technologies** with **human expertise**, we can overcome the challenges of TB treatment and save lives. Explore the platform to:
- Predict **treatment duration** tailored to patient needs.
- Forecast **treatment outcomes** for better planning.
- Learn more about how XAI is revolutionizing healthcare.
""")

# Placeholder for Call to Action Image
st.image("images/healthcare.jpg", width=600)