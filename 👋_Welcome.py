import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

# Set the title of the app
st.title("TB Treatment Duration and Outcome Prediction with Explainable AI 🩺")

# Introduction Section
st.markdown("""
### Welcome! 👋
This platform is designed to predict **TB Treatment Duration** and **Treatment Outcome** using advanced machine learning models with Explainable AI (XAI) techniques. You can select the prediction task of your choice from the **sidebar**.

---

### What is Tuberculosis (TB)? 🤔
Tuberculosis (TB) is a potentially serious infectious disease caused by the bacteria *Mycobacterium tuberculosis*. It primarily affects the lungs but can impact other parts of the body. TB is a global health challenge, and its management depends heavily on timely diagnosis, accurate treatment regimens, and consistent monitoring.

---

### What is Explainable AI (XAI)? 💡
Explainable AI (XAI) refers to methods and tools that make machine learning models transparent and interpretable. By understanding how a model makes its predictions, healthcare professionals can:

- **Trust** the predictions.
- Identify key factors driving treatment outcomes.
- Ensure ethical and unbiased decision-making.

In the context of TB treatment, XAI can help doctors and researchers understand the rationale behind predicted outcomes, leading to better and more informed healthcare decisions.

---

### Why is this Important? 🌟
Predicting TB treatment duration and outcomes accurately can significantly improve patient care by:

1. **Optimizing treatment plans** tailored to individual patients.
2. **Reducing unnecessary delays** in treatment adjustments.
3. Providing insights into key factors affecting treatment success or challenges.

Combining predictive power with explainability ensures that these insights are actionable and reliable.

---

### Dataset Source 📊
The predictions are powered by data from the **University of Malaya Medical Centre (UMMC)**, Kuala Lumpur, Malaysia. The dataset contains **435 records and 104 features** collected between **1st January 2018 and 30th September 2019**. It includes diverse data points, such as:

- **Sociodemographic details** (e.g., age, gender, nationality).
- **Medical histories** (e.g., HIV status, diabetes).
- **Clinical parameters** (e.g., weight, treatment regimens).
- **Behavioral factors** (e.g., smoking status, healthcare worker exposure).

This rich dataset enables robust model training and insightful predictions.

---

Feel free to explore the tool and uncover valuable insights into TB treatment prediction and explainability! 🎯

""")