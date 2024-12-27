import streamlit as st

st.title("🤖 What is SHAP (XAI)")
    
st.markdown("""
SHAP (SHapley Additive exPlanations) values explain how each feature contributes to a model's prediction.
""")

# Simple Visual Example
st.header("📊 Simple Example")
st.markdown("""
Imagine predicting a house price ($300,000) using three features:

```
Base Value (average): $250,000

Feature Contributions:
Size:     +$30,000
Location: +$15,000
Age:      +$5,000
```

Final Prediction = \$250,000 + 30,000 + $15,000 + $5,000 = $300,000
""")

# How It's Calculated
st.header("🔄 How SHAP Values Are Calculated")
st.markdown("""
1. **Look at Feature Combinations**
    - Check how the model performs with different feature combinations
    - Example: Size only, Size + Location, Size + Age, etc.

2. **Measure Feature Impact**
    - Compare predictions with and without each feature
    - Average the differences across all combinations

3. **Final Value**
    - The average impact becomes the feature's SHAP value
    - Shows how much each feature moves the prediction up or down
""")

# Key Takeaway
st.info("""
💡 **In Simple Terms**: SHAP values show how much each feature pushes the prediction 
up or down from the average prediction.
""")