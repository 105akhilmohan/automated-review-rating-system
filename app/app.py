import streamlit as st
import joblib
from sklearn.base import BaseEstimator

# Load models and vectorizers 
@st.cache_resource
def load_models():
    try:
        # Model A: SVM (balanced dataset)
        model_a = joblib.load(r"C:\AKHILMOHAN\ZECSER INTERNSHIP\APP\svm_balanced_model.pkl")
        vectorizer_a = joblib.load(r"C:\AKHILMOHAN\ZECSER INTERNSHIP\APP\tfidf_vectorizer_balanced.pkl")

        # Model B: SVM (imbalanced dataset)
        model_b = joblib.load(r"C:\AKHILMOHAN\ZECSER INTERNSHIP\APP\svm_imbalanced_model.pkl")
        vectorizer_b = joblib.load(r"C:\AKHILMOHAN\ZECSER INTERNSHIP\APP\tfidf_vectorizer.pkl")

        # Validate models and vectorizers
        for model, name in [(model_a, "Model A"), (model_b, "Model B")]:
            if not isinstance(model, BaseEstimator) or not hasattr(model, "predict"):
                raise TypeError(f"{name} is not a valid trained model.")
        for vectorizer, name in [(vectorizer_a, "Vectorizer A"), (vectorizer_b, "Vectorizer B")]:
            if not hasattr(vectorizer, "transform"):
                raise TypeError(f"{name} is invalid.")

        return model_a, vectorizer_a, model_b, vectorizer_b

    except Exception as e:
        st.error(f"Error loading models or vectorizers: {e}")
        st.stop()

# Load models
model_a, vectorizer_a, model_b, vectorizer_b = load_models()

# Streamlit UI 
st.title("Automated Review Rating System")
st.write("Enter a review and see predicted ratings from two models:")

# Input text area
user_input = st.text_area("Enter your review:", height=150)

# Predict button
if st.button("Submit"):
    if not user_input.strip():
        st.warning("Please enter a review before submitting.")
    else:
        try:
            # Transform input for each model separately
            X_input_a = vectorizer_a.transform([user_input])
            X_input_b = vectorizer_b.transform([user_input])

            # Predictions 
            pred_a = model_a.predict(X_input_a)[0]  # Model A prediction
            pred_b = model_b.predict(X_input_b)[0]  # Model B prediction

            # Display results side by side
            st.success("Prediction Results:")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Model A")
                st.write(f"Predicted Rating: {pred_a}")

            with col2:
                st.subheader("Model B")
                st.write(f"Predicted Rating: {pred_b}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")