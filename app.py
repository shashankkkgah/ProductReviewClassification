
import streamlit as st
import joblib

# Load saved model and preprocessing objects
model = joblib.load("product_review_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Page configuration
st.set_page_config(
    page_title="Product Review Classification",
    page_icon="🛍️"
)

st.title("🛍️ Product Review Classification")
st.write("Enter a product review to predict its sentiment.")

# User input
review = st.text_area(
    "Enter your product review:",
    placeholder="Example: This product is excellent and I love it!"
)

# Prediction button
if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:
        # Convert review to TF-IDF
        review_tfidf = tfidf.transform([review])

        # Make prediction
        prediction = model.predict(review_tfidf)

        # Convert prediction to label
        sentiment = label_encoder.inverse_transform(prediction)[0]

        # Display result
        if sentiment == "Positive":
            st.success("😊 Positive Review")

        else:
            st.error("😞 Negative Review")
