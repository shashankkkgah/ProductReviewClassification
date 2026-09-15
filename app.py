import streamlit as st
import joblib
import re

# Page configuration
st.set_page_config(
    page_title="Product Review Classification",
    page_icon="🛍️",
    layout="centered"
)

# Load saved files
model = joblib.load("product_review_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Title
st.title("🛍️ Product Review Classification")

st.write(
    "Enter a product review and the machine learning model "
    "will classify its sentiment."
)

# Review input
review = st.text_area(
    "Enter your product review:",
    placeholder="Example: This product is waste"
)

# Predict button
if st.button("🔍 Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:
        # Clean review
        cleaned_review = clean_text(review)

        # Convert to TF-IDF
        review_vector = tfidf.transform([cleaned_review])

        # Prediction
        prediction = model.predict(review_vector)

        # Convert encoded label back to text
        sentiment = label_encoder.inverse_transform(prediction)[0]

        # Display result
        if sentiment == "positive":
            st.success("😊 Positive Review")

        elif sentiment == "neutral":
            st.warning("😐 Neutral Review")

        else:
            st.error("😞 Negative Review")

        st.write("**Predicted Sentiment:**", sentiment.capitalize())