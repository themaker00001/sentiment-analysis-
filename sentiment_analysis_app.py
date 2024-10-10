import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    return scores

# Streamlit app layout
st.title("Sentiment Analysis with VADER")
st.write("Enter text to analyze its sentiment:")

# Text input from user
user_input = st.text_area("Text Input", height=200)

if st.button("Analyze"):
    if user_input:
        sentiment_scores = analyze_sentiment(user_input)
        st.write("Sentiment Scores:")
        st.json(sentiment_scores)
    else:
        st.warning("Please enter some text to analyze.")
