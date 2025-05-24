import streamlit as st
from transformers import pipeline
import re
from collections import defaultdict

# Load model once
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

emotion_model = load_emotion_model()

def split_into_sentences(text):
    return re.split(r'(?<=[.!?]) +', text.strip())

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_mood(text, confidence_threshold=0.2):
    sentences = split_into_sentences(text)
    total_scores = defaultdict(float)
    
    for sentence in sentences:
        results = emotion_model(sentence)[0]
        for res in results:
            total_scores[res['label']] += res['score']

    avg_scores = {label: score / len(sentences) for label, score in total_scores.items()}
    sorted_emotions = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
    top_emotion, top_score = sorted_emotions[0]

    if top_score < confidence_threshold:
        top_emotion = "neutral/uncertain"

    return top_emotion, avg_scores

# Streamlit UI
st.title("🧠 Mood Predictor from Journal Entry")

journal_text = st.text_area("Write your journal entry here:")

if st.button("Predict Mood"):
    if journal_text.strip():
        cleaned_text = preprocess_text(journal_text)
        predicted_mood, scores = predict_mood(cleaned_text)

        st.success(f"**Predicted Mood:** {predicted_mood.capitalize()}")
        st.subheader("Emotion Scores:")
        for emotion, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            st.write(f"- {emotion}: {score:.4f}")
    else:
        st.warning("Please enter some text.")
