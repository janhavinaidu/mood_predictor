# Mood Predictor App

This is a web application that predicts the emotional mood from journal entries.
![image](https://github.com/user-attachments/assets/a57fb2d5-792f-4144-a17b-b5131fc043f9)

## Frontend

The frontend is built using **Streamlit**, a Python framework for creating interactive web apps. 

Streamlit handles all the UI components — text input, buttons, and displaying results — making the app user-friendly and interactive.

## Backend / ML

The backend leverages state-of-the-art natural language processing (NLP) techniques to analyze and predict the emotional mood from journal entries.

Model Used:
The app uses a pre-trained Hugging Face Transformer model called j-hartmann/emotion-english-distilroberta-base, which is based on the DistilRoBERTa architecture. This model has been fine-tuned on a large dataset to recognize multiple emotions such as joy, sadness, anger, fear, surprise, and more.

How It Works:

Text Preprocessing: The journal entry input is first cleaned and preprocessed by converting to lowercase and removing punctuation to ensure consistent input formatting.

Sentence Splitting: The input text is split into individual sentences to allow fine-grained emotion detection at the sentence level.

Emotion Scoring: Each sentence is passed through the Transformer model, which outputs probabilities for each emotion class.

Aggregation: The emotion scores from all sentences are averaged to produce an overall emotion distribution for the entire journal entry.

Prediction: The emotion with the highest average score above a confidence threshold is selected as the predicted mood. If no emotion exceeds the threshold, the mood is labeled as "neutral/uncertain".

Technologies Used:

Hugging Face transformers library for easy integration of pre-trained NLP models.

PyTorch as the underlying deep learning framework powering the model inference.

pipeline API from Hugging Face for streamlined text classification tasks.

How to Run This App Locally
Follow these steps to run the Mood Predictor app on your own computer:
1. Clone the Repository
  Open your terminal and run:

  git clone https://github.com/YOUR_USERNAME/mood_predictor.git

  cd mood_predictor

2. Create a Virtual Environment (Recommended)

  This keeps dependencies isolated:

  python -m venv venv

  Activate it:

  On Windows (PowerShell):

  .\venv\Scripts\Activate.ps1

3. Install Dependencies

  Install all required Python packages from requirements.txt:

  pip install -r requirements.txt

4. Run the Streamlit App

  Start the app locally:

  streamlit run app.py
