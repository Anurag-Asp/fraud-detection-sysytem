# --------------- #
#  Core Libraries #
# --------------- #
import pandas as pd
import joblib
import re
import streamlit as st
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os
from groq import Groq
# ------------------ #
#  Constants & Setup #
# ------------------ #
MODEL_FILES = {
    'vectorizer': 'trained data/tfidf_vectorizer.joblib',
    'model': 'trained data/spam_model.joblib',
    'encoder': 'trained data/label_encoder.joblib'
}


load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'),)

# ------------------- #
#  Utility Functions  #
# ------------------- #
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

# ------------------- #
#  Model Management   #
# ------------------- #
def train_and_save_model(sms_path='data/sms_spam.csv', email_path='data/email.csv'):
    # Load SMS dataset (columns: v1=label, v2=text)
    sms_df = pd.read_csv(sms_path, encoding='latin-1')
    sms_df = sms_df.rename(columns={'v1': 'label', 'v2': 'text'})
    
    # Load Email dataset (columns: Category, Message)
    email_df = pd.read_csv(email_path)
    email_df = email_df.rename(columns={'Category': 'label', 'Message': 'text'})

    # Combine datasets
    combined_df = pd.concat([
        sms_df[['text', 'label']],
        email_df[['text', 'label']]
    ], axis=0)

    # Preprocess
    combined_df['clean_text'] = combined_df['text'].apply(clean_text)
    encoder = LabelEncoder()
    combined_df['label'] = encoder.fit_transform(combined_df['label'])

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        combined_df['clean_text'], 
        combined_df['label'], 
        test_size=0.2
    )

    # Vectorize
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)

    # Train model
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Save artifacts
    joblib.dump(model, MODEL_FILES['model'])
    joblib.dump(tfidf, MODEL_FILES['vectorizer'])
    joblib.dump(encoder, MODEL_FILES['encoder'])

    return model, tfidf, encoder

def load_models():
    try:
        return (
            joblib.load(MODEL_FILES['model']),
            joblib.load(MODEL_FILES['vectorizer']),
            joblib.load(MODEL_FILES['encoder'])
        )
    except FileNotFoundError:
        return train_and_save_model()

# -------------------- #
#  Gen AI Integration  #
# -------------------- #
def explain_with_genai(text):
    prompt = f"""
    Explain why this SMS might be fraudulent: "{text}".
    Highlight keywords and patterns. Use simple language.
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

# -------------------- #
#  Core Functionality  #
# -------------------- #
import speech_recognition as sr

model, vectorizer, encoder = load_models()
def transcribe_audio(audio_path):
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
        return r.recognize_google(audio)

def predict(input_text):
    cleaned_text = clean_text(input_text)
    vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized)[0]
    return prediction, explain_with_genai(input_text) if prediction == 1 else "Legitimate message"

# -------------------- #
#  Streamlit Interface #
# -------------------- #
def main():
    st.title("🛡️ CyberShield AI - Fraud Detection System")

    input_type = st.radio("Input Type:", ["Text/SMS", "Email", "Voice Recording"])

    if input_type == "Text/SMS":
        user_input = st.text_area("Enter message:")
    elif input_type == "Email":
        email_subject = st.text_input("Email Subject:")
        email_body = st.text_area("Email Body:")
        user_input = f"{email_subject} {email_body}"
    else:
        audio_file = st.file_uploader("Upload Recording", type=["wav", "mp3"])
        if audio_file:
            with open("temp_audio.mp3", "wb") as f:
                f.write(audio_file.read())
            user_input = transcribe_audio("temp_audio.mp3")
            st.write("Transcribed Text:", user_input)

    if st.button("Analyze") and 'user_input' in locals():
        with st.spinner("Analyzing..."):
            prediction, explanation = predict(user_input)
            
            st.subheader("Results")
            if prediction == 1:
                st.error("⚠️ Potential Fraud Detected!")
            else:
                st.success("✅ Legitimate Content")
            
            st.subheader("Analysis")
            st.markdown(explanation)

if __name__ == "__main__":
    main()