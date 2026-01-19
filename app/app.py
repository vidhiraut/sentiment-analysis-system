import os
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def load_data(base_path, limit=3000):
    texts, labels = [], []
    for label, folder in [(1, "pos"), (0, "neg")]:
        folder_path = os.path.join(base_path, folder)
        for filename in os.listdir(folder_path)[:limit]:
            with open(os.path.join(folder_path, filename), encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(label)
    return texts, labels

@st.cache_resource
def train_model():
    X, y = load_data("data/aclImdb/train")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("clf", MultinomialNB())
    ])
    model.fit(X, y)
    return model

st.set_page_config(page_title="Sentiment Analysis System")
st.title("Sentiment Analysis System")

with st.spinner("Training model (first run only)..."):
    model = train_model()

text = st.text_area("Enter text")

if st.button("Analyze"):
    if text.strip():
        pred = model.predict([text])[0]
        prob = model.predict_proba([text])[0].max()
        st.success("Positive 😊" if pred == 1 else "Negative 😞")
        st.metric("Confidence", f"{prob:.2%}")
