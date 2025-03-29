import string
import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import streamlit as st
import joblib
from catboost import CatBoostClassifier

def casefolding(text):
    return text.lower()

def hapus_angka(teks):
    if isinstance(teks, str):
        return ''.join([char for char in teks if not char.isdigit()])
    return teks

def remove_punctuation(teks):
    if isinstance(teks, str):
        punctuation_set = set(string.punctuation)
        return ''.join(char for char in teks if char not in punctuation_set)
    return teks

def remove_whitespace(text):
    return ' '.join(text.split())

def tokenize(text):
    return re.findall(r'\b\w+\b', text)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(text):
    return ' '.join(stemmer.stem(word) for word in tokenize(text))

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stopword_factory = StopWordRemoverFactory()
stopwords_sastrawi = set(stopword_factory.get_stop_words())
stopwords_nltk = set(stopwords.words('indonesian'))
stopwords_combined = stopwords_sastrawi.union(stopwords_nltk)

def remove_stopwords(text):
    return ' '.join(word for word in tokenize(text) if word not in stopwords_combined)

def preprocess_text_input(text):
    text = casefolding(text)
    text = remove_whitespace(text)
    text = hapus_angka(text)
    text = remove_punctuation(text)
    text = stemming(text)
    text = remove_stopwords(text)
    return text

model = CatBoostClassifier()
model.load_model("Running Model/catboost_Pemodelan Catboost dengan TF-IDF.cbm")
vectorizer = joblib.load("Running Model/tfidf_vectorizer_Pemodelan Catboost dengan TF-IDF.pkl")

label_mapping = {0: ('Negatif', 'red'), 1: ('Netral', 'yellow'), 2: ('Positif', 'green')}

st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to bottom, steelblue 50%, lightgrey 50%);
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Dashboard Sentimen Analisis")
st.write("Masukkan teks untuk memprediksi sentimen menggunakan model CatBoost.")

user_input = st.text_area("Masukkan teks : ")

col1, col2 = st.columns([1, 1])
with col1:
    predict_button = st.button("Prediksi")
with col2:
    clear_button = st.button("Bersihkan Input")

if clear_button:
    user_input = ""
    st.rerun()

if predict_button:
    if user_input:
        processed_text = preprocess_text_input(user_input)
        text_tfidf = vectorizer.transform([processed_text])
        prediction = model.predict(text_tfidf)
        predicted_label, color = label_mapping[int(prediction[0])]
        st.markdown(f'<div style="padding: 10px; background-color: {color}; color: white; border-radius: 5px;">Hasil prediksi : <b>{predicted_label}</b></div>', unsafe_allow_html=True)
    else:
        st.warning("Harap masukkan teks sebelum melakukan prediksi.")
