import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import streamlit as st
import pickle

nltk.download('punkt')
ps = PorterStemmer()

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))
st.title("EmailSpamClassifier")
input_sms = st.text_input("Enter the message : ")


def text_preprocessing(x):
    text = nltk.word_tokenize(x.lower())
    y = []
    for i in text:
        if i.isalnum() and i not in stopwords.words('english'):
            y.append(ps.stem(i))
    return " ".join(y)


if st.button("Predict"):
    t_text = text_preprocessing(input_sms)
    vInput = tfidf.transform([t_text])
    result = model.predict(vInput)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
