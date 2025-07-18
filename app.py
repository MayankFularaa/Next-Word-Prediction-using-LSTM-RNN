import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Load tokenizer and model assets
# -------------------------------
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("next_word_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.txt", "r") as f:
        max_len = int(f.read())
    return model, tokenizer, max_len

model, tokenizer, max_len = load_model_and_tokenizer()

# -------------------------------
# Predict next word
# -------------------------------
def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len - 1, padding='pre')
    pred = model.predict(padded, verbose=0)
    predicted_index = np.argmax(pred)
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return ""

# -------------------------------
# UI with suggestion
# -------------------------------
st.set_page_config(page_title="Next Word Prediction", layout="centered")
st.title("📝 Next Word Prediction App")

st.markdown("""
<style>
input {
    background-color: white !important;
    color: black !important;
    font-size: 18px !important;
}
.suggestion-box {
    color: brown;
    font-weight: bold;
    opacity: 0.6;
    font-size: 18px;
    margin-top: -18px;
    margin-left: 3px;
}
</style>
""", unsafe_allow_html=True)

user_input = st.text_input("Type something:", key="text_input")

if user_input.strip() != "":
    suggestion = predict_next_word(user_input)
    
    if suggestion:
        input_words = user_input.strip().split()
        last_word = input_words[-1]
        st.markdown(f"<div class='suggestion-box'>{user_input} <span>{suggestion}</span></div>", unsafe_allow_html=True)
        st.caption("Press `Tab` to accept suggestion (manually type it).")
    else:
        st.markdown("No suggestion found.")

