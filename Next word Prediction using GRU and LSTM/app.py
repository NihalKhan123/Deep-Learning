# Install Streamlit if running in Colab or Jupyter
# !pip install streamlit tensorflow keras numpy pickle-mixin

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------------------
# 🎯 App Configuration
# -------------------------------------------------------------
st.set_page_config(page_title="Next Word Predictor", page_icon="🔮", layout="centered")

st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.5em;
        font-size: 1em;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .prediction-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.1);
        text-align: center;
        font-size: 18px;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 📥 Load Models and Tokenizer
# -------------------------------------------------------------
@st.cache_resource
def load_resources():
    lstm_model = load_model('next_word_prediction_model.h5')
    gru_model = load_model('next_word_model_with_GRU.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return lstm_model, gru_model, tokenizer

try:
    lstm_model, gru_model, tokenizer = load_resources()
except Exception as e:
    st.error("⚠️ Error loading model or tokenizer. Make sure files are in the same folder.")
    st.stop()

# -------------------------------------------------------------
# 🔮 Prediction & Metrics Functions
# -------------------------------------------------------------
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

def calculate_perplexity(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    perplexity = np.exp(-np.mean(np.log(predicted + 1e-10)))  # add epsilon to prevent log(0)
    return perplexity

def calculate_loss(model, tokenizer, sequences, max_sequence_len):
    losses = []
    for sequence in sequences:
        text, next_word = sequence
        token_list = tokenizer.texts_to_sequences([text])[0]
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len - 1):]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        next_word_index = tokenizer.word_index.get(next_word, 0)
        if next_word_index < len(predicted[0]):
            losses.append(-np.log(predicted[0][next_word_index] + 1e-10))
    return np.mean(losses)

# Test sequences
test_sequences = [
    ("to be or not to", "be"),
    ("the quick brown fox", "jumps"),
    ("machine learning is", "fun"),
    ("artificial intelligence can", "predict"),
    ("deep learning helps", "vision"),
]

# -------------------------------------------------------------
# 🧠 Streamlit App UI
# -------------------------------------------------------------
st.title("🔮 Next Word Prediction App (LSTM + GRU)")
st.markdown("This app predicts the **next word** in a sentence using two deep learning models — LSTM and GRU.")

input_text = st.text_input("📝 Enter your text:", "to be or not to be")

if st.button("✨ Predict Next Word"):
    with st.spinner("Predicting... please wait ⏳"):
        max_sequence_len = lstm_model.input_shape[1] + 1

        # Predictions
        lstm_next = predict_next_word(lstm_model, tokenizer, input_text, max_sequence_len)
        gru_next = predict_next_word(gru_model, tokenizer, input_text, max_sequence_len)

        # Metrics
        lstm_perp = calculate_perplexity(lstm_model, tokenizer, input_text, max_sequence_len)
        gru_perp = calculate_perplexity(gru_model, tokenizer, input_text, max_sequence_len)
        lstm_loss = calculate_loss(lstm_model, tokenizer, test_sequences, max_sequence_len)
        gru_loss = calculate_loss(gru_model, tokenizer, test_sequences, max_sequence_len)

    # Results display
    st.markdown("### 🧩 Predictions:")
    st.markdown(f"""
        <div class='prediction-box'>
            <b>LSTM Prediction:</b> {lstm_next}<br>
            <b>GRU Prediction:</b> {gru_next}
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Model Metrics:")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("LSTM Perplexity", f"{lstm_perp:.4f}")
        st.metric("LSTM Loss", f"{lstm_loss:.4f}")
    with col2:
        st.metric("GRU Perplexity", f"{gru_perp:.4f}")
        st.metric("GRU Loss", f"{gru_loss:.4f}")

st.markdown("---")
st.caption("Developed by Nihal Khan • Powered by LSTM & GRU ⚡")
