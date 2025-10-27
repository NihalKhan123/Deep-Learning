
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model
lstm_model = load_model('next_word_prediction_model.h5')

# Load the GRU model
gru_model = load_model('next_word_model_with_GRU.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]  # Ensure the sequence length matches the max sequence length
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
    perplexity = np.exp(-np.mean(np.log(predicted)))
    return perplexity

def calculate_loss(model, tokenizer, sequences, max_sequence_len):
    losses = []
    for sequence in sequences:
        text, next_word = sequence[0], sequence[1]
        token_list = tokenizer.texts_to_sequences([text])[0]
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len - 1):]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        next_word_index = tokenizer.word_index.get(next_word, 0)
        if next_word_index < len(predicted[0]):
            losses.append(-np.log(predicted[0][next_word_index]))
        else:
            losses.append(np.inf)
    avg_loss = np.mean(losses)
    return avg_loss

st.title("Next Word Prediction with LSTM and GRU")

input_text = st.text_input("Enter the sequence of words", "to be or not to be")

if st.button("Predict Next Word"):
    max_sequence_len = lstm_model.input_shape[1] + 1

    # Predict the next word using LSTM model
    lstm_next_word = predict_next_word(lstm_model, tokenizer, input_text, max_sequence_len)

    # Predict the next word using GRU model
    gru_next_word = predict_next_word(gru_model, tokenizer, input_text, max_sequence_len)

    # Display the results
    st.write(f"Next word predicted by LSTM: {lstm_next_word}")
    st.write(f"Next word predicted by GRU: {gru_next_word}")

    # Calculate Perplexity
    lstm_perplexity = calculate_perplexity(lstm_model, tokenizer, input_text, max_sequence_len)
    gru_perplexity = calculate_perplexity(gru_model, tokenizer, input_text, max_sequence_len)
    st.write(f"LSTM Model Perplexity: {lstm_perplexity:.4f}")
    st.write(f"GRU Model Perplexity: {gru_perplexity:.4f}")

    # Calculate Loss over the test set
    lstm_loss = calculate_loss(lstm_model, tokenizer, test_sequences, max_sequence_len)
    gru_loss = calculate_loss(gru_model, tokenizer, test_sequences, max_sequence_len)
    st.write(f"LSTM Model Loss: {lstm_loss:.4f}")
    st.write(f"GRU Model Loss: {gru_loss:.4f}")
