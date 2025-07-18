# Next-Word-Prediction-using-LSTM-RNN
This project is a Next Word Prediction web application built using TensorFlow, LSTM (RNN), and Streamlit. It allows users to input partial phrases and get predictions for the next most likely word based on a custom-trained model.

# Model File:

Because of Large Size of Model -> You can Find the Model File at https://drive.google.com/drive/folders/13ASX_i3pRg0sIJcNWBV2OwKQq9RKJc6s

# Features

Trained on custom text data using Keras Tokenizer and LSTM

Supports next word prediction based on given input

Real-time predictions via a Streamlit UI

Easy to run locally or deploy on platforms like Streamlit Cloud

Saves and loads tokenizer and model files for reuse

Supports dynamic input length using max_len

# Model Training

The model is trained on a custom dataset using the following pipeline:

Text preprocessing and tokenization

Creating n-gram sequences

Padding sequences to uniform length

Categorical encoding of labels

LSTM model with embedding + dense layers

Trained for 50 epochs using categorical crossentropy

Model and tokenizer are saved as:

- next_word_model.h5
- tokenizer.pkl
- max_len.txt

# How to Run Locally

1. Download the Repository:

Next-Word-Prediction-using-LSTM-RNN

2. Install required packages:

pip install -r requirements.txt

3. Run the app:

streamlit run app.py

# Project Structure

app.py            # Streamlit frontend
tokenizer.pkl      # Saved tokenizer
next_word_model.h5    # Trained LSTM model
max_len.txt      # Max sequence length for padding
requirements.txt   # Dependencies
README.md    # Project documentation
