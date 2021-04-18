# ////////////////////////////////////////////////////////////

# MAIL CLASSIFICATION

# FILE NAME: train.py

# DEVELOPED BY: Vigneshwar Ravichandar

# TOPICS: Natural Language Processing, TensorFlow , Recurrent Neural Network

# ////////////////////////////////////////////////////////////

# IMPORTING REQUIRED MODULES
import numpy as np
import pandas as pd
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")

DATASET_PATH = "data/mailData.csv"
MODEL_PATH = './model/spamModel'

# IMPORTING THE DATASET
data = pd.read_csv(DATASET_PATH)
print("Dataset Description:\n",data.describe())
print("Dataset Head:\n",data.head())

# SEGMENTING DATA
x = data['EmailText'].values

# INITIALIZING TOKENIZER OBJECT AND FITTING TO DATA
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = 500000)
tokenizer.fit_on_texts(x)

# OBTAINING WORD INDICES
indices = tokenizer.word_index

# GETTING A CUSTOM EMAIL FROM THE USER
email = input("Enter the received mail: ")

# CONVERTING TO SEQUENCES
email_sequence = tokenizer.texts_to_sequences(email)

# PADDING THE SEQUENCES
email_padded = tf.keras.preprocessing.sequence.pad_sequences(email_sequence)

