# ////////////////////////////////////////////////////////////

# MAIL CLASSIFICATION

# FILE NAME: train.py

# DEVELOPED BY: Vigneshwar Ravichandar

# TOPICS: Natural Language Processing, TensorFlow , Recurrent Neural Network

# ////////////////////////////////////////////////////////////

# IMPORTING REQUIRED LIBRARIES
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

os.system('cd ..')

print(f"TensorFlow version: {tf.__version__}")

DATASET_PATH = "data/mailData.csv"

# IMPORTING THE DATASET
data = pd.read_csv(DATASET_PATH)
print("Dataset Description:\n",data.describe())
print("Dataset Head:\n",data.head())

# SEGMENTING DATA
x = data['EmailText'].values
y = data['Label'].values

# CONVERTING STRING LABELS INTO INTEGER LABELS
labels = []
for i in y:
    if i == 'ham':
        labels.append(0)
    if i == 'spam':
        labels.append(1)

# INITIALIZING TOKENIZER OBJECT AND FITTING TO DATA
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = 500000)
tokenizer.fit_on_texts(x)

# OBTAINING WORD INDICES
indices = tokenizer.word_index

# CONVERTING TO SEQUENCES
sequences = tokenizer.texts_to_sequences(x)

# PADDING THE SEQUENCES
x_padded = tf.keras.preprocessing.sequence.pad_sequences(sequences)

# SPLITING THE DATASET TO TRAIN AND TEST SET
x_train_padded, x_val_padded, y_train, y_val = train_test_split(x_padded, labels, test_size = 0.2, random_state = 0)

# DEFINING THE NEURAL NETWORK
def spamModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size = 10000, embedding_dim = 16, input_length = 189))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.Dense(32, activation = 'relu'))
    model.add(tf.keras.Dense(8, activation = 'relu'))
    model.add(tf.keras.Dense(1, activation = 'sigmoid'))
    return model

# INITITIALIZING THE CALLBACK
early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'accuracy', mode = 'max')

model = spamModel()



