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

# DATA SEGMENTATION
data = pd.read_csv('data/mailData.csv')
emails = data['EmailText'].values
labels = data['Label'].values

# TOKENIZATION OF DATA INPUT
token = tf.keras.preprocessing.text.Tokenizer(num_words = 5572)
token.fit_on_texts(emails)
x = token.texts_to_sequences(emails)
x = np.array(x)
x = tf.keras.preprocessing.sequence.pad_sequences(x)








