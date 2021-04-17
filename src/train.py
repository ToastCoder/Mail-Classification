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
print("Dataset Head\n",data.head())

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





