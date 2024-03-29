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
MODEL_PATH = './model/spamModel'

# IMPORTING THE DATASET
data = pd.read_csv(DATASET_PATH)
print("Dataset Description:\n",data.describe())
print("Dataset Head:\n",data.head())

# CONVERTING STRING LABELS INTO INTEGER LABELS
data['binary_label'] = data['Label'].map({'ham':0, 'spam': 1})

# SEGMENTING DATA
x = data['EmailText'].values
y = data['binary_label'].values

# SPLITTING THE DATASET TO TRAIN AND TEST SET
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = 0.33)

# INITIALIZING TOKENIZER OBJECT AND FITTING TO DATA
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = 20000)
tokenizer.fit_on_texts(x_train)

# OBTAINING WORD INDICES
indices = tokenizer.word_index
v = len(indices)
print(f"Found {v} unique tokens")

# CONVERTING TO SEQUENCES
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_val)

# PADDING THE SEQUENCES
x_train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_sequences)
print(f"Shape of the training data tensor: {x_train_padded.shape}")

length = x_train_padded.shape[1]

x_val_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen = length)
print(f"Shape of the testing data tensor: {x_val_padded.shape}")

# CONVERTING LISTS TO NUMPY ARRAYS
x_train_padded = np.array(x_train_padded)
x_val_padded = np.array(x_val_padded)
y_train = np.array(y_train)
y_val = np.array(y_val)

# DEFINING THE NEURAL NETWORK
def spamModel(length,v):

    # USED PARAMETERS
    embedding_dim = 20
    hidden_state_dim = 15

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(length,))
    model.add(tf.keras.layers.Embedding(v+1, embedding_dim))
    model.add(tf.keras.layers.LSTM(hidden_state_dim, return_sequences = True))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(8,activation = 'relu'))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    return model

# INITITIALIZING THE CALLBACK
early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'accuracy', mode = 'max')

model = spamModel(length,v)

# TRAINING THE MODEL
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(x_train_padded, y_train, batch_size = 5, epochs = 10, validation_data = (x_val_padded,y_val),callbacks = early_stopping)

# PLOTTING THE GRAPH FOR TRAIN-LOSS AND VALIDATION-LOSS
plt.figure(0)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Graph')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train Loss', 'Validation Loss'], loc='upper left')
plt.show()
plt.savefig('graphs/loss_graph.png')

# PLOTTING THE GRAPH FOR TRAIN-ACCURACY AND VALIDATION-ACCURACY
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy Graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='upper left')
plt.show()
plt.savefig('graphs/acc_graph.png')

# CALCULATING THE ACCURACY
score = model.evaluate(x_val_padded, y_val)
print(f"Model Accuracy: {round(score[1]*100,4)}")

# SAVING THE MODEL
tf.keras.models.save_model(model,MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")

