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

# CONVERTING LISTS TO NUMPY ARRAYS
x_train_padded = np.array(x_train_padded)
x_val_padded = np.array(x_val_padded)
y_train = np.array(y_train)
y_val = np.array(y_val)

# DEFINING THE NEURAL NETWORK
def spamModel():

    vocab_size = 10000
    embedding_dim = 16
    max_length = 189

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(32, activation = 'relu'))
    model.add(tf.keras.layers.Dense(8, activation = 'relu'))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    return model

# INITITIALIZING THE CALLBACK
early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'accuracy', mode = 'max')

model = spamModel()

# TRAINING THE MODEL
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(x_train_padded, y_train, batch_size = 5, epochs = 10, validation_data = (x_val_padded,y_val))

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

