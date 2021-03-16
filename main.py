# IMPORTING REQUIRED LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk

# GETTING THE DATA
dataset = pd.read_csv('data/mailData.csv')

# CLEANING THE TEXT
nltk.download('stopwords')
corpus = []
for i in range(0, 5572):
    review = re.sub('[^a-zA-Z]', ' ', dataset['EmailText'][i])
    review = review.lower()
    review = review.split()
    ps = nltk.stem.porter.PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(nltk.stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# CREATING BAG OF WORDS MODEL
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 0].values

# SPLITTING THE DATASET INTO TEST AND TRAIN SET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# FITTING NAIVR BAYES
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

