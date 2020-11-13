#IMPORTING REQUIRED LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.model_selection import GridSearchCV


# GETTING DATA
data = pd.read_csv("data/mailData.csv")
x = data.iloc[:,0].values
y = data.iloc[:,-1].values

# SPLITTING THE DATA INTO TEST DATA AND TRAIN DATA
x_train,y_train = x[0:4457],y[0:4457]
x_test,y_test = x[4457:],y[4457:]

# EXTRACTING FEATURES
cv = CountVectorizer()  
features = cv.fit_transform(x_train)

## BUILDING A MODEL
tune = {'kernel': ['rbf','linear'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}
model = GridSearchCV(svm.SVC(), tune)
model.fit(features,y_train)
print(model.best_params_)

# TESTING THE ACCURACY
print(model.score(cv.transform(x_test),y_test))
