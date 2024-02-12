import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

from sklearn import datasets

            
# chargement du jeu de données breast_cancer 
data = datasets.load_iris()
# variables dépendantes (features)
X = data.data
# variable indépendante (target) 
y = data.target

# importer la méthode train_test_split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# importer la classe LogisticRegression
from sklearn.linear_model import LogisticRegression
# créer le modèle de LogisticRegression
# avec les paramètres par défaut 
classifier = LogisticRegression()
# notre modèle est stocké dans la variable classifier

# entraîner le modèle 
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

import pickle
filename='model.pkl'
pickle.dump(classifier, open(filename, 'wb'))