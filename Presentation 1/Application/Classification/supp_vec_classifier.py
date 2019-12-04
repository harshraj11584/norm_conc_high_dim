import numpy as np
import pandas as pd 
import sklearn.svm
import sklearn.model_selection
import sklearn.preprocessing


data = pd.read_csv('autoUniv-au6-250-drift-au6-cd1-500.csv')
# print(data.head())

X = data._get_numeric_data()
Y = data['Class']

# print(Y)
enc = sklearn.preprocessing.LabelEncoder()
Y = enc.fit_transform(Y)
# print(Y) 
# print(enc.classes_)
# print(enc.transform(enc.classes_))

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y, test_size=0.33, random_state=42)

clf = sklearn.svm.SVC(gamma='scale',kernel='linear')
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)
print(y_pred)
print(y_test)