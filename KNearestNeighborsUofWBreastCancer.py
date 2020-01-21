import numpy as np
#removed from sklearn import preprocessing, cross_validation, neighbors
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import neighbors, svm
import pandas as pd

#importing file from sublime text of breast cancer data from the University of Wisconsin
#using k nearest neighbors algo to discover more about the data set, I may have to change the
#above settings surrounding sklearn and cross_validation

df = pd.read_csv('id,clump_thickness,unif_cell_size,unif_cell_shape,')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
 
#previously inserted crossvalidation. infront of train_test_split below

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#added NuSVC to SVC to impove accuracy by 30% without accuracy drops to ~65%

clf = svm.NuSVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 2, 2, 1, 2, 3, 2, 1]])

example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)
print(prediction)


