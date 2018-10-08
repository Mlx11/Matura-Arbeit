# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 16:26:42 2018

@author: Arbeiten
"""

from sklearn import datasets
import sklearn
import numpy as np

iris = datasets.load_iris()
X = iris.data[:100, :2]
y = iris.target[:100]
y = np.where(y==0, -1, 1)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.33)

np.save("Iris_X_train", X_train)
np.save("Iris_X_test", X_test)
np.save("Iris_y_train", y_train)
np.save("Iris_y_test", y_test)

