# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:29:04 2024

@author: Shubham Sharad Mavle
"""


from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import pandas as pd

digits = load_digits()

plt.gray()
for i in range(5):
    plt.matshow(digits.images[i])
    
dir(digits)

digits.data[0]

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

model.fit(X_train, y_train)

pred = model.predict(X_test)
np.mean(pred==y_test)

import seaborn as sns

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
cm
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('predicted')
plt.ylabel('truth')