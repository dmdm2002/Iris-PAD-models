import pandas as pd
from sklearn import svm
import numpy as np
from sklearn.model_selection import GridSearchCV

# A feature Data Load
A = pd.read_csv('../../ROC_Curve/prov/DCLNet/feature/nd/1-fold/A_feature_29.csv')
B = pd.read_csv('../../ROC_Curve/prov/DCLNet/feature/nd/1-fold/B_blur_feature_29.csv')

print(B)

A_label = A['128'].astype('float32')
A_feature = A.drop('128', axis=1).astype('float32')

B_label = B['128'].astype('float32')
B_feature = B.drop('128', axis=1).astype('float32')

print(B_feature)
print(B_label)
svm_clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')

svm_clf.fit(A_feature, A_label)

score = svm_clf.score(B_feature, B_label)

print(score)