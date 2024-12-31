from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import numpy as np
import pandas as pd
import scipy.io as sio

mat = sio.loadmat('HW6-SVM/data/ex6data3.mat')
training = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
training['y'] = mat.get('y')

cv = pd.DataFrame(mat.get('Xval'), columns=['X1', 'X2'])
cv['y'] = mat.get('yval')

candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

combination = [(C, gamma) for C in candidate for gamma in candidate]
search = []

for C, gamma in combination:
    svc = svm.SVC(C=C, gamma=gamma)
    svc.fit(training[['X1', 'X2']], training['y'])
    search.append(svc.score(cv[['X1', 'X2']], cv['y']))
best_score = search[np.argmax(search)]
best_param = combination[np.argmax(search)]

best_svc = svm.SVC(C = best_param[0], gamma=best_param[1])
best_svc.fit(training[['X1', 'X2']], training['y'])
y_pred = best_svc.predict(cv[['X1', 'X2']])

parameters = {'C': candidate, 'gamma': candidate}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, n_jobs=-1)
clf.fit(training[['X1', 'X2']], training['y'])

y_pred = clf.predict(cv[['X1', 'X2']])
all = res = pd.concat([training, cv], axis=0, ignore_index=True)

parameters = {'C': candidate, 'gamma': candidate}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, n_jobs=-1)
clf.fit(all[['X1', 'X2']], all['y'])

y_pred = clf.predict(cv[['X1', 'X2']])
print(metrics.classification_report(cv['y'], y_pred))
















