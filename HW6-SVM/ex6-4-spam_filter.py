from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import scipy.io as sio

train_mat = sio.loadmat('HW6-SVM/data/spamTrain.mat')
X, y = train_mat.get('X'), train_mat.get('y').ravel()
test_mat = sio.loadmat('HW6-SVM/data/spamTest.mat')
X_test, y_test = test_mat.get('Xtest'), test_mat.get('ytest').ravel()


svc = svm.SVC()
svc.fit(X, y)
pred = svc.predict(X_test)
logit = LogisticRegression()
pred = logit.predict(X_test)
print(metrics.classification_report(y_test, pred))




























