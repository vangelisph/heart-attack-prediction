from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

import pandas as pd

data = pd.read_csv("../data/dataset.csv")

y = data.DEATH_EVENT
data.drop(["DEATH_EVENT"], axis=1, inplace=True)
X = data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = LinearSVC(class_weight="balanced", dual=False, tol=1e-2, max_iter=1e5)
model.fit(X_train, y_train)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)


predictions = model.predict(X_test)

ySVC_pred = model.fit(X_train, y_train).predict(X_test)

print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))
print(f1_score(y_test, predictions))

print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))

predictions1 = gnb.predict(X_test)
print(confusion_matrix(y_test, predictions1))
print(accuracy_score(y_test, predictions1))
print(f1_score(y_test, predictions1))

print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))
