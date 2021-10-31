import streamlit as st
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score,classification_report
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

st.title("Heart attack event prediction")
st.markdown("---")
st.markdown("Team members: **Menelaos Kotoglou**, **Vangelis Photiou**, **Stefanos Christodoulou** ")

data = pd.read_csv("dataset.csv")

st.markdown("Regarding our given data and our project's goal we were sure that it is a classification problem.")
st.markdown("You can see our initial dataset here:")
st.write(data)


st.markdown("Following this image from our lecture: ")
st.image("ml_map.png")  # see *
st.markdown("We decided that it would be best if we first experiment with the **LinearSVC** algorithm. Following this decision and applying the LinearSVC algorithm we found out that it's performance was poor taking into consideration these results:")
st.markdown("### LinearSVC")
st.markdown("---")

y = data.DEATH_EVENT
data.drop(["DEATH_EVENT"], axis=1, inplace=True)
X = data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = LinearSVC(class_weight="balanced", dual=False, tol=1e-2, max_iter=1e5)
model.fit(X_train, y_train)

gnb = GaussianNB()
yGNB_pred = gnb.fit(X_train, y_train).predict(X_test)


predictions = model.predict(X_test)

ySVC_pred = model.fit(X_train, y_train).predict(X_test)

st.markdown("Confusion Matrix:")
st.write(confusion_matrix(y_test, predictions))
st.markdown("Accuracy Score:")
st.write(accuracy_score(y_test, predictions))
st.markdown("F1 score:")
st.write(f1_score(y_test, predictions))
st.markdown("Recall:")
st.write(recall_score(y_test, predictions))
st.markdown("Classification report:")
st.write(classification_report(y_test, predictions))

st.write(
    "Number of mislabeled points out of a total %d points : %d"
    % (X_test.shape[0], (y_test != ySVC_pred).sum())
)

st.markdown("Taking under consideration these results we thought that trying the Naive Bayes algorithm could be helpful in our case. Turned out that we were right, since it is visible from our results.")
st.markdown("### Gaussian Naive Bayes")
st.markdown("---")

predictions1 = gnb.predict(X_test)
st.markdown("Confusion Matrix:")
st.write(confusion_matrix(y_test, predictions1))
st.markdown("Accuracy Score:")
st.write(accuracy_score(y_test, predictions1))
st.markdown("F1 score:")
st.write(f1_score(y_test, predictions1))
st.markdown("Recall:")
st.write(recall_score(y_test, predictions1))
st.markdown("Classification report:")
st.write(classification_report(y_test, predictions1))

st.write(
    "Number of mislabeled points out of a total %d points : %d"
    % (X_test.shape[0], (y_test != yGNB_pred).sum())
)

st.markdown("You can find the project's code and data  [here](https://github.com/koti/heart-attack-prediction). For any questions, please don't hesitate to contact us at **__ms.kotoglou@edu.cut.ac.cy__** and **__vm.photiou@edu.cut.ac.cy__**.")