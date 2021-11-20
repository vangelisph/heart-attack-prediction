import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score, precision_score
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

# Team and Project Information

st.title("Heart attack event prediction")
st.markdown("---")
st.markdown("Team members:")
st.markdown("- **Menelaos Kotoglou**")
st.markdown("- **Vangelis Photiou**")
st.markdown("- **Stefanos Christodoulou**")

# Load data

data = pd.read_csv("dataset.csv")

st.markdown("You can see our initial dataset here:")
st.write(data)

st.markdown("The dataset given to us and our project's goal we were sure that it is a classification problem.")

# Data Exploration

st.markdown("### Data Preprocessing")
st.markdown("---")

fig1, ax1 = plt.subplots()
ax1 = sns.pairplot(data[["creatinine_phosphokinase", "ejection_fraction",
                       "platelets", "serum_creatinine",
                       "serum_sodium", "time", "DEATH_EVENT"]], hue = "DEATH_EVENT", 
            diag_kind='kde', kind='scatter', palette='husl')
st.pyplot(ax1)
st.markdown("Digging through our dataset we saw that in the `age` and `platelets` fields there were some float values but the rest of the entries had integer values.")
st.markdown("We had to get over them, so we used the `ceil` and `floor` python functions to make them int values.")

code_block_1 = """
data['age']=data['age'].apply(np.ceil)
data['platelets']=data['platelets'].apply(np.floor)
data['age']=data['age'].astype(int)
data['platelets']=data['platelets'].astype(int)
"""

st.code(code_block_1, language="python")

data['age']=data['age'].apply(np.ceil)
data['platelets']=data['platelets'].apply(np.floor)
data['age']=data['age'].astype(int)
data['platelets']=data['platelets'].astype(int)
data['serum_creatinine'] = data['serum_creatinine'].map('{:,.1f}'.format)

st.markdown("Hopefully, we didn't find any missing values in the dataset.")

cont_features = ['age', 'platelets']

st.markdown("We use the .describe() function to describe some basic")

st.write(data[cont_features].describe())

st.markdown("After applying these changes to our dataset, it finally looks like this:")

st.write(data)

# Data Visualization

st.markdown("This is a correlation matrix, explaining the correlation between our features.")
correlation_matrix = data.corr()
fig, ax = plt.subplots()
ax = sns.heatmap(correlation_matrix, annot=True, vmax=1, square=True, linewidths=.5, cmap="YlGnBu", fmt=".2f")
st.pyplot(fig)

st.markdown("### Model training")
st.markdown("---")
st.markdown("Following this image from our lecture: ")
st.image("ml_map.png")  # see *
st.markdown("Since our dataset has approximately 300 entries and the data is labeled, we decided that it would be best if we first experiment with the **LinearSVC** algorithm. Following this decision and applying the LinearSVC algorithm we found out that it's performance was poor taking into consideration these results:")

y = data.DEATH_EVENT
data.drop(["DEATH_EVENT"], axis=1, inplace=True)
X = data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# LinearSVC

st.markdown("### LinearSVC")
st.markdown("---")

model = LinearSVC(class_weight="balanced", dual=False, tol=1e-2, max_iter=1e5)
model.fit(X_train, y_train)

# GaussianNB

gnb = GaussianNB()
yGNB_pred = gnb.fit(X_train, y_train).predict(X_test)

predictions = model.predict(X_test)

ySVC_pred = model.fit(X_train, y_train).predict(X_test)

st.markdown("Confusion Matrix:")
st.write(confusion_matrix(y_test, predictions))
st.markdown("Accuracy Score:" + " %.2f" % accuracy_score(y_test, predictions))
st.write("%.2f" % f1_score(y_test, predictions))
st.markdown("Recall:" + " %.2f" % recall_score(y_test, predictions))
st.markdown("Precision:" + " %.2f" % precision_score(y_test, predictions))

st.write(
    "Number of mislabeled points out of a total %d points : %d"
    % (X_test.shape[0], (y_test != ySVC_pred).sum())
)

st.markdown("Taking under consideration these results we thought that trying the Naive Bayes algorithm could be helpful in our case. Turned out that we were right, since the model's performance increased significantly.")
st.markdown("### Gaussian Naive Bayes")
st.markdown("---")

predictions1 = gnb.predict(X_test)
st.markdown("Confusion Matrix:")
st.write(confusion_matrix(y_test, predictions1))
st.markdown("Accuracy Score:" + " %.2f" % accuracy_score(y_test, predictions1))
st.markdown("F1 score:" + " %.2f" % f1_score(y_test, predictions1))
st.markdown("Recall:" + " %.2f" % recall_score(y_test, predictions1))
st.markdown("Precision:" + " %.2f" % precision_score(y_test, predictions1))

st.write(
    "Number of mislabeled points out of a total %d points : %d"
    % (X_test.shape[0], (y_test != yGNB_pred).sum())
)

# Result Visualization



# Conclusion


st.markdown("### Conclusion")
st.markdown("---")
st.markdown("The Naive Bayes algorithm performace is far better than LinearSVC in our case comparing the two algorithms' perfomance.")

st.markdown("You can find the project's code and data  [here](https://github.com/koti/heart-attack-prediction). For any questions, please don't hesitate to contact us at **__ms.kotoglou@edu.cut.ac.cy__** and **__vm.photiou@edu.cut.ac.cy__**.")