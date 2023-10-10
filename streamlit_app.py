import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

# Define your dataset or load it here (replace this with your actual data)
df = pd.read_excel('resources/Raisin_Dataset.xlsx')

# Define a function to calculate accuracy and display the confusion matrix
def evaluate_classifier(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, cm

#split data into feature columns and labels
col_names = df.columns.tolist()
#exclude the label from the list of col names
feature_cols = col_names[:-1]
X = df[feature_cols]
y = df['Class']


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build and train the models
rf_clf = RandomForestClassifier(n_estimators=101)
lr_clf = LogisticRegression()
svc_clf = SVC(kernel='rbf')

rf_clf.fit(X_train, y_train)
lr_clf.fit(X_train, y_train)
svc_clf.fit(X_train, y_train)

# Create a Streamlit app
st.title("Classification Algorithms Dashboard")

# Buttons to evaluate and display models
if st.button("Evaluate Random Forest"):
    accuracy, cm = evaluate_classifier(rf_clf, X_test, y_test)
    st.write(f"Random Forest Accuracy: {accuracy:.2f}")
    st.write("Random Forest Confusion Matrix:")
    st.pyplot(plt.figure(figsize=(6, 6)))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y)).plot()

if st.button("Evaluate Logistic Regression"):
    accuracy, cm = evaluate_classifier(lr_clf, X_test, y_test)
    st.write(f"Logistic Regression Accuracy: {accuracy:.2f}")
    st.write("Logistic Regression Confusion Matrix:")
    st.pyplot(plt.figure(figsize=(6, 6)))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y)).plot()

if st.button("Evaluate Support Vector Classifier"):
    accuracy, cm = evaluate_classifier(svc_clf, X_test, y_test)
    st.write(f"SVC Accuracy: {accuracy:.2f}")
    st.write("SVC Confusion Matrix:")
    st.pyplot(plt.figure(figsize=(6, 6)))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y)).plot()

# Button to list models by accuracy
if st.button("List Models by Accuracy"):
    accuracies = {
        "Random Forest": metrics.accuracy_score(y_test, rf_clf.predict(X_test)),
        "Logistic Regression": metrics.accuracy_score(y_test, lr_clf.predict(X_test)),
        "Support Vector Classifier": metrics.accuracy_score(y_test, svc_clf.predict(X_test)),
    }
    sorted_models = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
    st.write("Models Ranked by Accuracy:")
    for model, accuracy in sorted_models:
        st.write(f"{model}: {accuracy:.2f}")

