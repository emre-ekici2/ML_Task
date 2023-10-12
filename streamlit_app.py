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

# load dataset into dataframe
df = pd.read_excel('resources/Raisin_Dataset.xlsx')

#split data into feature columns and labels
col_names = df.columns.tolist()
#exclude the label from the list of col names
feature_cols = col_names[:-1]
X = df[feature_cols]
y = df['Class']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

st.title("r0874339 ML Task: Benchmarking 3 classification algorithms")

st.write("The dataset used for this task was the raisin dataset from the UCI datasets page")

st.header("EDA")
st.write("First lets make sure our dataset is structured properly by checking the first and last rows")

if st.button("View dataframe"):
    st.write("Are there any null values in here?")
    st.write(df.isnull().any().any())
    st.write("Would you look at that, no null values. Awesome! Less work for me :)")
    st.write("Looks like the structure is good as well, but there might be a problem! It seems like the data was entered in order of the raisins class. This might not be a problem but to be sure let's randomize the rows in our dataframe.")
    st.write(df.head(5))
    st.write(df.tail(5))
    
    
if st.button("Randomize dataframe rows"):
    # Randomize the rows
    randomized_df = df.sample(frac=1, random_state=42)  # Set a specific seed for reproducibility (use any integer)

    # Reset the index of the randomized DataFrame
    randomized_df.reset_index(drop=True, inplace=True)

    df = randomized_df

    st.write("I think that's better! Maybe? Why not try comparing the algorithms with both randomized and non randomized rows?")
    st.write(df.head(5))
    st.write(df.tail(5))    




st.header("Determine hyperparameters for our algorithms")

st.write("Input amount of bags for Random Forest")
n_estimators_input = st.number_input("Enter n_estimators for the Random Forest algorithm(default = 101, min = 1, max = 500)", min_value=1, max_value=500, value=101)


st.write("Hyperparameters for SVC")
kernel_input = st.selectbox("Select Kernel", ["linear", "rbf", "poly", "sigmoid"])
C_input = st.number_input("Enter C (Regularization Parameter)", min_value=0.001, max_value=100.0, value=1.0)


if st.button("Compare Algorithms"):

    #building the models
    rf_clf = RandomForestClassifier(n_estimators=n_estimators_input)
    lr_clf = LogisticRegression()
    svc_clf = SVC(kernel=kernel_input, C = C_input)

    #training the models
    rf_clf = rf_clf.fit(X_train, y_train)
    lr_clf = lr_clf.fit(X_train, y_train)
    svc_clf = svc_clf.fit(X_train, y_train)

    #making predictions
    rf_pred = rf_clf.predict(X_test)
    lr_pred = lr_clf.predict(X_test)
    svc_pred = svc_clf.predict(X_test)


    rf_acc = metrics.accuracy_score(y_test, rf_pred).round(2)
    rf_cm = confusion_matrix(y_test, rf_pred)

    lr_acc = metrics.accuracy_score(y_test, lr_pred).round(2)
    lr_cm = confusion_matrix(y_test, lr_pred)

    svc_acc = metrics.accuracy_score(y_test, svc_pred).round(2)
    svc_cm = confusion_matrix(y_test, svc_pred)





    st.write("Confusion Matrices and Accuracies:")
    st.write("You can enlarge the plot by hovering over the image and clicking the enlarge button on the top right.")
    # Create a single subplot for all three plots
    fig, axs = plt.subplots(1, 3, figsize=(15, 3))  # 1 row, 3 columns for the plots

    # Plot Random Forest Confusion Matrix
    axs[0].set_title(f"Random Forest Accuracy: {rf_acc}")
    axs[0].set_xlabel("Predicted")
    axs[0].set_ylabel("True")
    ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=np.unique(y)).plot(ax=axs[0])

    # Plot Logistic Regression Confusion Matrix
    axs[1].set_title(f"Logistic Regression Accuracy: {lr_acc}")
    axs[1].set_xlabel("Predicted")
    axs[1].set_ylabel("True")
    ConfusionMatrixDisplay(confusion_matrix=lr_cm, display_labels=np.unique(y)).plot(ax=axs[1])

    # Plot SVC Confusion Matrix
    axs[2].set_title(f"SVC Accuracy: {svc_acc}")
    axs[2].set_xlabel("Predicted")
    axs[2].set_ylabel("True")
    ConfusionMatrixDisplay(confusion_matrix=svc_cm, display_labels=np.unique(y)).plot(ax=axs[2])

    # Display the entire subplot
    st.pyplot(fig)


