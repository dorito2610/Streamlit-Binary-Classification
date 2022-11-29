# Packages
import pandas as pd
from sklearn import metrics  # for dataframes
import streamlit as st  # streamlit
import numpy as np  # for statistics calculations
from sklearn.svm import SVC  # for support vector machines
from sklearn.ensemble import RandomForestClassifier  # for random forests
# to perform binary classification
from sklearn.linear_model import LogisticRegression
# for Onehotencoding or normalising the values
from sklearn.preprocessing import LabelEncoder
# for training and testing the data
from sklearn.model_selection import train_test_split
# for plotting the metrics
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, precision_score, recall_score


def main():
    # Title
    st.title("Binary Clssification Web App")
    st.sidebar.title("Binary Classifier")
    st.markdown("Are the mushrooms edible or poisonous? 🍄")

    @st.cache(persist=True)  # Memoizes the function
    def load():  # To load the dataset
        data = pd.read_csv("mushrooms.csv")
        label = LabelEncoder()  # Normalises values that is, categorical data to numerical data
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache(persist=True)
    def split(df):  # To perform training and testing
        y = df.type  # Target variable
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=0)  # 70% training 30% testing
        return x_train, x_test, y_train, y_test

# plotting the metrics - confusion matrix,ROC (reciever operating characteristic) Curve,
# precision recall curve

    def plot_metrics(metrics_list):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if 'Confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            plot_confusion_matrix(model, x_test, y_test,
                                  display_labels=class_names)
            st.pyplot()
        if 'ROC Curve' in metrics_list:
            st.subheader('ROC Curve')
            # display_labels=class_names)
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()
        if 'Precision Recall Curve' in metrics_list:
            st.subheader('Precision Recall Curve')
            plot_precision_recall_curve(
                model, x_test, y_test)  # display_labels=class_names)
            st.pyplot()

    df = load()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox(  # Selecting the classifier
        "Classifier", ('Support Vector Machine', 'Logistic Regression', 'Random Forest'))
    st.subheader("Mushroom dataset")
    st.write(df)
    st.write("Dataset dimensions are", str(df.shape))

# Support Vector machine

    if classifier == 'Support Vector Machine':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input(
            "C (Regularisation Parameter)", 0.01, 10.00, step=0.01)
        kernel = st.sidebar.radio("Kernel", ('rbf', 'linear'), key='kernel')
        gamma = st.sidebar.radio(
            'Gamma (Kernel coefficient)', ('scale', 'auto'), key='gamma')
        metrics = st.sidebar.multiselect(
            "Select plot metrics:", ('Confusion Matrix', 'Precision Recall Curve', 'ROC Curve'))
# Algorithm
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
        # C-Regularizzation Parameter, kernel-{'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
        # Specifies the kernel type to be used in the algorithm,gamma- Kernel coefficient for 'rbf', 'poly' and 'sigmoid
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy:", accuracy.round(2))
            st.write("Precision:", precision_score(
                y_test, y_pred, labels=class_names).round(2))
            st.write("Recall:", recall_score(
                y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

# Logisitic Regression

    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input(
            "C (Regularisation Parameter)", 0.01, 10.00, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider(
            "Max number of iterations:", 100, 500, key='max_iter')
        metrics = st.sidebar.multiselect(
            "Select plot metrics:", ('Confusion Matrix', 'Precision Recall Curve', 'ROC Curve'))
# Algorithm
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            # C-Regularization parameter, max_iter= max number of iterations
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy:", accuracy.round(2))
            st.write("Precision:", precision_score(
                y_test, y_pred, labels=class_names).round(2))
            st.write("Recall:", recall_score(
                y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

# Random Forest

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input(
            "The number of trees in the forest", 100, 500, step=10, key='n_est')  # num of trees
        max_depth = st.sidebar.number_input(
            "The maximum depth of the tree", 1, 20, step=1, key='max_step')  # max depth
        bootstrap = st.sidebar.radio(
            'Bootstrap sample when building trees', ('True', 'False'), key='bootstrap')
        # bootstrap : bool, default=True Whether bootstrap samples are used when building trees.
        # If False, the whole dataset is used to build each tree.
        metrics = st.sidebar.multiselect(
            "Select plot metrics:", ('Confusion Matrix', 'Precision Recall Curve', 'ROC Curve'))
# Algorithm
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy:", accuracy.round(2))
            st.write("Precision:", precision_score(
                y_test, y_pred, labels=class_names).round(2))
            st.write("Recall:", recall_score(
                y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)


if __name__ == '__main__':
    main()
