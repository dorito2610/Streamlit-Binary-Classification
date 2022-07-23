import pandas as pd  # for dataframes
import streamlit as st
import numpy as np  # for statistics calculations
from sklearn.svm import SVC  # for support vector machines
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, precision_score, recall_score


def main():
    # Title
    st.tile("Binary Clssification Web App")
    st.sidebar("Binary Classifier")


if __name__ == '__main__':
    main()
