# -*- coding: utf-8 -*-
"""
Script to learn a model with Scikit-learn.

Created on Mon Oct 24 20:51:47 2022

@author: ValBaron10
"""

import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import pickle
from joblib import dump
from features_functions import compute_features


count = 100
learningFeatures = False

for s in range(1, count+1):

    file = open(f"data/{s}.pkl", 'rb')
    input_sig = pickle.load(file)


    # Compute the signal in three domains
    sig_sq = input_sig**2
    sig_t = input_sig / np.sqrt(sig_sq.sum())
    sig_f = np.absolute(np.fft.fft(sig_t))
    sig_c = np.absolute(np.fft.fft(sig_f))

    # Compute the features and store them
    features_list = []
    N_feat, features_list = compute_features(sig_t, sig_f[:sig_t.shape[0]//2], sig_c[:sig_t.shape[0]//2])
    features_vector = np.array(features_list)[np.newaxis,:]

    # Store the obtained features in a np.arrays
    if type(learningFeatures) == bool: learningFeatures = features_vector.copy()
    else: learningFeatures = np.append(learningFeatures, features_vector, axis=0)

# Store the labels
with open("data/labels.pkl", "rb") as labels_file:
    learningLabels = pickle.load(labels_file) # np.array with labels in it, for each signal
    learningLabels = learningLabels[:count]
    

# Encode the class names
labelEncoder = preprocessing.LabelEncoder().fit(learningLabels)
learningLabelsStd = labelEncoder.transform(learningLabels)

# Learn the model
model = svm.SVC(C=10, kernel='linear', class_weight=None, probability=False)
scaler = preprocessing.StandardScaler(with_mean=True).fit(learningFeatures)
learningFeatures_scaled = scaler.transform(learningFeatures)


x_train, x_test, y_train, y_test = train_test_split(
    learningFeatures_scaled, learningLabelsStd, test_size=0.3, shuffle=False
)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(
    f"Classification report for classifier {model}:\n"
    f"{classification_report(y_test, y_pred)}\n"
)


# Export the scaler and model on disk
dump(scaler, "SCALER")
dump(model, "SVM_MODEL")

