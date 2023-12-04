#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 13:07:52 2023

@author: farahfrancis
"""

#load packages
### normalise please! Z normalisation, set seed = 123.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
#import graphviz
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, log_loss
from pandas.api.types import CategoricalDtype
KNeighboursClassifier = KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC, SVC
from sklearn import tree
import statistics
from sklearn.model_selection import KFold
from matplotlib import pyplot
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import euclidean_distances
#from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
import csv
from pandas.plotting import scatter_matrix
#import xgboost as xgb
from sklearn.preprocessing import StandardScaler
#from xgboost.sklearn import XGBClassifier
#%matplotlib inline



# compile all 552 csv into 1



path = '/Users/farahfrancis/Library/CloudStorage/OneDrive-UniversityofEdinburgh/PhD/CTG/CTG-opensource/csv_NN/csv'
sequences = []  # Initialize sequences as an empty list


# List all CSV files in the directory
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

for csv_file in csv_files:
    file_path = os.path.join(path, csv_file)
    print(file_path)

    df = pd.read_csv(file_path, header=0)
    
    # Drop the last column
    df = df.iloc[:, :-1]
    
    # Scale the remaining columns
    #scaled_values = scaler.fit_transform(df.values)
    
    # Append the scaled values to sequences
    sequences.append(df)

# Combine all sequences into one NumPy array if sequences exist
if sequences:
    combined_sequence = np.vstack(sequences)
    print('Combined, scaled, and column-dropped sequences shape:', combined_sequence.shape)
else:
    print('No valid data found in CSV files.')

#add target to the sequence
data_path = os.path.join(os.getcwd(), '/Users/farahfrancis/Library/CloudStorage/OneDrive-UniversityofEdinburgh/PhD/CTG/CTG-opensource/outputB.csv')
targets = pd.read_csv(data_path, delimiter = ',')
targets = targets['Apgar']

#pad the seqquences chose the max
len_sequences = []
for one_seq in sequences:
    len_sequences.append(len(one_seq))
pd.Series(len_sequences).describe()


#split dataset
train_ind = int(len(sequences)*0.7)
train = sequences[:train_ind]
test = sequences[train_ind:]

y_ind = int(len(targets)*0.7)
y_train = targets[:train_ind]
y_test = targets[train_ind:]


#Preprocessing the Data
# Reshape the input data to have the shape (number_of_samples, sequence_length, features)
X = X.reshape(-1, common_sequence_length, 2)

#Before feeding the data into the model, we need to preprocess it. This involves reshaping the data into a format suitable for the LSTM layers and encoding the class labels:
# Reshape the data
train = train.reshape((train.shape[0], train.shape[1], 1))
test = test.reshape((test.shape[0], test.shape[1], 1))
# define the Autoencoder model

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_squared_error

from numpy.random import seed
seed(1)

model = Sequential()
# Input shape should be a tuple (number of time steps, number of features)
model.add(LSTM(32, activation='relu', dropout = 0.25, recurrent_dropout = 0.25))
model.add(LSTM(16, activation='relu', return_sequences=True))
model.add(LSTM(16, activation='relu', return_sequences=False))
# model.add(LSTM(16, activation='relu', return_sequences=True))
# add early stop
model.add(Dropout(0.2))

model.compile(optimizer='adam', loss='mse')
model.summary()

# fit the model
history = model.fit(train, y_train, epochs=25, batch_size=258, validation_split=0.3, verbose=1)

 












