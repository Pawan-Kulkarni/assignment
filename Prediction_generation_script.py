# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.ensemble import RandomForestClassifier


#######$####$######INPUT TEXT PATH HERE ####################

test_path = "test_set.csv"
############################################################


def training(test_path):
    data = pd.read_csv("training_set.csv")
    data.drop(['Unnamed: 0'],axis=1,inplace=True)

    corr_matrix = data.corr().abs()


    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))


    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]

    data.drop(to_drop, axis=1, inplace=True)

    data.drop_duplicates(subset=None, inplace=True)
    cols = data.columns[:-1]
    X = data[cols]
    # create a scaler object
    scaler = MinMaxScaler()
    scaler.fit_transform(X)
    # fit and transform the data
    X_norm = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    Y = data["Y"]
    logistic = LogisticRegression()

    sfs1 = sfs(logistic, k_features=30, forward=False, verbose=1) 
    sfs1 = sfs1.fit(X_norm, Y)
    df = X_norm[list(sfs1.k_feature_names_)]
    x_train_r,x_test_r,y_train_r,y_test_r=train_test_split(df,Y,test_size=.2,stratify=Y,random_state=42)
    clf = RandomForestClassifier(max_features='sqrt',max_depth=30,criterion='entropy',min_samples_leaf=4,  n_estimators=300,class_weight={0:40, 1:60})
    clf.fit(x_train_r,y_train_r)
    test = pd.read_csv(test_path)
    test_norm = pd.DataFrame(scaler.fit_transform(test), columns=test.columns)
    reduced_test_norm = test_norm[x_train_r.columns] 
    test_pred_r = clf.predict(reduced_test_norm)
    print(test_pred_r)

training(test_path)