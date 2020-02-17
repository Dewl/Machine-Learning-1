import numpy as np
import pandas as pd
import re

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
from sklearn.model_selection import GridSearchCV

def explode_product(data, index):
    for i, row in data.iterrows():
        ddict = defaultdict(int)
        for item in row['Products']:
            cate = item.split('/')[index]
            ddict[cate] += 1
            
        for key in ddict:
            if key not in data.columns:
                data.insert(loc=len(data.columns), column=key,
                            value=0, allow_duplicates=False)
            data.at[i, key] = ddict[key]

    return data

def extract_feature(data_loc, label_loc):
    column_names = ['ID', 'Start time', 'End time', 'Products']
    data = pd.read_csv(data_loc, names=column_names, header=None, sep=',')
    label = pd.read_csv(label_loc, names=['Gender'], header=None)

    data['Start time'] = pd.to_datetime(data['Start time'])
    data['End time'] = pd.to_datetime(data['End time'])
    data['Duration'] = data['End time'] - data['Start time']
    data['Duration'] = data['Duration'].dt.total_seconds().astype(int)

    # Day | Month | Start Hour | End Hour | Date | Number of Product | Duration | Avarage Viewing Time
    data['Day'] = data['Start time'].dt.day
    data['Month'] = data['Start time'].dt.month
    data['Start Hour'] = data['Start time'].dt.hour
    data['End Hour'] = data['End time'].dt.hour
    data['Date'] = data['Start time'].dt.dayofweek

    data['Products'] = data.apply(lambda x: data['Products'].str.split(";"))
    data['Number of Product'] = data['Products'].str.len()
    data['Avarage Viewing Time'] = data['Duration'] / data['Number of Product']

    data = data.drop(['ID'], axis=1)
    return data, label

def match_feature(training_data, testing_data):
    for column in training_data.columns:
        if column not in testing_data.columns:
            testing_data.insert(loc=len(testing_data.columns), column=column, value=0, allow_duplicates=False)
    return testing_data

def process(training_data_loc, training_label_loc, testing_data_loc, testing_label_loc):

    ### Training data
    print("--Extracting features for training--")
    training_data, training_label = extract_feature(
        training_data_loc, training_label_loc)
    # Drop all training cases with day difference >= 1
    print("--Dropping some training cases--")
    indexes = training_data.index[(training_data['End time'].dt.day -
                          training_data['Start time'].dt.day) >= 1].tolist()
    training_data = training_data.drop(indexes)
    training_label = training_label.drop(indexes)
    # Explode product column for training data
    print("--Exploding product column for training dataset--")
    training_data = explode_product(training_data, 0)
    training_data = explode_product(training_data, 1)
    training_data = explode_product(training_data, 2)
    training_data = training_data.drop(['Start time', 'End time', 'Products'], axis=1)


    ### Testing data
    if(testing_data_loc == "" and testing_label_loc == ""):
        print("--Spliting data for testing--")
        X_train, X_test, y_train, y_test = train_test_split(
            training_data, training_label, test_size=0.2)
    else:
        print("--Extracting features for testing--")
        testing_data, testing_label = extract_feature(testing_data_loc, testing_label_loc)
        testing_data = match_feature(training_data, testing_data)
        # Don't drop cases with date difference
        # Explode product column for testing data
        print("--Exploding product column for testing dataset--")
        testing_data = explode_product(testing_data, 0)
        testing_data = explode_product(testing_data, 1)
        testing_data = explode_product(testing_data, 2)
        testing_data = testing_data.drop(['Start time', 'End time', 'Products'], axis=1)

        # Finalize
        X_train = training_data
        X_test = testing_data
        y_train = training_label
        y_test = testing_label

    # SVM
    print("--Setup SVM--")
    clf = svm.SVC(kernel='rbf', gamma=0.0001, C=1000)
    clf.fit(X_train, y_train.values.ravel())
    y_pred = clf.predict(X_test)

    print("--Prediction result--")
    print(confusion_matrix(y_test, y_pred, labels=['male', 'female']))

if __name__ == "__main__":
    training_data_loc = "trainingData.csv"
    training_label_loc = "trainingLabels.csv"
    testing_data_loc = ""
    testing_label_loc = ""

    process(training_data_loc, training_label_loc,
            testing_data_loc, testing_label_loc)
