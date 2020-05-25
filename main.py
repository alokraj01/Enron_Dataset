#!/usr/bin/env python
# coding: utf-8
#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
#sys.path.append('E:\Data_Analysis\Nanodegree\L7_Machine_learning\for_submission_24_May')
from feature_format import featureFormat, targetFeatureSplit
#tester was modified. cross_validation module doesn't work now and is replaced by model_selection. tester_v1 is for that
from tester_v1 import dump_classifier_and_data 
#Import useful libraries for analysis and visulisation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#Convering dict to list format. 
name=[]
features_unique=[]
for key in data_dict:
    name.append(key)
    for key2 in data_dict[key]:
        #features.append(key2)
        if key2 not in list(features_unique):
            features_unique.append(key2)
#Names in dataset
print 'There are {} datapoints in dataset'.format(len(name))
# Features in data
print 'There are {} features in dataset'.format(len(features_unique))
#print features_unique

features_unique.remove('email_address')
enron_data= featureFormat(data_dict,features_unique,remove_NaN=False)
enron_data_df = pd.DataFrame(data=enron_data,columns=features_unique,index=name)
enron_data_df=enron_data_df.fillna(0)
#enron_data_df.loc[(enron_data_df==0).all(axis=1)]
enron_data_df=enron_data_df.drop(['TOTAL','THE TRAVEL AGENCY IN THE PARK','LOCKHART EUGENE E'])
#Creating new features
enron_data_df['from_this_person_to_poi_ratio']= enron_data_df['from_this_person_to_poi']/enron_data_df['from_messages']
enron_data_df['from_poi_to_this_person_ratio']=enron_data_df['from_poi_to_this_person']/enron_data_df['to_messages']
enron_data_df['poi_message_interaction']=enron_data_df['from_this_person_to_poi_ratio']+enron_data_df['from_poi_to_this_person_ratio']

enron_data_df['bonus_Salary_Incentives']=enron_data_df['bonus']+enron_data_df['salary']+enron_data_df['long_term_incentive']
enron_data_df['income_ratio']=enron_data_df['bonus_Salary_Incentives']/enron_data_df['total_payments']
#enron_data_df.head()
#It was found some data points were NaN for new features. If we groupby poi, mean of these have a significant difference
#Replacing NaN value using groupby mean of poi
enron_data_df['from_this_person_to_poi_ratio'].fillna(enron_data_df.groupby('poi')['from_this_person_to_poi_ratio'].transform("mean"),inplace=True)
enron_data_df['from_poi_to_this_person_ratio'].fillna(enron_data_df.groupby('poi')['from_poi_to_this_person_ratio'].transform("mean"),inplace=True)
enron_data_df['poi_message_interaction'].fillna(enron_data_df.groupby('poi')['poi_message_interaction'].transform("mean"),inplace=True)
enron_data_df['income_ratio'].fillna(enron_data_df.groupby('poi')['income_ratio'].transform("mean"),inplace=True)
cols = list(enron_data_df.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('poi')) #Remove b from list
#print cols
#cols.pop(cols.index('x')) #Remove x from list
enron_data_df= enron_data_df[['poi']+cols] #Create new dataframe with columns in the order you want

#labels=enron_data_df['poi']
#features=enron_data_df.drop('poi', axis=1)

# Converting the above modified dataframe to a dictionary
enron_dict = enron_data_df.to_dict('index')

### Store to my_dataset for easy export below.

Dataset= enron_dict
my_feature_list=list(enron_data_df.columns)
### Extract features and labels from dataset for local testing
data = featureFormat(Dataset, my_feature_list, sort_keys = False)
labels, features = targetFeatureSplit(data)

### split data into training and testing datasets
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2,  random_state=42)

# Stratified ShuffleSplit cross-validator
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2,random_state = 42)

# Importing modules for feature scaling and selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# Defining functions to be used via the pipeline
scaler = MinMaxScaler()
skb = SelectKBest(f_classif)

from sklearn.naive_bayes import GaussianNB
clf_gnb = GaussianNB()

pipeline = Pipeline(steps = [("SKB", skb), ("NaiveBayes",clf_gnb)])
param_grid = {"SKB__k":[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]}

from time import time
grid = GridSearchCV(pipeline, param_grid, verbose = 0, cv = sss, scoring = 'f1')

t0 = time()
grid.fit(features, labels)
print "training time: ", round(time()-t0, 3), "s"

print grid.best_estimator_

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
clf = grid.best_estimator_

t0 = time()
# refit the best GausianNB algorithm:
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
print "testing time: ", round(time()-t0, 3), "s"

print "Accuracy of GaussianNB classifer is  : ",accuracy_score(labels_test, prediction)
print "Precision of GaussianNB classifer is : ",precision_score(prediction, labels_test)
print "Recall of GaussianNB classifer is    : ",recall_score(prediction, labels_test)
print "f1-score of GaussianNB classifer is  : ",f1_score(prediction, labels_test)


from sklearn.tree import DecisionTreeClassifier
clf_tree=DecisionTreeClassifier()
pipeline_2 = Pipeline(steps = [("SKB", skb),("Dec_tree",clf_tree)])
param_grid = {"SKB__k":[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
             "Dec_tree__max_depth":[1,2,3,4,5,6,7,8,9,10],
             "Dec_tree__min_samples_leaf":[1,2,3,4,5],
             "Dec_tree__criterion":['entropy','gini']}

grid_2 = GridSearchCV(pipeline_2, param_grid, verbose = 0, cv = sss, scoring = 'f1')

t0 = time()
grid_2.fit(features, labels)
print "training time: ", round(time()-t0, 3), "s"

# best algorithm
clf = grid_2.best_estimator_
print clf
t0 = time()
# refit the best algorithm:
clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
print "testing time: ", round(time()-t0, 3), "s"

print "Accuracy of DecisionTreeclassifer is  : ",accuracy_score(labels_test, prediction)
print "Precision of DecisionTreeclassifer is : ",precision_score(prediction, labels_test)
print "Recall of DecisionTreeclassifer is    : ",recall_score(prediction, labels_test)
print "f1-score of DecisionTreeclassifer is  : ",f1_score(prediction, labels_test)

dump_classifier_and_data(clf, Dataset, my_feature_list)
