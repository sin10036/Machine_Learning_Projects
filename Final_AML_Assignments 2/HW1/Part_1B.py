#Importing Libraries

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn import model_selection
from sklearn import svm
from sklearn.model_selection import GridSearchCV


# Read the Data

data=pd.read_csv("pima-indians-diabetes.csv",header=None)


# Separate predictors and output feature

X=data.iloc[:,0:8]
Y=data.iloc[:,8]

# Naive Bayes Classifier for fitting the data

def fit(X_train, Y_train):
    result = {}
    result["total_data"] = len(Y_train)
    class_values = set(Y_train)
    for current_class in class_values:
        result[current_class] = {}
        
        current_class_rows = (Y_train == current_class)
        X_train_current = X_train[current_class_rows]
        Y_train_current = Y_train[current_class_rows]
        num_features = X_train.shape[1]
        result[current_class]["total_count"] = len(Y_train_current)
        for j in range(1, num_features + 1):
            result[current_class][j] = {}
            all_possible_values = ['Mean',]
            for current_value in all_possible_values:
                if(j==3 or j==4 or j==6 or j==8):
                    result[current_class][j][current_value] = X_train_current.iloc[:,j-1][X_train_current.iloc[:,j-1]!=0].mean()
                    
                else:
                    result[current_class][j][current_value] = X_train_current.iloc[:,j-1].mean()
                all_possible_values = ['standard_deviation']
                for current_value in all_possible_values:
                       if(j==3 or j==4 or j==6 or j==8):
                              result[current_class][j][current_value] = X_train_current.iloc[:,j-1][X_train_current.iloc[:,j-1]!=0].std()
                       else:
                
                              result[current_class][j]['standard_deviation'] = X_train_current.iloc[:,j-1].std()
    return result



# Naive Bayes classifier for calculating Probability

def probability(dictionary, x, current_class):
    output = np.log(dictionary[current_class]["total_count"]) - np.log(dictionary["total_data"])
    num_features = len(dictionary[current_class].keys()) - 1;
    for j in range(1, num_features + 1):
        
        xj = x[j - 1]
        if(j==3 or j==4 or j==6 or j==8):
            if xj==0:
                current_xj_probablity=0
            else:
                first_term = 1/(np.sqrt(2*3.14*dictionary[current_class][j]['standard_deviation']))
                second_term = (((xj-dictionary[current_class][j]['Mean'])**2)/dictionary[current_class][j]['standard_deviation'])
                current_xj_probablity = np.log(first_term) - second_term
        else:
                first_term = 1/(np.sqrt(2*3.14*dictionary[current_class][j]['standard_deviation']))
                second_term = (((xj-dictionary[current_class][j]['Mean'])**2)/dictionary[current_class][j]['standard_deviation'])
                current_xj_probablity = np.log(first_term) - second_term
        
                
        
        output = output + current_xj_probablity
    return output

# Predict Single Point of the test data

def predictSinglePoint(dictionary, x):
    classes = dictionary.keys()
    best_p = -1000
    best_class = -1
    first_run = True
    for current_class in classes:
        if (current_class == "total_data"):
            continue
        p_current_class = probability(dictionary, x, current_class)
        if (first_run or p_current_class > best_p):
            best_p = p_current_class
            best_class = current_class
        first_run = False
    return best_class


# Predict the whole test data given as input


def predict(dictionary, X_test):
    y_pred = []
    for x in range(len(X_test)):
        
        x_class = predictSinglePoint(dictionary, X_test.iloc[x,:])
        y_pred.append(x_class)
    return y_pred



Accuracy=[]
for i in range(10):
    X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.20)
    dictionary = fit(X_train,Y_train)
    Y_pred = predict(dictionary,X_test)
    Accuracy.append(accuracy_score(Y_test,Y_pred))
print("The Average accuracy after handling missing data 10 randon train-test splits",sum(Accuracy)/len(Accuracy)*100,"%",sep=":")
    


