from sklearn import svm
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn import model_selection



# Read the Data

data=pd.read_csv("pima-indians-diabetes.csv",header=None)


# Separate predictors and output feature

X=data.iloc[:,0:8]
Y=data.iloc[:,8]

Accuracy=[]
for i in range(10):
    X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.20)
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    Accuracy.append(accuracy_score(Y_test,Y_pred))
print("The  Average accuracy after 10 randon train-test splits (Support Vecotor Machines)",sum(Accuracy)/len(Accuracy)*100,"%",sep=":")
    



