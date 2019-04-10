import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from numpy import array

import skimage.transform
from sklearn.ensemble import RandomForestClassifier


#Load Training dataset 
data=pd.read_csv("train.csv")

# Drop unnecessay column
data.drop([data.columns[0]],inplace=True,axis=1)


# Splitting data into training and test data set
Y=data.iloc[:,0].values
X=data.iloc[:,1:].values


# Create Stretched Bounding Box (20 x 20)



def bounding_box(k):
    
        #resize the image (28 x 28)
        
        k1=[]
        for i in range(len(k)):
            k1.append(k[i,:].reshape(28,28))
        k=array(k1)
    
    
        #find the box boundaries (upper/loer/left/right)
        
        f=np.zeros(shape=(k.shape[0],400))
        
        for j in range(len(k)):
            q=[]
            done1=False
            for c in range(k[j].shape[0]):
                for d in range(k[j].shape[1]):

                    if k[j][c,d]!=0:
                        rowup=c
                        done1=True
                        break
                if done1:
                    break


            done2=False    
            for c in range(k[j].shape[0]-1,-1,-1):
                for d in range(k[j].shape[1]-1,-1,-1):
                    if k[j][c,d]!=0:
                        rowd=c
                        done2=True
                        break

                if done2:
                    break

            done3=False
            for c in range(k[j].shape[0]):
                for d in range(k[j].shape[1]):
                    if k[j][d,c]!=0:
                        colleft=c
                        done3=True
                        break

                if done3:
                    break


            done4=False

            for c in range(k[j].shape[0]-1,-1,-1):
                for d in range(k[j].shape[0]-1,-1,-1):
                    if k[j][d,c]!=0:
                        collright=c
                        done4=True
                        break

                if done4:
                    break


            #Find the mean of the box

            col=(colleft+collright)//2
            row=(rowup+rowd)//2
            

            rowup=row-10
            rowd=row+9
            colleft=col-9
            collright=col+10


            #create (20 X 20)image  with that mean value


            v=np.zeros(shape=(rowd-rowup+1,collright-colleft+1))
            h=0
            
            if(rowup<0):
                  rowup=0
            if(rowd>27):
                  rowd=27

            if(colleft<0):
                  colleft=0
            if(collright>27):
                  collright=27





            for a in range(rowup,rowd+1):

                 for b in range(colleft,collright+1):

                      

                      q.append(k[j][a,b])

                 while(len(q)!=20):
                      q.append(0)

                 v[h]=q
                 q=[]
                 h=h+1



            r=[]
            done1=False
            for c in range(v.shape[0]):
                for d in range(v.shape[1]):

                    if v[c,d]!=0:
                        rowup=c
                        done1=True
                        break
                if done1:
                    break


            done2=False    
            for c in range(v.shape[0]-1,-1,-1):
                for d in range(v.shape[1]-1,-1,-1):
                    if v[c,d]!=0:
                        rowd=c
                        done2=True
                        break

                if done2:
                    break

            done3=False
            for c in range(v.shape[0]):
                for d in range(v.shape[1]):
                    if v[d,c]!=0:
                        colleft=c
                        done3=True
                        break

                if done3:
                    break


            done4=False

            for c in range(v.shape[0]-1,-1,-1):
                for d in range(v.shape[0]-1,-1,-1):
                    if v[d,c]!=0:
                        collright=c
                        done4=True
                        break

                if done4:
                    break


            w=np.zeros(shape=(rowd-rowup+1,collright-colleft+1))
            m=0

            for a in range(rowup,rowd+1):

                 for b in range(colleft,collright+1):
                      r.append(v[a,b])

                 w[m]=r
                 r=[]
                 m=m+1
            
            # resize the image (20 * 20)

            n=skimage.transform.resize(w,(20,20)).reshape(1,400)
            f[j]=n
            
        return f
    
    
    
# Validation set
dat_val=pd.read_csv("val.csv")
X_val=dat_val.iloc[:,1:].values
Y_val=dat_val.iloc[:,0].values


## Test Data
test=pd.read_csv("test.csv",header=None)
X_test=test.iloc[:,0:].values



#validation set Bounding Box


Stretched_bounding=bounding_box(X_val)

l=np.zeros(shape=(2000,400))
for j in range(l.shape[0]):
    for k in range(l.shape[1]):
            if(Stretched_bounding[j,k]>=120):
                l[j,k]=1
            else:
                l[j,k]=0
                
#validation untouched Bounding Box

untouched_validation=np.zeros(shape=(2000,784))
for j in range(X_val.shape[0]):
        for k in range(X_val.shape[1]):
            if(X[j,k]>=120):
                untouched_validation[j,k]=1
                
            else:
                untouched_validation[j,k]=0




                
# Training set Bounding Box
                
f=bounding_box(X)

l1=np.zeros(shape=(48000,400))
    
for j in range(f.shape[0]):
    for k in range(f.shape[1]):
        if(f[j,k]>=120):
            l1[j,k]=1
                
        else:
            l1[j,k]=0

                
                
#Untouched (Training Data)

untouched=np.zeros(shape=(48000,784))
for j in range(X.shape[0]):
        for k in range(X.shape[1]):
            if(X[j,k]>=120):
                untouched[j,k]=1
                
            else:
                untouched[j,k]=0


#Untouched (Test Data)

untouched_test=np.zeros(shape=(20000,784))
for j in range(X_test.shape[0]):
        for k in range(X_test.shape[1]):
            if(X[j,k]>=120):
                untouched_test[j,k]=1
                
            else:
                untouched_test[j,k]=0

            
# Testing set Bounding Box
                
f_test=bounding_box(X_test)

l1_test=np.zeros(shape=(20000,400))
    
for j in range(f_test.shape[0]):
    for k in range(f_test.shape[1]):
        if(f_test[j,k]>=120):
            l1_test[j,k]=1
                
        else:
            l1_test[j,k]=0
            
            
#Gaussian + untouched

clf = GaussianNB()
clf.fit(X,Y)
    
y_pred=clf.predict(X_val)
print("The accuracy on validation data (Gaussian + untouched) :",accuracy_score(Y_val,y_pred)*100,"%",sep="")

y_test=clf.predict(X_test)


df = pd.DataFrame(columns=['ImageId','Label'])
index=list(range(0,20000))

for i in range(len(y_test)):
    df.loc[i]=[index[i],y_test[i]]
    
    
df.to_csv("ts8_1",index=False)


# Ploting images after doing mean distribution of particular image in test data set

#class_means=pd.DataFrame(np.column_stack((X_test, y_test)))

#fig = plt.figure()
#for i in range(0,10):
#   plt.subplot(1,10, i+1)
#   plt.imshow(class_means.iloc[:,0:784][class_means.iloc[:,784]==i].mean().values.reshape(28,28),cmap='gray')
#   fig.subplots_adjust(right=6)

#plt.show()



#Gaussian + stretched
clf = GaussianNB()
clf.fit(l1,Y)
    
y_pred=clf.predict(l)
print("The accuracy on validation data (Gaussian + stretched) :",accuracy_score(Y_val,y_pred)*100,"%",sep="")


y_test=clf.predict(l1_test)


df = pd.DataFrame(columns=['ImageId','Label'])
index=list(range(0,20000))

for i in range(len(y_test)):
    df.loc[i]=[index[i],y_test[i]]
    
    
df.to_csv("ts8_2",index=False)


# Ploting images after doing mean distribution of particular image in test data set


#class_means=pd.DataFrame(np.column_stack((l1_test, y_test)))

#fig = plt.figure()
#for i in range(0,10):
#   plt.subplot(1,10, i+1)
#   plt.imshow(class_means.iloc[:,0:400][class_means.iloc[:,400]==i].mean().values.reshape(20,20),cmap='gray')
#   fig.subplots_adjust(right=6)
#plt.show()



# Bernoulli + untouched

clf = BernoulliNB()
clf.fit(X,Y)
    
y_pred=clf.predict(X_val)
print("The accuracy on validation data (Bernoulli + untouched) :",accuracy_score(Y_val,y_pred)*100,"%",sep="")




y_test=clf.predict(X_test)


df = pd.DataFrame(columns=['ImageId','Label'])
index=list(range(0,20000))

for i in range(len(y_test)):
    df.loc[i]=[index[i],y_test[i]]
    
    
df.to_csv("ts8_3",index=False)



# Ploting images after doing mean distribution of particular image in test data set
#class_means=pd.DataFrame(np.column_stack((X_test, y_test)))

#fig = plt.figure()
#for i in range(0,10):
#   plt.subplot(1,10, i+1)
#   plt.imshow(class_means.iloc[:,0:784][class_means.iloc[:,784]==i].mean().values.reshape(28,28),cmap='gray')
#   fig.subplots_adjust(right=6)
#plt.show()



#Bernoulli + stretched

clf = BernoulliNB()
clf.fit(l1,Y)
    
y_pred=clf.predict(l)
print("The accuracy on validation data (Bernoulli + stretched) :",accuracy_score(Y_val,y_pred)*100,"%",sep="")



y_test=clf.predict(l1_test)

df = pd.DataFrame(columns=['ImageId','Label'])
index=list(range(0,20000))

for i in range(len(y_test)):
    df.loc[i]=[index[i],y_test[i]]
    
    
df.to_csv("ts8_4",index=False)

# Ploting images after doing mean distribution of particular image in test data set
#class_means=pd.DataFrame(np.column_stack((l1_test, y_test)))

#fig = plt.figure()


#for i in range(0,10):

#   plt.subplot(1,10, i+1)  
#   plt.imshow(class_means.iloc[:,0:400][class_means.iloc[:,400]==i].mean().values.reshape(20,20),cmap='gray')
#   fig.subplots_adjust(right=6)
   
#plt.show()


#10 trees + 4 depth + untouched
clf = RandomForestClassifier(n_estimators=10,max_depth=4, random_state=0)
clf.fit(X,Y)
y_pred=clf.predict(X_val)
print("The accuracy on validation data (10 trees + 4 depth + untouched) :",accuracy_score(Y_val,y_pred)*100,"%",sep="")





y_test=clf.predict(X_test)


df = pd.DataFrame(columns=['ImageId','Label'])
index=list(range(0,20000))

for i in range(len(y_test)):
    df.loc[i]=[index[i],y_test[i]]
    
    
df.to_csv("ts8_5",index=False)



#10 trees + 4 depth + stretched

clf = RandomForestClassifier(n_estimators=10,max_depth=4, random_state=0)
clf.fit(l1,Y)
y_pred=clf.predict(l)
print("The accuracy on validation data (10 trees + 4 depth + stretched) :",accuracy_score(Y_val,y_pred)*100,"%",sep="")



y_test=clf.predict(l1_test)

df = pd.DataFrame(columns=['ImageId','Label'])
index=list(range(0,20000))

for i in range(len(y_test)):
    df.loc[i]=[index[i],y_test[i]]
    
    
df.to_csv("ts8_6",index=False)



#10 trees + 16 depth + untouched
clf = RandomForestClassifier(n_estimators=10,max_depth=16, random_state=0)
clf.fit(X,Y)
y_pred=clf.predict(X_val)
print("The accuracy on validation data (10 trees + 16 depth + untouched) :",accuracy_score(Y_val,y_pred)*100,"%",sep="")




y_test=clf.predict(X_test)


df = pd.DataFrame(columns=['ImageId','Label'])
index=list(range(0,20000))

for i in range(len(y_test)):
    df.loc[i]=[index[i],y_test[i]]
    
    
df.to_csv("ts8_7",index=False)



# 10 trees + 16 depth + stretched

clf = RandomForestClassifier(n_estimators=10,max_depth=16, random_state=0)
clf.fit(l1,Y)
y_pred=clf.predict(l)

print("The accuracy on validation data (10 trees + 16 depth + stretched) :",accuracy_score(Y_val,y_pred)*100,"%",sep="")




y_test=clf.predict(l1_test)

df = pd.DataFrame(columns=['ImageId','Label'])
index=list(range(0,20000))

for i in range(len(y_test)):
    df.loc[i]=[index[i],y_test[i]]
    
    
df.to_csv("ts8_8",index=False)



#30 trees + 4 depth + untouched

clf = RandomForestClassifier(n_estimators=30,max_depth=4, random_state=0)
clf.fit(X,Y)
y_pred=clf.predict(X_val)
print("The accuracy on validation data (30 trees + 4 depth + untouched) :",accuracy_score(Y_val,y_pred)*100,"%",sep="")




y_test=clf.predict(X_test)


df = pd.DataFrame(columns=['ImageId','Label'])
index=list(range(0,20000))

for i in range(len(y_test)):
    df.loc[i]=[index[i],y_test[i]]
    
    
df.to_csv("ts8_9",index=False)



#30 trees + 4 depth + stretched


clf = RandomForestClassifier(n_estimators=30,max_depth=4, random_state=0)
clf.fit(l1,Y)
y_pred=clf.predict(l)
print("The accuracy on validation data (30 trees + 4 depth + stretched) :",accuracy_score(Y_val,y_pred)*100,"%",sep="")



y_test=clf.predict(l1_test)

df = pd.DataFrame(columns=['ImageId','Label'])
index=list(range(0,20000))

for i in range(len(y_test)):
    df.loc[i]=[index[i],y_test[i]]
    
    
df.to_csv("ts8_10",index=False)





#30 trees + 16 depth + untouched

clf = RandomForestClassifier(n_estimators=30,max_depth=16, random_state=0)
clf.fit(X,Y)
y_pred=clf.predict(X_val)
print("The accuracy on validation data (30 trees + 16 depth + untouched) :",accuracy_score(Y_val,y_pred)*100,"%",sep="")




y_test=clf.predict(X_test)


df = pd.DataFrame(columns=['ImageId','Label'])
index=list(range(0,20000))

for i in range(len(y_test)):
    df.loc[i]=[index[i],y_test[i]]
    
    
df.to_csv("ts8_11",index=False)



#30 trees + 16 depth + stretched

clf = RandomForestClassifier(n_estimators=30,max_depth=16, random_state=0)
clf.fit(l1,Y)
y_pred=clf.predict(l)
print("The accuracy on validation data (30 trees + 16 depth + strtched) :",accuracy_score(Y_val,y_pred)*100,"%",sep="")




y_test=clf.predict(l1_test)

df = pd.DataFrame(columns=['ImageId','Label'])
index=list(range(0,20000))

for i in range(len(y_test)):
    df.loc[i]=[index[i],y_test[i]]
    
    
df.to_csv("ts8_12",index=False)














