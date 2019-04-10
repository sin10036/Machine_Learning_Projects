#12.5

library('caTools')
library('glmnet')
Crusio1 <- read.csv("~/Downloads/Crusio1.csv")
View(Crusio1)

X<-Crusio1[,4:41]
Y<-as.data.frame(Crusio1[,2])

X_Y<-cbind(X,Y)
X_Y<-na.omit(X_Y)



glm<-cv.glmnet(as.matrix(X_Y[,1:38]),as.factor(as.numeric(X_Y[,39])),type.measure = "class",alpha=1,family='multinomial')
plot(glm)
#The value of lambda which gives the low classification error=0.01589668


#Baseleine Accuracy:50.88%
Baseline_Accuracy=max(table(X_Y[,39]))/dim(X_Y)[1]

#Prediction based on lambda.min
lmpredic<-predict(glm,as.matrix(X_Y[,1:38]),type='class',s='lambda.min')
numright<-sum(as.numeric(X_Y[,39])==lmpredic)
# The accuracy on lambda.min =86.53%
Accuracy_lambda_min<-numright/dim(lmpredic)[1]


#Prediction based on lambda.1sde
l1predn<-predict(glm,as.matrix(X_Y[,1:38]),type='class',s='lambda.1se')
n1umright<-sum(as.numeric(X_Y[,39])==l1predn)
# The accuracy on lambda.1se =85.65%
Accuracy_1sde<-n1umright/dim(l1predn)[1]



#B)

X<-Crusio1[,4:41]
Y<-as.data.frame(as.numeric(Crusio1[,1]))

X_Y<-cbind(X,Y)
X_Y<-na.omit(X_Y)
set.seed(123)

# Shuffling the data set
X_Y<-X_Y[sample(nrow(X_Y)),]


# Removing the classes(categories)  which have less than 10 rows
table(X_Y[,39])<10
#REMOVE (1,10,11,18,19,33,37,40,45,50)


tf=X_Y[,39]==1 | X_Y[,39]==10 | X_Y[,39]==11 | X_Y[,39]==18 | X_Y[,39]==19 | X_Y[,39]==33 | X_Y[,39]==37 | X_Y[,39]==40 | X_Y[,39]==45 | X_Y[,39]==50
X_Y<-subset(X_Y,tf==FALSE)



glm<-cv.glmnet(as.matrix(X_Y[,1:38]),as.factor(X_Y[,39]),type.measure = "class",alpha=1,family='multinomial')
#The minimum value of lambda which gives the low classification error=0.0002577546
plot(glm)


#Baseleine Accuracy:2.33%
Baseline_Accuracy=max(table(X_Y[,39]))/dim(X_Y)[1]


#
#Prediction based on lambda.min
lmpredic<-predict(glm,as.matrix(X_Y[,1:38]),type='class',s='lambda.min')
numright<-sum(X_Y[,39]==lmpredic)
# The accuracy on lambda.min =70.39%
Accuracy_lambda_min<-numright/dim(lmpredic)[1]


#Prediction based on lambda.1sde
l1predn<-predict(glm,as.matrix(X_Y[,1:38]),type='class',s='lambda.1se')
n1umright<-sum(X_Y[,39]==l1predn)
# The accuracy on lambda.1se =52.33%
Accuracy_1sde<-n1umright/dim(l1predn)[1]






