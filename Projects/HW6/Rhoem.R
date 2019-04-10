library(MASS)
housing <- read.table("~/Desktop/housing.data.txt", quote="\"", comment.char="")
View(housing)
length(housing)
dim(housing)
attach(housing)
summary(housing)
table(is.na(housing))

# Applyting linear Regression

linear_model<-lm(V14~V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13,data=housing)
plot(linear_model)

# Plot standardresidual vs Fitted values
st=rstandard(linear_model)
plot(fitted(linear_model),st,xlab = "Fitted_values",ylab = "standardised Residual")


outlier_1<-housing[c(369,373,365), ]

# Removed first set of outliers


myData<-housing[-c(369,373,365), ] 
linear_model_1<-lm(myData$V14~myData$V1+myData$V2+myData$V3+myData$V4+myData$V5+myData$V6+myData$V7+myData$V8+myData$V9+myData$V10+myData$V11+myData$V12+myData$V13,data=myData)
plot(linear_model_1)

outlier_2<-myData[c(366,370,368), ]

# Removed 2nd set of outliers

myData_1<-myData[-c(366,370,368), ] 
linear_model_2<-lm(V14~V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13,data=myData_1)
plot(linear_model_2)


outlier_3<-myData_1[c(371,368,366), ]

# Removed 3rd set of outliers

myData_2<-myData_1[-c(371,368,366), ] 
linear_model_3<-lm(V14~V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13,data=myData_2)
plot(linear_model_3)


# combining all the outliers

outlier<-rbind(outlier_1,outlier_2,outlier_3)
rownames(outlier) <- NULL

# Box Cox Transformation after removing outliers

bc=boxcox(linear_model_3,lambda=seq(-3,3))

# maximum value of lambda where log likelihood is maximum

b_lam=bc$x[bc$y==max(bc$y)]



#After box transformation apply linear regression

bx<-lm((myData_2$V14)^b_lam ~ myData_2$V1+myData_2$V2+myData_2$V3+myData_2$V4+myData_2$V5+myData_2$V6+myData_2$V7+myData_2$V8+myData_2$V9+myData_2$V10+myData_2$V11+myData_2$V12+myData_2$V13,data=myData_2)

#standard residual values vs fitted values after transformation

st=rstandard(bx)
plot(fitted(bx)^(1/(b_lam)),st,xlab = "Fitted_values",ylab = "standardised Residual")


# Plot True price vs Predicted price

plot(myData_2$V14,(fitted(bx))^(1/(b_lam)),xlab="True Price",ylab = "Predicted Price",main="Housing Dataset")



#Points (outliers)refering to the original Dataset
k<-c()
for (i in 1:dim(outlier)[1]) {  
  for (j in 1:dim(housing)[1]) {
    if (sum(outlier[i,]==housing[j,])==14) {
           k<-c(k,j)
      }
    }
}





