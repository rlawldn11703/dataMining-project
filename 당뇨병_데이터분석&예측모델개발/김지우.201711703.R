### library setting
library(glmnet)
library(class)
library(naivebayes)
library(spatstat)
library(ROSE) #roc.curve
library(nnet)
library(spatstat)
library(ncpen)
library(class)
library(naivebayes)
library(rpart)
library(rpart.plot)
library(e1071)
library(MASS)
library(randomForest)
library(gbm)
library(caTools)

rm(list=ls())
setwd("C:\\Users\\사용자\\Desktop\\김지우\\건국대학교\\2020 1학기\\데이터마이닝\\프로젝트\\")
source("data.mining.functions.R")
xy.df=read.csv("김지우.201711703.diabetes.csv")
head(xy.df)
summary(xy.df)

ggplot(aes(x=SkinThickness),data=xy.df)+geom_histogram(binwidth=1.5,color='darkblue',fill='blue')+xlab("SkinThickness")+ylab("Frequency")
ggplot(aes(x=Insulin),data=xy.df)+geom_histogram(binwidth=1.5,color='darkred',fill='red')+xlab("Insulin")+ylab("Frequency")


xy.df$SkinThickness=ifelse(xy.df$SkinThickness==0,mean(xy.df$SkinThickness),xy.df$SkinThickness)
xy.df$Insulin=ifelse(xy.df$Insulin==0,mean(xy.df$Insulin),xy.df$Insulin)

### 나이와 임신 횟수의 상관관계 상당히 높음.
par(mfrow=c(1,1))
library(corrplot)
corrplot(cor(xy.df),method="number",type="lower")
plot(xy.df$Age,xy.df$Pregnancies)

### 나이가 20~40대에 몰려있음.나이를 범주형 변수로 바꾸는 작업 진행
library(ggplot2)
ggplot(aes(x=Age),data=xy.df)+geom_histogram(binwidth=1,color='darkblue',fill='blue')+xlab("Age")+ylab("Frequency")

xy.df$Age=ifelse(xy.df$Age<21,"21세 미만",
                 ifelse((xy.df$Age>=21)&(xy.df$Age<=30),"21-30세",
                 ifelse((xy.df$Age>=31)&(xy.df$Age<=40),"31-40세",
                 ifelse((xy.df$Age>=41)&(xy.df$Age<=50),"41-50세",
                 ifelse((xy.df$Age>=51)&(xy.df$Age<=60),"51-60세",
                 "61세 이상")))))
head(xy.df)                      
###train-test data split
set.seed(1234)
x.mat=as.matrix(cbind(xy.df[,c(1:7)],dummify(xy.df[,8])[,-5]))
y.mat=as.matrix(dummify(xy.df[,9])) #outcome변수 dummify
xy.df=data.frame(cbind(y.mat,x.mat))

sample=sample.split(xy.df$V1,SplitRatio=0.7)
txy.df=subset(xy.df,sample==TRUE)
tst_xy.df=subset(xy.df,sample==FALSE)

### data
x.mat=as.matrix(xy.df[,-1])
y.mat=as.matrix(dummify(xy.df[,1])) #outcome변수 dummify
y.vec=as.vector(dummify(xy.df[,1])[,1])
xy.df=data.frame(cbind(y.mat,x.mat))
tx.mat=as.matrix(txy.df[,-1])
ty.mat=as.matrix(dummify(txy.df[,1])) #outcome변수 dummify
ty.vec=as.vector(dummify(txy.df[,1])[,1])
txy.df=data.frame(cbind(ty.mat,tx.mat))
nx.mat=as.matrix(tst_xy.df[,-1])
ny.mat=as.matrix(dummify(tst_xy.df[,1])) #outcome변수 dummify
ny.vec=as.vector(dummify(tst_xy.df[,1])[,1])
nxy.df=data.frame(cbind(ny.mat,nx.mat))

###50 random test errors with cross validation
set.seed(1234)
s.num=50
r.mat=rand.index.fun(y.vec,s.num=s.num)
mod=c("ridge","lasso","scad","mbridge","knn","nbayes","tree","LDA","boosting","svm")
e.mat=matrix(NA,s.num,length(mod))
colnames(e.mat)=mod
for (s.id in 1:s.num){
  print(s.id)
  set=r.mat[,s.id]
  txy.df=xy.df[set,]
  nxy.df=xy.df[!set,]
  tx.mat=x.mat[set,]
  nx.mat=x.mat[!set,]
  ty.vec=y.vec[set]
  ny.vec=y.vec[!set]
  ##### lasso
  cv.fit=cv.glmnet(x=tx.mat,y=ty.vec,family="binomial",alpha=1,type.measure="class")
  fit=cv.fit$glmnet.fit
  opt=which.min(cv.fit$cvm)
  tab=table(ny.vec,ifelse(predict(fit,newx=nx.mat,s=cv.fit$lambda[opt])>0,1,0))
  e.mat[s.id,"lasso"]=1-sum(diag(tab))/sum(tab)
  ##### ridge
  cv.fit=cv.glmnet(x=tx.mat,y=ty.vec,family="binomial",alpha=0,type.measure ="class")
  fit=cv.fit$glmnet.fit
  opt=which.min(cv.fit$cvm)
  tab=table(ny.vec,ifelse(predict(fit,newx=nx.mat,s=cv.fit$lambda[opt])>0,1,0))
  e.mat[s.id,"ridge"]=1-sum(diag(tab))/sum(tab)
  ##### scad
  cv.fit=cv.ncpen(ty.vec,tx.mat,family="binomial",penalty="scad")
  fit=cv.fit$ncpen.fit
  opt=which.min(cv.fit$like)
  tab=table(ny.vec,ifelse(cbind(1,nx.mat)%*%coef(fit)[,opt]>0,1,0))
  e.mat[s.id,"scad"]=1-sum(diag(tab))/sum(tab)
  ##### mbridge
  cv.fit=cv.ncpen(ty.vec,tx.mat,family="binomial",penalty="mbridge")
  fit=cv.fit$ncpen.fit
  tab=table(ny.vec,ifelse(cbind(1,nx.mat)%*%coef(fit)[,opt]>0,1,0))
  e.mat[s.id,"mbridge"]=1-sum(diag(tab))/sum(tab)
  ##### knn
  c.vec=rep(NA,50)
  for(k in 1:50){
    cv.fit=knn.cv(train=tx.mat,cl=ty.vec,k=k)
    tab=table(ty.vec,cv.fit)
    c.vec[k]=1-sum(diag(tab))/sum(tab)
  }
  fit=knn(train=tx.mat,test=nx.mat,cl=ty.vec,k=which.min(c.vec))
  tab=table(ny.vec,fit)
  e.mat[s.id,"knn"]=1-sum(diag(tab))/sum(tab)
  ##### naivebayes
  fit=naive_bayes(x=txy.df[,-1],y=as.logical(txy.df[,1]))
  tab=table(nxy.df[,1],predict(fit,newdata=nxy.df[,-1]))
  e.mat[s.id,"nbayes"]=1-sum(diag(tab))/sum(tab)
  ##### tree
  fit=rpart(V1~.,data=txy.df,method="class")
  rpart.plot(fit)
  tab=table(nxy.df[,1],predict(fit,newdata=nxy.df[,-1],type="class"))
  e.mat[s.id,"tree"]=1-sum(diag(tab))/sum(tab)
  ##### LDA
  fit=lda(V1~.,data=txy.df)
  tab=table(nxy.df[,1],predict(fit,newdata=nxy.df)$class)
  e.mat[s.id,"LDA"]=1-sum(diag(tab))/sum(tab)
  ##### boosting
  fit=gbm(V1~.,data=txy.df,distribution = "adaboost",n.trees=100,cv.fold=5)
  opt=gbm.perf(fit,method="cv",plot.it=TRUE)
  tab=table(nxy.df[,1],predict(fit,newdata=nxy.df[,-1],n.trees=opt)>0)
  e.mat[s.id,"boosting"]=1-sum(diag(tab))/sum(tab)
  ##### SVM
  cv.fit=tune.svm(as.logical(V1)~.,data=txy.df,gamma=2^c(-4,-2,0,2),cost=2^c(-2,0,2,4),type="C")
  opt=cv.fit$best.parameters
  fit=svm(V1~.,data=nxy.df,type="C",gamma=opt[1],cost=opt[2])
  tab=table(nxy.df[,1],predict(fit,newdata=nxy.df))
  e.mat[s.id,"svm"]=1-sum(diag(tab))/sum(tab)
  
}


boxplot(e.mat)
ourmodel=which.min(colMeans(e.mat))

### 1.svm 방법론
mod=c("radial","sigmoid")
svm.eval=matrix(NA,1,length(mod))
colnames(svm.eval)=mod
###svm에서 kernerl trick(sigmoid,polynoimial 이용) default="radial"
cv.fit=tune.svm(as.logical(V1)~.,data=txy.df,gamma=2^c(-5:5),cost=2^c(-4:4),type="C")
opt=cv.fit$best.parameters
fit=svm(V1~.,data=nxy.df,type="C",gamma=opt[1],cost=opt[2])
tab=table(nxy.df[,1],predict(fit,newdata=nxy.df))
svm.eval["radial"]=1-sum(diag(tab))/sum(tab)

###kernel="sigmoid"
cv.fit=tune.svm(as.logical(V1)~.,data=txy.df,kernel="sigmoid",gamma=2^c(-5:5),cost=2^c(-4:4),type="C")
opt=cv.fit$best.parameters
fit=svm(V1~.,data=nxy.df,kernel="sigmoid",type="C",gamma=opt[1],cost=opt[2])
tab=table(nxy.df[,1],predict(fit,newdata=nxy.df))
svm.eval["sigmoid"]=1-sum(diag(tab))/sum(tab)
sen=tab[2,2]/sum(tab[,2])

####2.lasso로 방법론
m.vec=c("lasso")
m.vec=c(paste("err-",m.vec,sep=""),paste("dev-",m.vec,sep=""))
b.mat=matrix(0,nrow=ncol(x.mat)+1,ncol=length(m.vec))
colnames(b.mat)=m.vec
###type.measure="deviance"
cv.fit=cv.glmnet(x=tx.mat,y=ty.vec,family="binomial",alpha=1,type.measure="deviance")
fit=cv.fit$glmnet.fit
opt=which.min(cv.fit$cvm)
b.mat[,"dev-lasso"]=coef(cv.fit$glmnet.fit)[,opt]
cv.fit$lambda[opt]
sen=ass[1,"sen"]/(ass[1,"n1"]+ass[1,"n0"])
acc=ass[1,"acc"]/(ass[1,"n1"]+ass[1,"n0"])
###type.measure="class"
cv.fit=cv.glmnet(x=tx.mat,y=ty.vec,family="binomial",alpha=1,type.measure="class")
fit=cv.fit$glmnet.fit
opt=which.min(cv.fit$cvm)
b.mat[,"err-lasso"]=coef(cv.fit$glmnet.fit)[,opt]
cv.fit$lambda[opt]


###final assessment based on new samples
ass=glm.ass.fun(ny.vec,nx.mat,b.mat,mod="binomial")$ass
acc=ass[,"acc"]/nrow(nx.mat)
auc=ass[,"auc"]
m.vec[which.max(ass[,"acc"])]
m.vec[which.max(ass[,"sen"])]

##############################최종 lasso를 가지고 만든 모형
fit=glmnet(x=x.mat,y=y.vec,family="binomial",alpha=1,type.measure="deviance")

####3.ridge로 방법론
m.vec=c("ridge")
m.vec=c(paste("err-",m.vec,sep=""),paste("dev-",m.vec,sep=""))
b.mat=matrix(0,nrow=ncol(x.mat)+1,ncol=length(m.vec))
colnames(b.mat)=m.vec
###type.measure="deviance"
cv.fit=cv.glmnet(x=tx.mat,y=ty.vec,family="binomial",alpha=0,type.measure="deviance")
fit=cv.fit$glmnet.fit
opt=which.min(cv.fit$cvm)
cv.fit$lambda[opt]
b.mat[,"dev-ridge"]=coef(cv.fit$glmnet.fit)[,opt]
sen=ass[1,"sen"]/(ass[1,"n1"]+ass[1,"n0"])
acc=ass[1,"acc"]/(ass[1,"n1"]+ass[1,"n0"])
###type.measure="class"
cv.fit=cv.glmnet(x=tx.mat,y=ty.vec,family="binomial",alpha=0,type.measure="class")
fit=cv.fit$glmnet.fit
opt=which.min(cv.fit$cvm)
cv.fit$lambda[opt]
b.mat[,"err-ridge"]=coef(cv.fit$glmnet.fit)[,opt]

###final assessment based on new samples
ass=glm.ass.fun(ny.vec,nx.mat,b.mat,mod="binomial")$ass
acc=ass[,"acc"]/nrow(nx.mat)
auc=ass[,"auc"]
m.vec[which.max(ass[,"acc"])]
m.vec[which.max(ass[,"sen"])]

##########최종모형
mod=c("svm","lasso","ridge")
eval=matrix(NA,2,length(mod))
colnames(eval)=mod
###svm
fit=svm(V1~.,data=xy.df,kernel="radial",type="C",gamma=0.03125,cost=2)
tab=table(xy.df[,1],predict(fit,newdata=xy.df))
eval[1,"svm"]=sum(diag(tab))/sum(tab)
eval[2,"svm"]=tab[2,2]/sum(tab[,2])
###ridge
fit=glmnet(x=x.mat,y=y.vec,lambda=0.01004344,family="binomial",alpha=0)
tab=table(y.vec,ifelse(predict(fit,newx=x.mat,s=0.01004344)>0,1,0))
eval[1,"ridge"]=sum(diag(tab))/sum(tab)
eval[2,"ridge"]=tab[2,2]/sum(tab[,2])
###lasso
fit=glmnet(x=x.mat,y=y.vec,lambda=0.02163793,family="binomial",alpha=1)
tab=table(y.vec,ifelse(predict(fit,newx=x.mat,s=0.02163793)>0,1,0))
eval[1,"lasso"]=sum(diag(tab))/sum(tab)
eval[2,"lasso"]=tab[2,2]/sum(tab[,2])
eval
