setwd ("D:/zhang/paper_RMST/RMST_github")
library(doParallel)
library(foreach)
library(future)
library(Icens)
library(bayesSurv)
library(creditmodel)
library(reticulate)
library(MASS)
use_python("C:/ProgramData/Anaconda3/python.exe",required = T)
py_config()
py_available()
py_module_available("tensorflow")
source_python("train.py")


cores <- detectCores(logical=F)
availableCores()
cl <- makeCluster(cores)
registerDoParallel(7, cores=cores)


set.seed(28)
n=200
p=50

TrueTime<-matrix(0,1,n)
sf<-matrix(0,1,n)
timesup=100
MAE_<-matrix(0,timesup,1)
MSE_<-matrix(0,timesup,1)

tau=c(1)
S_area=matrix(0,n,length(tau))
xx=matrix(0,n,length(tau))
probe=matrix(0,n,length(tau))

############

compute_area<- function(tau,smat,interv){
  index=c(tau>interv[,2])
  nb_index=sum(index)
  cuts0=c(0,interv[,2])
  if (nb_index==0){
    cuts_tau=c(0,tau)
    RM=sum(interv[1,3]*diff(cuts_tau))
  } else if (nb_index==nrow(interv)){
    RM=sum(interv[1:(nb_index),3]*diff(cuts0))
  }else{
    cuts_tau=c(cuts0[1:(nb_index+1)],tau)
    RM=sum(interv[1:(nb_index+1),3]*diff(cuts_tau))
  }
  
  #RMi
  RMi=matrix(0,n,1)
  prob=matrix(0,n,1)
  # tau=3
  for (i in 1:n){
    indexi=c(tau>smat[i,,2]) 
    nb_indexi=sum(indexi)-sum(smat[i,,2]==0)
    cutsi0=c(0,smat[i,,2])
    
    if (nb_indexi==0){
      cutsi_tau=c(0,tau)
      RMi[i]=sum(smat[i,1,3]*diff(cutsi_tau))
    }else if(nb_indexi==nrow(interv)){
      RMi[i]=sum(smat[i,1:nb_indexi,3]*diff(cuts0))
      prob[i]<-smat[i,(nb_indexi),3]
    } else {
      cutsi_tau=c(cutsi0[1:(nb_indexi+1)],tau)
      RMi[i]=sum(smat[i,1:(nb_indexi+1),3]*diff(cutsi_tau))
      prob[i]<-smat[i,(nb_indexi+1),3]
    }
    
    
  }
  pseudo_mean=n*RM-(n-1)*RMi
  ret<-list(pseudo_mean=pseudo_mean,RMi=RMi,prob=prob)
  return(ret)
}


###################simulation#######################
for (times in 1:timesup){
  
  print("this is times=")
  print(times)
  Sigma<-matrix(0,p,p)#[1:p,1:p]
  for(i in 1:p)
  { 
    for(j in 1:p)
    {
      Sigma[i,j]<-0.5^(abs(j-i))
    }
  }
  X<-mvrnorm(n,rep(0,p),Sigma)#[300,10]
  lambda=abs(X[,1])+(X[,2]-0.5)^2+sqrt(abs(X[,3]-X[,4]))
  
  S_area[1:n,]=sapply(1:n,function(i)(-1/lambda[i]*exp(-lambda[i]*tau))+1/lambda[i])
  TrueTime[1:n]=sapply(1:n,function(i)(-1/lambda[i]*log(runif(1,min = 0,max=1))))
  
  Right<-rep(Inf,n)
  eps=1e-8
  nb.visit=10
  visTime=0;visit=matrix(0,n,nb.visit+1)
  visit=cbind(visit,rep(Inf,n))
  visit[,2]=visit[,1]+runif(n,0,2)
  schedule=1
  for (i in 3:(nb.visit+1))
  {
    visit[,i]=visit[,i-1]+runif(n,0,schedule*1)
  }
  Left<-visit[,(nb.visit+1)]
  J=sapply(1:(n),function(i)cut(TrueTime[i],breaks=c(visit[1:(n),][i,]),
                                labels=1:(nb.visit+1),right=FALSE)) #sum(is.na(J)) check!
  Left[1:(n)]=sapply(1:(n),function(i)visit[1:(n),][i,J[i]])
  Left[1:(n)]=sapply(1:(n),function(i) ifelse(Left[i]==0,TrueTime[i]-eps,Left[i]))
  Right[1:(n)]=sapply(1:(n),function(i)visit[1:(n),][i,as.numeric(J[i])+1])
  
  csub1 <- cbind(Left,Right)
  n=nrow(csub1)
  e4 <- EMICM(csub1)

  interv<-cbind(MLEintvl(csub1),1-e4$sigma)
  
  intPFS <- (1-cumsum(e4$pf))
  surv=c(1-e4$sigma,0)
  ttmp=c(0,interv[,2])
  smat=array(0, dim =c(n,nrow(interv),3), dimnames = NULL)
  intervi=array(0, dim =c(n,nrow(interv)), dimnames = NULL)
  

  res <- foreach(i=1:n, .combine=rbind, .packages=c('Icens'),.inorder=TRUE) %dopar% {
    m=csub1[-i,]
    ei <- EMICM(m)
    intervi<-cbind(i,MLEintvl(m),1-ei$sigma)#L,R,Probability
    return(intervi)
    
  }
  for (i in 1:n){
    smat[i,1:length(which(res[,1]==i)),1:3]<-res[which(res[,1]==i),2:4]
  }

  
  for(j in 1:length(tau)){
    ret=compute_area(tau[j],smat,interv)
    xx[,j]=ret$pseudo_mean
    probe[,j]=ret$prob
  }
  

  S<-Gsimulation(X_ori=X, xx=xx,S_area=S_area,prob=probe,tau=tau)
  MAE_[times]<-S$MAE
  MSE_[times]<-S$MSE

}

mean(MAE_)
mean(MSE_)
data_final<-data.frame(MAE_,MSE_)

stopImplicitCluster()
stopCluster(cl)
