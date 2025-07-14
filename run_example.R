#run example, centroid strategy
#iris
iris_label=iris[,5]
dm_iris=rspatial_dp(iris[,-5])
ILD_mat_iris<-integrated_ld(iris[,-5],dm_iris$dm,Lmatrix = dm_iris$Lmatrix)
set.seed(2025)
iris_results=AUTO_DLCC(ILD_info = ILD_mat_iris,data=iris[,-5], dm0=dm_iris$dm,class_method = 'rf')
cluster_performance(iris_results$cluster_result$cluster_vector,iris_label)

#high dimensional, large sample example
library(tensorflow)
library(keras)
dm_yale=rspatial_dp(yale$x,gpu=T)
ILD_mat_yale<-integrated_ld(yale$x,dm_yale$dm,Lmatrix = dm_yale$Lmatrix)
yale_results=AUTO_DLCC(ILD_info = ILD_mat_yale, data=yale$x,dm0=dm_yale$dm,class_method = 'rf')
cluster_performance(yale_results$cluster_result$cluster_vector,yale$c)


#run example, linkage strategy

library(R.matlab)
path<-getwd()
pathname<-file.path(path,'ba.mat')
databa<-readMat(pathname)
ba_label=databa$ba.label
databa=databa$ba
#run
dm_ba=rspatial_dp(databa)
ILD_mat_ba<-integrated_ld(databa,dm_ba$dm,Lmatrix=dm_ba$Lmatrix)
ba_results<-AUTO_DLCC(ILD_info = ILD_mat_ba,data=databa,dm0=dm_ba$dm)
cluster_performance(ba_results$cluster_result$cluster_vector,ba_label)#100%



#some UCL data:
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
parkinsons <- read.csv(url)
pa_data=scale(parkinsons[,-c(1,18)])
pa_label=parkinsons$status
#bc
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
data_bc <- read.csv(url, header=FALSE)
bc_label=data_bc$V2
data_bc=data_bc%>%dplyr::select(-c('V1','V2'))
data_bc=scale(data_bc)

#run_sxample

dm_bc=rspatial_dp(data_bc)
ILD_mat_bc<-integrated_ld(data_bc,dm_bc$dm,Lmatrix = dm_bc$Lmatrix)
set.seed(2025)
bc_results=AUTO_DLCC(ILD_info = ILD_mat_bc,data=data_bc, dm0=dm_bc$dm,class_method = 'knn')
cluster_performance(bc_results$cluster_result$cluster_vector,bc_label)


#run other data in the similar way, see readme for more data sources
