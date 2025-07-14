path<-getwd()
pathname<-file.path(path,'frog_data.mat')
data_frog<-readMat(pathname)
frog_L=data_frog$frog.Lmatrix
dm_frog=data_frog$dm.frog
data_frog=data_frog$X

path<-getwd()
pathname<-file.path(path,'frog_label.mat')
frog_label<-readMat(pathname)
frog_label=frog_label$label

ILD_mat_frog<-integrated_ld(data_frog,dm_frog,Lmatrix = frog_L)
frog_result=AUTO_DLCC(ILD_info = ILD_mat_frog, data=data_frog,dm0=dm_frog,class_method = 'rf')

#step by step running

n=nrow(data)
d=ncol(data)
ILD_mat=ILD_info$ILD_mat
dm0.order=ILD_info$dm0.order
nbr_list=ILD_info$nbr_list
sym_dm=sym_mat(dm0)

size_info=find_peak_size(ILD_mat,dm0.order,nbr_list)

est_size=size_info$est_size
highest_ld_idx=size_info$highest_ld_idx
save_lc_contents=check_lc(est_size,highest_ld_idx,nbr_list,dm0.order)
save_lc=save_lc_contents$save_lc
freq_table=save_lc_contents$freq_table
matrix_info=sm_computer(save_lc,sym_dm)

frog_stable_centers_info=find_stable_centers_ld(ILD_mat = ILD_mat_frog$ILD_mat,ILD_mat_frog$nbr_list,matrix_info = matrix_info,sym_dm,dm0.order=dm0.order)
frog_temp_clus=get_temp_clus(sym_dm,frog_stable_centers_info$group_list,strategy =frog_stable_centers_info$strategy,data=data_frog,freq_table)
frog_result<-DAobs(data_frog,frog_temp_clus$temp_clus, method='knn',K_knn=7,depth='spatial',dm0=dm_frog,ILD_info = ILD_mat_frog)
tmp_frog1=cluster2cv(data_frog,frog_temp_clus$temp_clus) 

frog_stable_centers_info2=find_stable_centers_ld(ILD_mat = ILD_mat_frog$ILD_mat,ILD_mat_frog$nbr_list,matrix_info = matrix_info,sym_dm,strategy = 'linkage',dm0.order=dm0.order)
frog_temp_clus2=get_temp_clus(sym_dm,frog_stable_centers_info2$group_list,strategy =frog_stable_centers_info2$strategy,data=data_frog,freq_table)
set.seed(2025)
frog_result2<-DAobs(data_frog,frog_temp_clus2$temp_clus, method='knn',K_knn=5,depth='spatial',dm0=dm_frog,ILD_info = ILD_mat_frog)
cluster_performance(frog_result2$cluster_vector,frog_label[,3]+1)

tmp_frog2=cluster2cv(data_frog,frog_temp_clus2$temp_clus) 

cluster_performance(tmp_frog2[tmp_frog2!=0],frog_label[,3][tmp_frog2!=0])

cluster_performance(frog_result$cluster_vector,frog_label[,3])
table(frog_result2$cluster_vector,frog_label[,3])



true_lab=frog_label[,3]+1
true_lab[true_lab!=6]<-0
p3=color_plot(tsne_frogdf,true_lab)
library(cowplot)
cbp=combine_graph_generator(p1,p2,p3,label="Anuran")

set.seed(42)
tsne_frog<-Rtsne(data_frog,dims=2,preplexity=30,verbose=T)

tsne_frogdf<-data.frame(
  Dim1=tsne_frog$Y[,1],
  Dim1=tsne_frog$Y[,2]
)

p1=color_plot(tsne_frogdf,frog_result$cluster_vector,frog_temp_clus$stable_centers)$p1
p2=color_plot(tsne_frogdf,frog_result2$cluster_vector,local_center = frog_temp_clus2$stable_centers)
p3=
tc=frog_result$cluster_vector
tc[!tc%in%c(19)]<-0
color_plot(tsne_frogdf,tc)

merge_ids <- frog_result$cluster[c(1,19)]%>%unlist()
after_mer <- reachable_similarity(sym_dm[merge_ids, merge_ids])
mcs=compute_mcs(after_mer)
pos_idx=match(frog_result$cluster[[19]],merge_ids)
to_other<-after_mer[pos_idx,-pos_idx]%>%rowmeans()
delta[u]<- mean(to_other/reachable_list[[2]]) #0.9234
mean(to_other/reachable_list[[16]])
#0.9471371


reachable_list=lapply(frog_result$cluster,function(x){
  rsm=reachable_similarity(sym_dm[x,x])
  cs <- colSums(rsm)
  (cs-1)/(length(x)-1)
})

mR=R[R!=0]%>%mean
vm=as.matrix(value>mR)
rowSums(mat_div)/rowSums(R>0)

all_R=sum(R)
v1=sum(value[sym_sm<0.01])/all_R
v2=sum(value)/all_R
v1/v2
total=sum(R!=0)
cs=sum((sym_sm<0.01) & (R > 0))
compute_LNI2 <- function(sym_sm, R) {
  diag(R)<-0
  diag(sym_sm)<-0
  
  value=R-sym_sm
  prop=sum(value)/sum(R)
  return(list(value=value,prop=prop))
}
compute_LNI <- function(sym_sm, R) {
  diag(R)<-0
  D <- sym_sm
  th=D %>%rowMaxs(value=T)%>%quantile(0.05)
  
  mask_no_direct <- (D<th) & (R > 0)
  maskednd_R<-R*mask_no_direct
  LNI <- rowSums(maskednd_R) / rowSums(R)
  
  mask_direct <- sym_sm==R
  masked_R <- R * mask_direct 
  min_dl=apply(masked_R, 1, function(x) if(any(x != 0)) min(x[x != 0]) else 0)
  mat_div <- value/ min_dl   # R自动按行广播
  mat_div <- pmin(mat_div, 1)
  # 计算每行masked R值的和除以非零R的总和
  LNI <- rowSums(masked_R) / rowSums(R)
  LNI[is.na(LNI)] <- 0
  ret

library(Spectrum)
res<-Spectrum(t(data_frog)%>%as.data.frame())
cluster_performance(res$assignments,frog_label[,3]+1)#7
0.5899859 0.6472550 0.5500447
res2<-Spectrum(t(data_frog)%>%as.data.frame(),method=2)
cluster_performance(res2$assignments,frog_label[,3]+1)
