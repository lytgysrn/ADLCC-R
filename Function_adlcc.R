library(tidyverse)
library(ddalpha)
library(randomForest)
library(Rfast)
library(PPCI)
library(ggplot2)
library(pald)
library(igraph)
library(zoo)
library(pracma)
library(R.matlab)
library(Rtsne)

#spatial depth
spatial.d<-function(x,data,Lmatrix=c()){
  #x: whose depth is to be found
  #data: sample points of this distribution
  #Lmatrix: distance matrix between points, when provided, much faster
  #Based on Serfling
  if (is.data.frame(data)) {
    data = data.matrix(data)
  }
  if (is.matrix(x)==F) {
    if (is.vector(x)) 
      x <- matrix(x, nrow = 1)
    if (is.data.frame(x)) 
      x = data.matrix(x)
  }
  
  n<-nrow(data)
  d<-ncol(data)
  if (nrow(Lmatrix)%>%is.null()){
    N<-nrow(x)
    t.matrix<-diag(d)
    dep<-rep(0,N)
    for (i in 1:N) {
      a<-t(t.matrix%*%(x[i,]-t(data)))
      a<-a[rowSums(abs(a))!=0,]
      a<-sweep(a,1,sqrt(rowSums(a^2)),'/')
      e<-colSums(a)/n
      dep[i]<-1-sqrt(sum(e^2))
      # vardi's version
      # e<-colMeans(a)
      # if (nrow(a)!=n){
      #   dep[i]<-dep[i]+1/n
      # } 
    }
  } else {
    tol = 1e-5;
    index=Lmatrix<tol
    Lmatrix[index] <- Inf
    Lmatrix <- 1 / Lmatrix
    
    C <- t(data) %*% Lmatrix
    C <- t(C)
    
    norm_csum <- colSums(Lmatrix)
    
    C2 <- norm_csum * data
    C <- C2 - C
    dep <- 1 -  apply(C /n, 1, function(x) sqrt(sum(x^2)))
  }
  return(dep)
}
#Spatial-depth-based similarity matrix
rspatial_dp<-function(data, gpu=F) {
  #when gpu=T, tensorflow package is used for acceleration, but not recommend (DLCC codes in Matlab is much faster)
  data <- as.matrix(data)
  # build reflection spatial-depth based similarity matrix
  n <- nrow(data)
  d <- ncol(data)
  rn_inv <- 1/(2*n-1)
  dm <- matrix(0, n, n)
  
  tol <- 1e-5
  
  Ematrix <- matrix(0, n, d)
  Lmatrix <- matrix(0, n, n)
  
  for (i in 1:n) {
    a <- sweep(-data, 2, data[i,], FUN = "+")
    Lmatrix[,i] <- apply(a, 1, function(x) sqrt(sum(x^2))) 
    norm_a <- Lmatrix[,i]
    norm_a[norm_a < tol] <- 1
    a <- sweep(a, 1, norm_a, FUN = "/")
    Ematrix[i,] <- colSums(a)
  }
  Lmatrix_save=Lmatrix
  Lmatrix <- Lmatrix^2
  
  if (gpu){
    # Convert data to tensor
    data_tensor <- tf$constant(data, dtype = tf$float64)
    
    Lmatrix_tensor<-tf$constant(Lmatrix, dtype = tf$float64)
    two_Lmatrix=2*Lmatrix_tensor
    Ematrix_tensor <- tf$constant(Ematrix, dtype = tf$float64)
    
    for (j in 1:n) {
      idx <- as.integer(which(!(1:n == j))-1)  #python starts from 0
      idx_tensor <- tf$constant(idx, dtype = tf$int32)
      b_temp_tensor <- tf$gather(data_tensor, idx_tensor) + tf$constant(-2, dtype = tf$float64) * data_tensor[j, ]
      
      repeated_values <- tf$reshape(tf$gather(two_Lmatrix[, j], idx_tensor), shape = as.integer(c(length(idx), 1)))
      two_Lmatrix_idx_j <- tf$tile(repeated_values, multiples = as.integer(c(1, n)))
      
      # Calculate norm_bM_tensor
      norm_bM_tensor <- tf$sqrt(
        tf$abs(two_Lmatrix_idx_j + two_Lmatrix[j, ] - tf$gather(Lmatrix_tensor, idx_tensor))
      )
      norm_bM_tensor <- tf$where(norm_bM_tensor < tol, tf$fill(tf$shape(norm_bM_tensor), tf$constant(Inf, dtype = tf$float64)), norm_bM_tensor)
      norm_bM_tensor <- 1 / norm_bM_tensor
      C <- tf$matmul(tf$transpose(b_temp_tensor), norm_bM_tensor)
      C <- tf$transpose(C)
      norm_csum <- tf$reduce_sum(norm_bM_tensor, axis = 0L)
      C2 <- tf$reshape(norm_csum, shape = c(-1L, 1L)) * data_tensor
      
      C <- tf$add(C, C2)
      C <- tf$add(C, Ematrix_tensor)
      dm[j, ] <- 1 - apply(as.array(C * tf$constant(rn_inv, dtype = tf$float64)), 1, function(x) sqrt(sum(x^2)))
    }
    
  } else {
    
    two_Lmatrix <- 2 * Lmatrix
    
    for (j in 1:n) {
      idx <- !(1:n == j)
      b_temp <- sweep(data[idx,],2,STATS = -2*data[j,],FUN = '+')
      norm_bM <- sqrt(abs(matrix(two_Lmatrix[idx, j], nrow = sum(idx), ncol = n) + matrix(two_Lmatrix[j,], nrow = sum(idx), ncol = n, byrow = TRUE) - Lmatrix[idx,]))
      index <- (norm_bM < tol)
      if (sum(index) > 0) {
        norm_bM[index] <- Inf
      }
      norm_bM <- 1/norm_bM
      C <- t(t(b_temp)%*%norm_bM)
      norm_csum <- colSums(norm_bM)
      C2 <- norm_csum * data
      C <- C + C2 + Ematrix
      dm[j,] <- 1 - apply(C * rn_inv, 1, function(x) sqrt(sum(x^2)))
    }
  }
  return(list(dm=dm, Lmatrix=Lmatrix_save))
}
#knn for depth-based similarity matrix
KNNdep<-function(k,K,dm0,classes){
  #k: k for knn
  #K: number of clusters
  #dm0: depth-based similarity matrix (can be non-symmetric)
  #classes: known classes of points
  D<-dim(dm0)[1]
  Dmatrix<-matrix(0,D,k)
  for (j in 1:D) {
    d<-dm0[j,]
    c<-classes[sort.list(d,decreasing = T)[1:K]]
    Dmatrix[j,]<-sapply(1:k,function(i){which(c==i)%>%length()})
  }
  Class_med=rep(0,D)
  for (x in 1:D){
    result<-which(Dmatrix[x,]==max(Dmatrix[x,]))
    lr=length(result)
    #tie breaker, based on the most similar neighbor
    if (lr>1){
      maxsim2clus=sapply(1:lr,function(i){max(dm0[x,which(classes==result[i])])})
      max_index=which.max(maxsim2clus)
      result=result[max_index]
    }
    Class_med[x]=result
  }
  return(list(Dmatrix=Dmatrix,class=Class_med))
}

#depth_by_cluster: depth with regard to each cluster
depth_by_cluster<-function(X,Kclus,cluster,sub,depth){
  if (missing(depth)){
    depth<-'Mahalanobis'
  }
  subX<-X[sub,]
  DD<- matrix(0,dim(subX)[1],Kclus) 
  for(i in (1:Kclus)){ 
    if (depth=='Mahalanobis'){
      DD[,i] <- depth.Mahalanobis(subX,X[cluster[[i]],],mah.estimate = 'MCD')
    } else if (depth=='spatial'){
      DD[,i] <- spatial.d(subX,X[cluster[[i]],])
    }
  } 
  return(DD)
}
#integrated local depth with uniformaly weighted function
integrated_ld<-function(data,dm0,nbr_list=c(),depth,beta_list=c(),Lmatrix=c()){
#dm0: depth-based similarity matrix (non-symmetric) for defining beta neighborhood
#beta_list/nbr_list: locality/neighborhood levels for computing (if user want to define manually, input either of them)
#depth: depth used
#Lmatrix: when provided, quick spatial depth based ILD computation is applied
  n=nrow(data)
  d=ncol(data)
  dm0.order<-sapply(1:n,function(i){sort.list(dm0[i,],decreasing = T)})
  if (length(beta_list)!=0){
    nbr_list<-ceiling(beta_list*n)
  }
  if (length(nbr_list)==0){
    start_point<-min((2*d),10)
    nbr_list<-seq(start_point,n,1)
  }
  b=length(nbr_list)
  ld.mat=matrix(0, n, b)
  
  if (nrow(Lmatrix)%>%is.null()){
    for (i in nbr_list) {
      ri=which(nbr_list==i)
      if (i!=n){
        for(j in 1:n){
          if (depth=='Mahalanobis'){
            ld.mat[j,ri]<-depth.Mahalanobis(data[dm0.order[1,j],],data[dm0.order[1:i,j],],mah.estimate	='MCD')
          } else if (depth=='spatial'){
            ld.mat[j,ri]<-spatial.d(data[dm0.order[1,j],],data[dm0.order[1:i,j],])
          }
        }
      } else {
        if (depth=='Mahalanobis'){
          ld.mat[,ri]=depth.Mahalanobis(data,data,mah.estimate	='MCD')
        } else if (depth=='spatial'){
          ld.mat[,ri]=spatial.d(data,data)
        }
      }
    }
  } else {
    tol = 1e-5;
    index=Lmatrix<tol
    Lmatrix[index] <- Inf
    Lmatrix <- 1 / Lmatrix
    
    for (j in 1:n){
      label=dm0.order[1:max(nbr_list),j]
      Lmatrix_sub=Lmatrix[j,label]
      norm_csum <- cumsum(Lmatrix_sub)
      norm_csum<-norm_csum[nbr_list]
      C2<-outer(norm_csum, data[j,]%>%as.numeric, FUN = "*")
      C_all <- data[label,]*Lmatrix_sub
      cumsum_matrix <- apply(C_all, 2, cumsum)
      C <- cumsum_matrix[nbr_list, ]
      C <- C2-C
      mean_C<-sweep(C, 1, nbr_list, FUN = "/")
      ld.mat[j,]=1-apply(mean_C, 1, function(x) sqrt(sum(x^2)))
    }
  }
  
  int_ld_mat=t(apply(ld.mat, 1, function(row) cumsum(row) / seq_along(row)))

  return(list(ld.mat=ld.mat,dm0.order=dm0.order,ILD_mat=int_ld_mat,nbr_list=nbr_list))
}
#find B*n and valid local centers satisfying locally deepest in their own neighborhoods
find_peak_size <- function(ILD_mat, dm0.order, nbr_list,min_nbr_th=20) {
#ILD_mat: integrated-local-depth matrix
#dm0.order: beta-nbr matrix
#nbr_list: num of nbrs considered in each level
#min_nbr_th: the smallest number of locality levels involved for a reasonable ILD value
#(20 is a suggested rule-of-thumb except for extremely small data sets, see  Statistical Rules of Thumb (2nd ed.), p.22, 1.15)
  n_points <- nrow(ILD_mat)
  n_levels <- ncol(ILD_mat)
  ranking_mat <- apply(ILD_mat, 2, function(x) rank(-x, ties.method = "first"))
  highest_ld_idx <- vector("list", n_points)
  est_size=rep(0,n_points)
  #save_ild=c()
  for (i in seq_len(n_points)) {
    ild_row <- ILD_mat[i, ]
    highest_ld <- numeric(n_levels)
    for (j in seq_along(nbr_list)) {
      k <- nbr_list[j]
      # Get the neighbor indices for point i at local level j (first k neighbors)
      nbrs <- dm0.order[1:k, i]
      candidate_ranks <- ranking_mat[nbrs, j]
      #the one with highest ild
      best_neighbor <- nbrs[which.min(candidate_ranks)]
      highest_ld[j] <- best_neighbor
    }
    #save result
    highest_ld_idx[[i]] <- highest_ld
    own_max_idx=which(highest_ld==i) 
    own_max_idx=own_max_idx[own_max_idx>min_nbr_th]
    
    lidx <- length(own_max_idx)
    if (lidx >= 2) {

      pos_idx=own_max_idx[which.max(ild_row[own_max_idx])]
      est_size[i]=nbr_list[pos_idx]
      #  save_ild=c(save_ild,ild_row[pos_idx])
    }
  }
  return(list(highest_ld_idx=highest_ld_idx,est_size=est_size))
}
#filter local centers to satisfy frequency requirement
check_lc<-function(est_size,highest_ld_idx,nbr_list,dm0.order){
  current_lc<-which(est_size!=0)
  save_lc=current_lc
  save_size=est_size[est_size!=0]
  own_prop_save=rep(0,length(save_lc))
  min_count=0
  
  for (i in current_lc){
    s=est_size[i]
    pos_s=which(nbr_list==s)
    cols_with_tg <- apply(dm0.order[1:s, ], 2, function(col) any(col == i))
    result_vector <- sapply(highest_ld_idx[cols_with_tg], function(sublist) sublist[pos_s])
    own_prop=sum(result_vector==i)/length(result_vector)
    if (own_prop<0.5){
      idx=which(save_lc==i)
      save_size=save_size[-idx]
      save_lc=save_lc[-idx]
    }
  }
  
  
  freq_table=unlist(highest_ld_idx)%>%table%>%sort(decreasing = T)
  sorted_idx=order(match(save_lc%>%as.character(),names(freq_table)))
  sorted_save_lc <- save_lc[sorted_idx]
  save_size=save_size[sorted_idx]
  
  return(list(save_lc=sorted_save_lc,save_size=save_size,freq_table=freq_table))
}
#group similarity matrix computation
sm_computer<-function(save_lc,sym_dm){
  #save_lc: local centers
  #sym_dm: symmetric similarity matrix for points
  
  #G_1, to G_T
  assignments <- apply(sym_dm, 1, function(x) {
    save_lc[which.max(x[save_lc])]
  })
  counts <- table(assignments)
  sort_counts<-counts[save_lc%>%as.character()]
  #check any group only contains 1 point, if exists drop and update
  if (any(sort_counts==1)){
    save_lc=save_lc[sort_counts!=1]
    assignments <- apply(sym_dm, 1, function(x) {
      save_lc[which.max(x[save_lc])]
    })
    counts <- table(assignments)
    sort_counts<-counts[save_lc%>%as.character()]
  }
  n_lc=length(save_lc)
  r<-rep(0,n_lc)
  nbr_save <- vector("list", n_lc)
  #fr_save<-nbr_save
  
  for (i in 1:n_lc) {
    own=save_lc[i]
    nbr=which(assignments==own)
    nbr_save[[i]]=nbr
  }
  
  #to reduce some computation, only compute group similarity with T
  TFmatrix <- matrix(FALSE, nrow = n_lc, ncol = n_lc)
  
  for (g in seq_along(nbr_save)) {
    own_pt= save_lc[g]
    pts <- nbr_save[[g]]
    inter_min=sym_dm[pts,pts]%>%min()
    intra_max=sapply(nbr_save[-g],function(x){
      sym_dm[own_pt, x]%>%max()
    })
    rec_flag <- intra_max>=inter_min
    #rec_flag=rec_flag1 | rec_flag2
    TFmatrix[g, -g] <- rec_flag
  }
  TFmatrix<-pmax(TFmatrix,t(TFmatrix))
  sym_sm=sim_matrix_generator(sym_dm,assignments,save_lc,TFmatrix,nbr_save,sort_counts)
  
  return(list(sym_sm=sym_sm,counts=sort_counts,TFmatrix=TFmatrix,nbr_save=nbr_save,save_lc=save_lc))
}
#GLS computation
sim_matrix_generator<-function (d,assignments,save_lc,TFmatrix,nbr_save,sort_counts) {
  #d: a similarity matrix
  #assignments: groups assignments
  #save_lc: local center
  
  N <- dim(d)[1]
  n <- length(save_lc)
  if (is.null(rownames(d)[1])) {
    rownames(d) <- 1:N
  }
  c <- matrix(0, n, n)
  upper_idx <- which(TFmatrix == 1 & upper.tri(TFmatrix), arr.ind = TRUE)
  for (pair in 1:nrow(upper_idx)) {
    group_i=upper_idx[pair,][1]
    group_j=upper_idx[pair,][2]
    for (x in nbr_save[[group_i]]) {
      for (y in nbr_save[[group_j]]) {
        #depth similarity from other points to x,y
        dx <- d[x,]
        dy <- d[y,]
        assignment_y=assignments[y]
        assignment_x=assignments[x]
        #find points in the Uxy
        uxy <- (which((dx >= d[x, y]) | (dy >= d[y, x])))
        uxy_x<-uxy[assignments[uxy]==assignment_x]
        uxy_y<-uxy[assignments[uxy]==assignment_y]
        
        #if points in cluster of x is closer to point x
        wx <- 1 * (dx[uxy_x] > dy[uxy_x]) + 0.5 * ((dx[uxy_x] ==
                                                      dy[uxy_x]))
        #if points in cluster of y is closer to point x
        wy  <- 1 * (dx[uxy_y] > dy[uxy_y]) + 0.5 * ((dx[uxy_y] == 
                                                       dy[uxy_y]))
        pos_x=group_i
        pos_y=group_j
        c[pos_x,pos_y]=c[pos_x,pos_y]+sum(wy)/length(uxy_y)
        c[pos_y,pos_x]=c[pos_y,pos_x]+sum(1-wx)/length(uxy_x)
        
      }
    }
  }
  
  normalized_c <- t(sapply(1:nrow(c), function(i) {
    c[i,] / (sort_counts * sort_counts[i])
  }))
  rownames(normalized_c) <- save_lc
  colnames(normalized_c) <- save_lc
  #sym_normalized_c=pmin(normalized_c,t(normalized_c))
  sym_normalized_c=sym_mat(normalized_c)
  return(sym_normalized_c)
}
#gls, similar with the above-one, but can use individually
group_level_sim<-function (d,assignments,nbr_save,TFmatrix) {
  N <- nrow(d)
  n <- length(nbr_save)
  if (missing(TFmatrix)) {
    TFmatrix <- matrix(1, n, n)
  }
  
  sort_counts=lengths(nbr_save)
  if (is.null(rownames(d)[1])) {
    rownames(d) <- 1:N
  }
  c <- matrix(0, n, n)
  upper_idx <- which(TFmatrix == 1 & upper.tri(TFmatrix), arr.ind = TRUE)
  for (pair in 1:nrow(upper_idx)) {
    group_i=upper_idx[pair,][1]
    group_j=upper_idx[pair,][2]
    for (x in nbr_save[[group_i]]) {
      for (y in nbr_save[[group_j]]) {
        #depth similarity from other points to x,y
        dx <- d[x,]
        dy <- d[y,]
        assignment_y=assignments[y]
        assignment_x=assignments[x]
        #find points in the Uxy
        uxy <- (which((dx >= d[x, y]) | (dy >= d[y, x])))
        uxy_x<-uxy[assignments[uxy]==assignment_x]
        uxy_y<-uxy[assignments[uxy]==assignment_y]
        
        #if points in cluster of x is closer to point x
        wx <- 1 * (dx[uxy_x] > dy[uxy_x]) + 0.5 * ((dx[uxy_x] ==
                                                      dy[uxy_x]))
        #if points in cluster of y is closer to point x
        wy  <- 1 * (dx[uxy_y] > dy[uxy_y]) + 0.5 * ((dx[uxy_y] == 
                                                       dy[uxy_y]))
        pos_x=group_i
        pos_y=group_j
        c[pos_x,pos_y]=c[pos_x,pos_y]+sum(wy)/length(uxy_y)
        c[pos_y,pos_x]=c[pos_y,pos_x]+sum(1-wx)/length(uxy_x)
        
      }
    }
  }
  
  normalized_c <- t(sapply(1:nrow(c), function(i) {
    c[i,] / (sort_counts * sort_counts[i])
  }))
  rownames(normalized_c) <- 1:n
  colnames(normalized_c) <- 1:n
  #sym_normalized_c=pmin(normalized_c,t(normalized_c))
  sym_normalized_c=sym_mat(normalized_c)
  return(list(sym_mat=sym_normalized_c,original_mat=normalized_c))
}
#convert a similarity matrix to its reachable similaity version
reachable_similarity <- function(sim_matrix) {
  n <- nrow(sim_matrix)
  if (n==2){
    R=sim_matrix
  } else {
    # Initialize result matrix with 0 (will fill in reachable similarities)
    R <- matrix(0, n, n)
    diag(R) <- 1 
    
    
    dimnames(R) <- dimnames(sim_matrix)  # preserve names if any
    
    # Treat values <= 0 as no edge (no connection)
    # Create a list of all edges (i,j) with weight > 0
    edges <- which(sim_matrix > 0, arr.ind = TRUE)
    edges <- edges[edges[,"row"] < edges[,"col"], ]        # upper triangle unique pairs
    if (is.matrix(edges)){
      edge_list <- data.frame(
        i = edges[,"row"],
        j = edges[,"col"],
        w = sim_matrix[edges]
      )
    } else {
      edge_list <- data.frame(
        i = edges["row"],
        j = edges["col"],
        w = sim_matrix[edges]
      )
    }
    
    # Sort edges by weight descending
    edge_list <- edge_list[order(-edge_list$w), ]
    
    comp <- 1:n   # comp[k] = component label of node k
    for (e in seq_len(nrow(edge_list))) {
      u <- edge_list$i[e];  v <- edge_list$j[e];  w <- edge_list$w[e]
      # If u and v are currently in different components, unify them
      if (comp[u] != comp[v]) {
        # Current component labels
        cu <- comp[u];  cv <- comp[v]
        # All nodes in each component
        nodes_u <- which(comp == cu)
        nodes_v <- which(comp == cv)
        # Update reachable similarity for all new connections between comp u and comp v
        R[nodes_u, nodes_v] <- pmax(R[nodes_u, nodes_v], w)
        R[nodes_v, nodes_u] <- pmax(R[nodes_v, nodes_u], w)
        # Merge the components: relabel component cv as cu
        comp[comp == cv] <- cu
      }
    }

  }
  
  return(R)
}
#based on new group_list, update  similarity matrix
update_group_similarity <- function(group_list, sym_sm) {
  K <- length(group_list)
  group_sim <- matrix(0, K, K)
  
  for (i in 1:(K - 1)) {
    for (j in (i + 1):K) {
      group_i <- group_list[[i]]
      group_j <- group_list[[j]]
      max_sim <- max(sym_sm[group_i, group_j])
      group_sim[i, j] <- max_sim
      group_sim[j, i] <- max_sim
    }
  }
  
  return(group_sim)
}
#compute intra-group similarity
compute_mcs <- function(after_mer) {
  cs <- colSums(after_mer)
  mcs <- (cs - 1) / (nrow(after_mer) - 1)
  return(mcs)
}
# update index
update_index <- function(input_idx, drop_idx) {
  sapply(input_idx, function(x) {
    if (x %in% drop_idx) {
      0
    } else {
      x - sum(drop_idx < x)
    }
  })
}

#accept singularly most similar pairing
check_pairs=function(pairs,sym_sm,group_list,min_close){
  keep=rep(FALSE,length(pairs))
  for (i in seq_along(pairs)) {
    pair=pairs[[i]]
    from=pair[which(lengths(group_list[pair])==1)]
    to=setdiff(pair,from)
    if (length(to)>0){
      from_to_sim=sym_sm[group_list[[from]],group_list[[to]]]
      ftc=mean(from_to_sim)
      tc=compute_mcs(sym_sm[group_list[[to]],group_list[[to]]])%>%min()
      tot=sym_sm[group_list[[to]],-group_list[[to]]]%>%rowMaxs(value=T)
      # judge1=all(from_to_sim>=tot)
      judge1= all(from_to_sim>min_close)
      judge2= ftc>=tc
      if (any(c(judge1,judge2))) keep[i]=T 
    }
  }
  pairs=pairs[keep]
  return(pairs)
}
#delta computing (linkage)
calc_pro<-function(smaller_group, larger_group, 
                   result_clusters,group_list, sym_dm, radius) {
  # smaller_group, larger_group: group index
  counts_small=result_clusters[[smaller_group]]%>%length
  counts_large=result_clusters[[larger_group]]%>%length
  to_larger <- sym_dm[result_clusters[[smaller_group]], result_clusters[[larger_group]]] %>%
    Rfast::rowMaxs(value = TRUE)
  lower_bound=sym_dm[group_list[[smaller_group]],group_list[[larger_group]]]%>%max()
  if (lower_bound>=radius){
    pro <- sum(to_larger >= radius) / counts_small
  } else {
    p_value=counts_small/counts_large
    dynamic_radius <- lower_bound * (1 - sqrt(p_value)) + radius * sqrt(p_value)
    pro <- sum(to_larger >= dynamic_radius) / counts_small
  }
  return(pro)
}
#mutually most similar pair
cnbr_generator <- function(temp_sm,singleton_group=c()) {
  n <- nrow(temp_sm)
  pairs <- list()
  if (length(singleton_group)==0){
    row_max_idx <- Rfast::rowMaxs(temp_sm, value = FALSE)
    
    for (i in seq_len(n)) {
      j <- row_max_idx[i]
      # 互为最近且不重复（只记录i<j）
      if (i < j && row_max_idx[j] == i) {
        pairs[[length(pairs) + 1]] <- c(i, j)
      }
    }
  } else {
    from_list <- integer()
    to_list <- integer()
    sim_list <- numeric()
    for (i in singleton_group) {
      sims <- temp_sm[i, ]
      sims[i] <- -Inf
      j <- which.max(sims)
      from_list <- c(from_list, i)
      to_list <- c(to_list, j)
      sim_list <- c(sim_list, sims[j])
    }
    df <- data.frame(from = from_list, to = to_list, sim = sim_list)
    df <- df[order(-df$sim), ]
    used <- rep(FALSE, n)
    result <- list()
    for (k in seq_len(nrow(df))) {
      i <- df$from[k]
      j <- df$to[k]
      if (!used[i] && !used[j]) {
        result[[length(result) + 1]] <- c(i, j)
        used[i] <- TRUE
        used[j] <- TRUE
      }
    }
    pairs <- result
  }
  return(pairs)  
}
#adaptive merging (linkage strategy)
adaptive_merge<- function(save_lc, R, sym_dm, sym_sm, nbr_save,reachable_list) {
  
  all_mer <- reachable_similarity(sym_dm)
  global_th=all_mer%>%rowmeans()%>%quantile(0.05)
  N=length(save_lc)
  current_mcs <-reachable_list
  result_clusters<-nbr_save
  # sym_dm_sub=sym_dm[save_lc,save_lc]
  N_obs=nrow(sym_dm)
  row_mean_to <-lapply(result_clusters,function(x){
    all_mer[x,-x]%>%rowmeans()
  })
  
  
  bth_all=sapply(1:N,function(x){mean(row_mean_to[[x]]/current_mcs[[x]])})
  bth_all[bth_all>0.99]<-0.99
  bth_all[bth_all<0.9]<-0.9
  
  R_active <- (sym_sm != 0) * 1
  diag(R_active) <- 0
  group_list<-lapply(1:N,function(x){x})

  temp_sm=sym_sm
  diag(temp_sm)<-0
  
  accept_save<-vector("list",length=N)
  
  # first stage
  pairs <- cnbr_generator(temp_sm)  # find all mutually most similar group
  ps<- sapply(pairs, function(x) sym_sm[x, x] %>% min) 
  th_fs=min(ps)
  min_close=min(temp_sm%>%rowMaxs(value=T))
  if (th_fs==min_close){
    pairs=pairs[ps>th_fs]
    th_fs=ps[ps>th_fs]%>%min()
  }
  min_pos=sym_sm[sym_sm!=0]%>%min
  # repeatly merging mutually pairs
  repeat {
    if (length(pairs) == 0){
      merged_flag=T
      break
    }
    merged_flag <- FALSE
    to_merge_list <- list()
    mcs_save <- list()
    Q_save= c()
    for (merge_target in pairs) {
      merge_ids <- unlist(result_clusters[merge_target])
      after_mer <- reachable_similarity(sym_dm[merge_ids, merge_ids])
      mcs <- compute_mcs(after_mer = after_mer)
      delta <- rep(0, length(merge_target))
      for (i in seq_along(merge_target)) {
        idx <- merge_target[i]
        g_points <- result_clusters[[idx]]
        g_mean <- current_mcs[[idx]]
        pos_idx <- match(g_points, merge_ids)
        to_other <- after_mer[pos_idx, -pos_idx, drop=FALSE] %>% rowMeans()
        delta[i] <- mean(to_other / g_mean)
      }
      Q_min <- min(delta)
      Q_th=bth_all[merge_target]%>%min()
      if (Q_min > Q_th) {
        to_merge_list[[length(to_merge_list) + 1]] <- merge_target
        mcs_save= append(mcs_save,list(mcs))
        Q_save=c(Q_save,Q_min) 
      } else {
        R_active[merge_target,merge_target]<-0
      }
    }
    
    n_m=length(to_merge_list)
    if (n_m > 0){
      drop_idx=c()
      for (i in 1:n_m) {
        merge_target=to_merge_list[[i]]
        change_pos=min(merge_target)
        drop_idx=c(drop_idx,setdiff(merge_target,change_pos))
        result_clusters[[change_pos]]=unlist(result_clusters[merge_target])
        current_mcs[[change_pos]]=mcs_save[[i]]
        rmt=all_mer[result_clusters[[change_pos]],-result_clusters[[change_pos]]]%>%rowmeans()
        bth_value <- max(0.9, min(0.99, mean(rmt/mcs_save[[i]])))
        bth_all[change_pos]=bth_value
        group_list[[change_pos]]<-group_list[merge_target]%>%unlist()
        #matrix update
        value= colSums(R_active[merge_target,]) > 0
        R_active[change_pos,]<-value
        R_active[,change_pos]<-value
        diag(R_active)<-0
        accept_save[[change_pos]]<-c(accept_save[[change_pos]],accept_save[[drop_idx[length(drop_idx)]]],Q_save[i])
      }
      if (length(drop_idx)>0){
        R_active=R_active[-drop_idx,-drop_idx]
        result_clusters[drop_idx]<-c()
        group_list[drop_idx]<-c()
        current_mcs[drop_idx]<-c()
        accept_save[drop_idx]<-c()
        bth_all=bth_all[-drop_idx]
      }
    }
    temp_sm=update_group_similarity(group_list,sym_sm)
    temp_sm<- temp_sm*(R_active!=0)
    pairs=cnbr_generator(temp_sm)
    sim_pairs=sapply(pairs,function(x){temp_sm[x,x]%>%max})
    idx_accept=sim_pairs>=th_fs
    if (sum(idx_accept)>0){
      pairs=pairs[idx_accept]
    } else {
      #check singly pair between singleton group and others
      lg=lengths(group_list)
      singleton_group=which(lg==1)
      if (length(singleton_group)>0){
        pairs=cnbr_generator(temp_sm,singleton_group)
        pairs=check_pairs(pairs,sym_sm,group_list,min_close = min_pos)
      } else {
        pairs=pairs[idx_accept]
      }
      if (length(pairs)>0){
        th_fs <- min(sapply(group_list[lg>1],function(x){
          sub=sym_sm[x,x]
          diag(sub)<-0
          Rfast::rowMaxs(sub,value=T)%>%min()
        }))
      }
    }
  }
  # scd stage
  while  (max(R_active)>0) {
    temp_sm=update_group_similarity(group_list,sym_sm)
    temp_sm<- temp_sm*(R_active!=0)
    mts=max(temp_sm)
    if (mts!=0) {
      all_pairs=cnbr_generator(temp_sm)
      involve_point=unlist(all_pairs)%>%unique()%>%sort()
      base=involve_point[which.min(lengths(result_clusters[involve_point]))]
      merge_target=all_pairs[sapply(all_pairs,function(x){base%in%x})]%>%unlist()%>%unique()
      merge_ids <- unlist(result_clusters[merge_target])
      after_mer <- reachable_similarity(sym_dm[merge_ids, merge_ids])
      mcs=compute_mcs(after_mer = after_mer)
      delta=rep(0,length(merge_target))
      prop_worse=delta
      for (i in seq_along(merge_target)) {
        idx=merge_target[i]
        g_points=result_clusters[[idx]]
        g_mean <- current_mcs[[idx]]
        pos_idx=match(g_points, merge_ids)
        to_other<-after_mer[pos_idx,-pos_idx]%>%rowmeans()
        delta[i]<- mean(to_other/g_mean)
        prop_worse[i]<-sum(mcs[pos_idx]<g_mean)/length(g_points)
      }
      Q_min=min(delta)
      bth=bth_all[merge_target]%>%min()
      prop_worse=mean(prop_worse)
      median_accept=accept_save[merge_target]%>%unlist%>%median()
      median_accept=min(median_accept,1)
      if (Q_min<median_accept){
        cluster_size=lengths(result_clusters[merge_target])
        smaller_group=merge_target[which.min(cluster_size)]
        larger_group=merge_target[which.max(cluster_size)]
        radius=mean(c(global_th,quantile(mcs,0.05)))
        gap_prop=1-calc_pro(smaller_group,larger_group,result_clusters,group_list,sym_dm,radius = radius)
        Q_th=bth+gap_prop*prop_worse^2*(median_accept-bth)
      } else {
        Q_th=bth
      }
      if (Q_min>Q_th){
        change_pos=min(merge_target)
        drop_idx=setdiff(merge_target,change_pos)
        result_clusters[[change_pos]]=unlist(result_clusters[merge_target])
        current_mcs[[change_pos]]=mcs
        rmt=all_mer[result_clusters[[change_pos]],-result_clusters[[change_pos]]]%>%rowmeans()
        bth_value <- max(0.9, min(0.99, mean(rmt/mcs)))
        bth_all[change_pos]=bth_value
        group_list[[change_pos]]<-group_list[merge_target]%>%unlist()
        #matrix update
        value= colSums(R_active[merge_target,]) > 0
        R_active[change_pos,]<-value
        R_active[,change_pos]<-value
        diag(R_active)<-0
        accept_save[[change_pos]]<-c(accept_save[[change_pos]],accept_save[[drop_idx]],Q_min)
        R_active=R_active[-drop_idx,-drop_idx]
        result_clusters[drop_idx]<-c()
        group_list[drop_idx]<-c()
        current_mcs[drop_idx]<-c()
        accept_save[drop_idx]<-c()
        bth_all=bth_all[-drop_idx]
      } else {
        R_active[merge_target,merge_target]<-0
      }
    } else {
      R_active[R_active!=0]=0
    } 
  }
  group_list<-lapply(group_list,function(x){save_lc[x]})
  

  return(list(group_list=group_list,temp_clus=result_clusters))
  
}
#adaptive merging (centroid strategy)
adaptive_flex_merge<- function(save_lc, R, sym_dm, sym_sm, nbr_save,reachable_list) {
  all_mer <- reachable_similarity(sym_dm)
  
  valid_neighbors_list=lapply(seq_along(save_lc),function(x){
    label=setdiff(which(sym_sm[x,]!=0),x)
    label=sort(sym_sm[x,][label],decreasing = T)
  })
  group_list=lapply(save_lc,function(x){x})
  
  row_mean_to <-lapply(nbr_save,function(x){
    all_mer[x,-x]%>%rowmeans()
  })
  bth_all=sapply(1:length(save_lc),function(x){mean(row_mean_to[[x]]/reachable_list[[x]])})
  drop_idx_base <- which(bth_all > 1)
  if (length(drop_idx_base)>0){
    dl_mat=(sym_sm==R) & (sym_sm!=0)
    mean_reachable=sapply(reachable_list,mean)
    est_size=lengths(nbr_save)
    m_size=median(est_size)
    
    is_drop <- sapply(drop_idx_base, function(i) {
      direct_nbr<-setdiff(which(dl_mat[i,]),drop_idx_base)
      own_reach=mean_reachable[i]
      if (length(direct_nbr)>0){
        nbr_reach=mean_reachable[direct_nbr]
        min(nbr_reach)>own_reach
      } else {
        est_size[i]<m_size
      }
    })
    drop_idx <- drop_idx_base[is_drop]
  } else {
    drop_idx=c()
  }
  if ((length(save_lc)-length(drop_idx))>1){
    if (length(drop_idx)>0){
      save_lc=save_lc[-drop_idx]
      valid_neighbors_list=valid_neighbors_list[-drop_idx]
      sym_sm=sym_sm[-drop_idx,-drop_idx]
      nbr_save=nbr_save[-drop_idx]
      reachable_list=reachable_list[-drop_idx]
      R=R[-drop_idx,-drop_idx]

      left_obs=unlist(nbr_save)
      all_mer_sub<- reachable_similarity(sym_dm[left_obs,left_obs])
      row_mean_to <-lapply(nbr_save,function(x){
        idx=match(x,left_obs)
        all_mer[idx,-idx]%>%rowmeans()
      })
      bth_all=sapply(1:length(save_lc),function(x){mean(row_mean_to[[x]]/reachable_list[[x]])})
    }
  }
  bth_all[bth_all>0.99]<-0.99
  bth_all[bth_all<0.9]<-0.9
  N <- length(save_lc)
  if (N>2){
    accept_save<-vector("list",length=N)
    reject_save=accept_save
    considered_nbr_list <- vector('list',length=N)
    overall_max=max(sym_sm*upper.tri(sym_sm))
    bound=1
    stop_bound=0.9*min(bth_all)
    for (i in 1:N){
      sort_sim=valid_neighbors_list[[i]]
      neighbors=sort_sim%>%names()
      pos=match(neighbors,save_lc)
      pos=na.omit(pos)
      pos=pos[pos<i]
      sim_values=sym_sm[i,][-i]
      min_sim=min(sim_values)
      gap_sim=overall_max-min(sim_values)
      if (length(pos)>0){
        l_pos=length(pos)
        Q=1
        j=1
        while (Q > stop_bound &j<=l_pos) {
          idx=pos[j]
          sv=sym_sm[i,idx]
          merge_target=c(i,idx)
          merge_ids <- unlist(nbr_save[merge_target])
          after_mer <- reachable_similarity(sym_dm[merge_ids, merge_ids])
          # direct_nbr=which(dl_mat[i,])
          mcs=compute_mcs(after_mer = after_mer)
          delta=rep(0,length(merge_target))
          prop_worse=delta
          
          for (u in seq_along(merge_target)) {
            v=merge_target[u]
            g_points=nbr_save[[v]]
            g_mean <- reachable_list[[v]]
            pos_idx=match(g_points, merge_ids)
            to_other<-after_mer[pos_idx,-pos_idx]%>%rowmeans()
            delta[u]<- mean(to_other/g_mean)
            prop_worse[u]<-sum(mcs[pos_idx]<g_mean)/length(g_points)
          }
          Q=min(delta)
          Q_base=bth_all[merge_target]%>%min()
          if (Q_base<bound){
            prop_worse=mean(prop_worse)

            gap_prop=1-(sv-min_sim)/gap_sim
            Q_th=Q_base+gap_prop*prop_worse^2*(bound-Q_base)
          } else {
            Q_th=Q_base
          }
          if (Q>Q_th){
            considered_nbr_list[[i]]=c(considered_nbr_list[[i]],idx)
            accept_save[[i]]<-c(accept_save[[i]],Q)
            if (Q<bound){
              bound=Q
            }
          } else {
            reject_save[[i]]<-c(reject_save[[i]],setNames(Q, idx))
          }
          
          j=j+1
        }
      }
    }
    
    ifstable<-lengths(considered_nbr_list)==0
    
    if (sum(ifstable)==1){
      accept_value=sapply(accept_save, function(x) if(length(x) == 0) 1 else max(x))
      ifstable[which.min(accept_value)]=T
    }
  } else {
    ifstable=rep(T,N)
  }
  
  
  group <- sapply(seq_along(save_lc), function(x) {
    row_R <- R[x, ifstable] * sym_sm[x, ifstable]
    candidate_idx <- which(row_R == max(row_R))
    if(length(candidate_idx) == 1) {
      return(candidate_idx)
    } else {
      row_sim <- R[x, ifstable][candidate_idx]
      chosen <- candidate_idx[which.max(row_sim)]
      return(chosen)
    }
  })
  
  group_list <- split(seq(save_lc), group)
  nbr_save <- split(seq_len(nrow(sym_dm)), apply(sym_dm[, save_lc, drop=FALSE], 1, which.max))
  result_clusters <- lapply(group_list, function(s) nbr_save[s]%>%unlist)
  
  est_num=lengths(result_clusters)

  low_size=median(est_num)/2
  reconsider=which(est_num<low_size)

  lrec=length(reconsider)
  lgap=length(group_list)-lrec
  #check for small size group
  if (lrec>0& lgap>1){
    reachable_list=lapply(result_clusters,function(x){
      rsm=reachable_similarity(sym_dm[x,x])
      cs <- colSums(rsm)
      (cs-1)/(length(x)-1)
    })
    row_mean_to <-lapply(result_clusters,function(x){
      all_mer[x,-x]%>%rowmeans()
    })
    bth_all=sapply(1:length(result_clusters),function(x){mean(row_mean_to[[x]]/reachable_list[[x]])})
    
    
    max_num=max(est_num)
    weighted_R=sym_sm*R
    temp_sm=update_group_similarity(group_list,weighted_R)
    if (length(reconsider)>1){
      row_max_idx <- Rfast::rowMaxs(temp_sm[reconsider,], value = FALSE)  # 子矩阵内部列下标
    } else {
      row_max_idx=which.max(temp_sm[reconsider,])
    }
    pair_list <- lapply(seq_along(reconsider), function(i) c(reconsider[i],row_max_idx[i])%>%sort)
    pair_list=unique(pair_list)
    num_point=sapply(pair_list,function(x){est_num[x]%>%sum})
    pair_list=pair_list[num_point<=max_num]
    while(length(pair_list)>0){
      merge_target=pair_list[[1]]
      sizes=result_clusters[merge_target]%>%lengths
      size_sum=sum(sizes)
      merge_ids <- unlist(result_clusters[merge_target])
      after_mer <- reachable_similarity(sym_dm[merge_ids, merge_ids])
      mcs=compute_mcs(after_mer)
      current_rowmean <- all_mer[merge_ids,-merge_ids]%>%rowmeans()
      update_ratio=mean(current_rowmean/mcs)
      w1=sqrt(sizes[1])
      w2=sqrt(sizes[2])
      origin_ratio=w1/(w1+w2)*bth_all[merge_target[1]]+w2/(w1+w2)*bth_all[merge_target[2]]
      # origin_ratio=sizes[1]/(size_sum)*bth_all[merge_target[1]]+sizes[2]/(size_sum)*bth_all[merge_target[2]]
      if (update_ratio<=origin_ratio){
        left_label <- merge_target[which.min(merge_target)]
        drop_label <- merge_target[which.max(merge_target)]
        group_list[[left_label]] <- c(group_list[[left_label]], group_list[[drop_label]])
        result_clusters[[left_label]]<-c(result_clusters[[left_label]], result_clusters[[drop_label]])
        bth_all[left_label]<-update_ratio
        group_list[[drop_label]] <- NULL
        result_clusters[[drop_label]] <- NULL
        pair_list <- pair_list[-1]

        pair_list <- lapply(pair_list, function(x) sort(replace(x, x == drop_label, left_label)))
        pair_list <- lapply(pair_list, function(x) update_index(x,drop_label))
        if (size_sum < low_size) {
          temp_sm=update_group_similarity(group_list,weighted_R)
          append_content <- c(left_label,which.max(temp_sm[left_label,]))
          if (sum(lengths(result_clusters[append_content]))<=max_num){
            pair_list=append(pair_list,list(append_content))
          }
        }
       
        pair_list <- unique(pair_list)
      } else {
        pair_list <- pair_list[-1]
      }
    }
    
  }
  
  group_list=lapply(group_list,function(x){save_lc[x]})
  

  
  return(list(group=group,group_list=group_list,temp_clus=result_clusters))
}
# choosing strategies
LNI_strategy_choice <- function(sym_sm, R) {
  d=nrow(R)
  diag(R)<-0
  diag(sym_sm)<-0
  value=R-sym_sm
  prop=sum(value[sym_sm<0.01])/sum(R)
  ambiguity=F
  if (prop<=0.4){
    strategy='centroid'
  } else if (prop>=0.6){
    strategy='linkage'
  } else {
    R0_prop=(sum(R<0.01)-d)/(sum(sym_sm<0.01)-d)
    if (R0_prop>0.5){
      strategy='linkage'
    } else {
      strategy='centroid'
    }
    ambiguity=T
  }
  return(list(prop=prop,strategy=strategy,ambiguity=ambiguity))
}
#if proceed to merging steps or each local center one group
ifmerging<-function(dm0.order,nbr_save,save_lc,N){
  if (N==2){
    need_merging=F
  } else {
    max_nbr <- max(lengths(nbr_save))
    size <- round(max_nbr / 2)
    
    need_merging <- FALSE
    for(i in seq_along(save_lc)) {
      idx <- save_lc[i]
      nbrs <- dm0.order[2:size, idx]        
      # if any other exemplars in its neighbors, set T
      if(any(nbrs %in% setdiff(save_lc, idx))) {
        need_merging <- TRUE
        break
      }
    }
  }
  return(need_merging)
}

#stable local centers grouping
find_stable_centers_ld <- function(ILD_mat,nbr_list,matrix_info,sym_dm,strategy=c(),dm0.order) {
  save_lc=matrix_info$save_lc
  counts=matrix_info$counts
  sym_sm=matrix_info$sym_sm
  TFmatrix=matrix_info$TFmatrix
  nbr_save=matrix_info$nbr_save
  
  N=length(save_lc)
  
  
  R=reachable_similarity(sym_sm)
  diag(sym_sm)<-1
  
  if (length(strategy)==0){
    LNI_result=LNI_strategy_choice(sym_sm,R)
    strategy=LNI_result$strategy
    if (LNI_result$ambiguity){
      warning(sprintf(
        "p is %.3f, which is in the ambiguous range (0.4 ~ 0.6). You may try the other strategy to see the difference.",
        LNI_result$prop
      ))    }
  }
  reachable_list=lapply(nbr_save,function(x){
    rsm=reachable_similarity(sym_dm[x,x])
    cs <- colSums(rsm)
    (cs-1)/(length(x)-1)
  })
  if (strategy=='linkage'){
    group_list=adaptive_merge(save_lc,R,sym_dm, sym_sm,nbr_save,reachable_list)
    return(list(group_list=group_list, strategy='linkage'))
  } else {
    merging=ifmerging(dm0.order,nbr_save,save_lc,N)
    if (merging){
      group_info=adaptive_flex_merge(save_lc,R,sym_dm,sym_sm,nbr_save,reachable_list)
    } else {
      group_info=list(group_list=split(save_lc,1:N),temp_clus=nbr_save)
    }
    return(list(group_list=group_info,strategy='centroid'))
  }
}
#check retained points, construct initial temporary clusters
check_point<-function(group_list,sym_dm,assignments,n){
  K=length(group_list)
  
  temp_clus<-vector('list',length=K)
  left_clus<-temp_clus
  
  for (i in 1:K) {
    a_c=group_list[[i]]
    lac=length(a_c)
    if (lac>1){
      nbr_ac<-lapply(a_c,function(x){which(assignments==x)})
      ng_points=setdiff(1:n,unlist(nbr_ac))
      for (j in 1:lac) {
        points=nbr_ac[[j]]
        is=sym_dm[
          points,
          nbr_ac[-j]%>%unlist
        ] %>% rowMaxs(value = T)
        bs=sym_dm[
          points,
          ng_points
        ] %>% rowMaxs(value = T)
        TFlabel=is-bs>0
        TFlabel[which(points==a_c[j])]<-T
        temp_clus[[i]]=c(temp_clus[[i]],points[TFlabel])
        left_clus[[i]]=c(left_clus[[i]],points[!TFlabel])
      }
    } else {
      points=which(assignments==a_c)
      ng_points=setdiff(1:n,points)

      overlap = sapply(points, function(j) {
        threshold = sym_dm[j, a_c]
        overlap = which(sym_dm[j, ng_points] >= threshold)
        length(overlap)
      })
      TFlabel=(overlap==0)
      temp_clus[[i]]=points[TFlabel]
      left_clus[[i]]=points[!TFlabel]
    }
  }
  
  return(list(temp_clus=temp_clus,left_clus=left_clus))
}

#obtain final temporary clusters
get_temp_clus<-function(sym_dm,group_info,strategy,data,freq_table=c()){
  group_list=group_info$group_list
  tmp_clus=group_info$temp_clus
  
  a<-unlist(group_list)
  lg<-lengths(group_list)
  Kclus=length(group_list)
  clus.ind<-1:Kclus
  cl<-rep(clus.ind,lg)
  
  if (any(lg==1)){
    add_pos=which(lg==1)

    candidates=setdiff(freq_table%>%names,a)
    candidates=candidates%>%as.numeric()
    filled_singletons <- rep(0, length(add_pos))
    
    for (v in add_pos) {
      n_can=length(candidates)
      singleton_a=group_list[[v]]
      to_sa=sym_dm[candidates,singleton_a]
      to_others=sym_dm[candidates,setdiff(a,singleton_a)]%>%rowMaxs(value=T)
      ranks_idx=which(to_sa>to_others)
      candidates_v=candidates[ranks_idx]
      
      depth_value=spatial.d(data[candidates_v,],data[tmp_clus[[v]],])*(n_can-ranks_idx+1)/n_can
      best_point=candidates_v[which.max(depth_value)]
      filled_singletons[which(add_pos==v)]=best_point
      candidates=setdiff(candidates,best_point)
    }
    

    for (u in seq_along(add_pos)) {
      if (filled_singletons[u]!=0){
        group_list[[add_pos[u]]]<-c(group_list[[add_pos[u]]],filled_singletons[u])
      }
    }
    #update_info
    a<-unlist(group_list)
    lg<-lengths(group_list)
    Kclus=length(group_list)
    clus.ind<-1:Kclus
    cl<-rep(clus.ind,lg)
  }
  
  
  n=nrow(sym_dm)
  
  assignments <- apply(sym_dm, 1, function(x) {
    a[which.max(x[a])]
  })
  
  temp_clus_result=check_point(group_list,sym_dm,assignments,n)
  temp_clus=temp_clus_result$temp_clus
  left_clus=temp_clus_result$left_clus
  
  # if (strategy!='linkage') {
  score_temp<-Assign_score(Kclus,sym_dm,temp_clus,group_list)
  left_num<-sum(lengths(left_clus))
  if (left_num>0){
    score_temp2<-Assign_score(Kclus,sym_dm,left_clus,group_list)
    left_scores=unlist(score_temp2)
    if (length(left_scores)>0){
      mean_score2 <- mean(left_scores)
    } else {
      mean_score2=0
    }
    if (strategy!='linkage'){
      lq=sapply(1:Kclus, function(x) {
        len2 <- length(score_temp2[[x]])
        len <- length(score_temp[[x]])
        if (len2 <=len) {
          return(mean_score2)
        }else {
          return(0.5*mean_score2)
        }
      })
    } else {
      lq=rep(0.5*mean_score2,Kclus)
    }
    
    
    for (i in 1:Kclus){
      index<-which(score_temp[[i]]<lq[i])
      if (length(index)>0){
        left_clus[[i]]<-c(left_clus[[i]],temp_clus[[i]][index])
        score_temp2[[i]]<-c(score_temp2[[i]],score_temp[[i]][index])
        temp_clus[[i]]<-temp_clus[[i]][-index]
        score_temp[[i]]<-score_temp[[i]][-index]
      }
    }
    
    
    new.border<-c()
    ltc<-lengths(temp_clus)
    llc<-lengths(left_clus)
    for (i in clus.ind) {
      num_obs_current<-ltc[i]
      num_total<-num_obs_current+llc[i]
      sortscore<-sort(score_temp[[i]],decreasing = T)[floor(num_obs_current/2):num_obs_current]
      cdd_1<-sortscore[which.min(diff(sortscore))]
      if (num_obs_current<(0.5*num_total)){
        temp.s<-1-(0.5*num_total-num_obs_current)/length(score_temp2[[i]])
        if (temp.s>0){
          cdd_2<-quantile(score_temp2[[i]],temp.s)
        } else {
          cdd_2<-0
        }
      } else {
        cdd_2=1
      }
      new.border[i]<-min(cdd_1,cdd_2)
    }
    
    for (x in clus.ind) {
      index<-which(score_temp2[[x]]>new.border[x])
      left_clus[[x]]<-left_clus[[x]][index]
    }
    
    for (i in clus.ind) {
      temp_clus[[i]]<-c(temp_clus[[i]],left_clus[[i]])
    }
  }
  # }
  return(list(temp_clus=temp_clus,group_list=group_list,stable_centers=a))
}

#scores for clustering
Assign_score<-function(Kclus,dm0,temp.clus,temp.cl){
  a<-unlist(temp.cl)
  deflist<-vector('list',length=Kclus)
  for (k in 1:Kclus){
    tmp.l<-length(temp.clus[[k]])
    if (tmp.l!=0){
      max_a<-sapply(1:tmp.l,function(x){max(dm0[temp.cl[[k]],temp.clus[[k]][x]])})
      max_b<-sapply(1:tmp.l,function(x){max(dm0[setdiff(a,temp.cl[[k]]),temp.clus[[k]][x]])})
      # max_a<-sapply(1:tmp.l,function(x){max(dm0[temp.clus[[k]][x]],temp.cl[[k]])})
      # max_b<-sapply(1:tmp.l,function(x){max(dm0[temp.clus[[k]][x]],setdiff(a,temp.cl[[k]]))})
      larger<-sapply(1:tmp.l,function(x){ifelse((max_a[x]-max_b[x])>0,max_a[x],max_b[x])})
      deflist[[k]]<-(max_a-max_b)/larger
    } 
  }
  return(deflist)
}

#classification steps
DAobs=function(data,temp_clus,method,maxdepth=F,depth,ntrees,K_knn,dm0){
  if (missing(depth)){
    depth<-'spatial'
  }
  if (missing(ntrees)){
    ntrees<-100
  }
  if (missing(K_knn)){
    K_knn<-7
  }
  if (missing(dm0)){
    dm0<-c()
  }
  X<-data
  d<-dim(X)[2]
  Nobs=dim(X)[1]
  Kclus<-length(temp_clus)
  labelledobs<-unlist(temp_clus)
  left.obs<-X[-labelledobs,]
  left.obs.label<-setdiff(1:Nobs,labelledobs)
  N_left=length(left.obs.label)
  if (N_left>0){
    if (method=='maxdep'){
      depth.mat<-matrix(rep(0),nrow(left.obs),Kclus)
      for (i in 1:nrow(left.obs)){
        for (j in 1:Kclus){
          if(depth=='Mahalanobis'){
            depth.mat[i,j]<-depth.Mahalanobis(left.obs[i,],X[temp_clus[[j]],])
          } else if (depth=='spatial'){
            depth.mat[i,j]<-spatial.d(left.obs[i,],X[temp_clus[[j]],])
          }
        }
      }
      depth.order<-sapply(1:nrow(depth.mat),function(i){sort.list(depth.mat[i,],decreasing = T)})
      
      cluster=list()
      for (i in 1:Kclus) {
        cluster[[i]]=left.obs.label[which(depth.order[1,]==i)]
        cluster[[i]]=c(temp_clus[[i]],cluster[[i]])
      }
      cc<-rep(0,Nobs)
      for (i in 1:length(cluster)) {
        cc[cluster[[i]]]<-i
      }
    }   else if (method=='rf'){
      
      current.label<-rep(0,Nobs)
      for (i in 1:length(temp_clus)) {
        current.label[temp_clus[[i]]]<-i
      }
      
      data.rf<-cbind(X[current.label!=0,],current.label[current.label!=0])%>%as.data.frame()
      names(data.rf)[d+1]<-'LABEL'
      
      rf_ntree<-randomForest(as.factor(LABEL)~.,data=data.rf, ntree=ntrees,nodesize=3)
      pred<-predict(rf_ntree,newdata=X[current.label==0,]%>%as.data.frame())
      
      current.label[current.label==0]<-pred
      cc<-current.label
      cluster=list()
      for (i in 1:Kclus) {
        cluster[[i]]=left.obs.label[which(pred==i)]
        cluster[[i]]=c(temp_clus[[i]],cluster[[i]])
      }
    } else if (method=='knn'){
      current.label<-rep(0,nrow(data))
      for (i in 1:length(temp_clus)) {
        current.label[temp_clus[[i]]]<-i
      }
      classes<-current.label[current.label!=0]
      knnc<-KNNdep(Kclus,K_knn,dm0[which(current.label==0),which(current.label!=0)],classes)
      current.label[current.label==0]<-knnc$class
      cc<-current.label
      cluster=list()
      for (i in 1:Kclus) {
        cluster[[i]]=left.obs.label[which(knnc$class==i)]
        cluster[[i]]=c(temp_clus[[i]],cluster[[i]])
      }
    } 
  } else {
    cluster=temp_clus
    cc<-rep(0,Nobs)
    for (i in 1:length(temp_clus)) {
      cc[temp_clus[[i]]]<-i
    }
  }
  
  
  
  if (maxdepth==TRUE){
    nelabel=0
    
    while (length(nelabel)!=0) {
      DD<- matrix(0,dim(X)[1],Kclus) 
      for(i in (1:Kclus)){ 
        if (depth=='Mahalanobis'){
          DD[,i] <- depth.Mahalanobis(X,X[cluster[[i]],])
        } else if (depth=='spatial'){
          DD[,i] <- spatial.d(X,X[cluster[[i]],])
        }
      } 
      DDmat<-sapply(1:nrow(DD),function(i){sort.list(DD[i,],decreasing = T)})
      nelabel<-which(DDmat[1,]!=cc)
      cc<-DDmat[1,]
      if (length(nelabel)!=0){
        for (i in 1:length(cluster)) {
          cluster[[i]]<-which(cc==i)
        }
      }
    }
  }
  return(list(cluster=cluster,cluster_vector=cc))
}
#symmetric a matrix
sym_mat<-function(mat){
  return((mat+t(mat))/2)
}
#Auto_DLCC main function
AUTO_DLCC<-function(ILD_info,dm0,data,class_method='knn',K_knn=7,depth='spatial',strategy=c()){
  #strategy is either linkage or centroid. If missing, empirical method will automatically select the strategy. 
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
  stable_centers_info=find_stable_centers_ld(ILD_mat = ILD_mat,nbr_list,matrix_info = matrix_info,sym_dm,strategy = strategy,dm0.order)
  temp_clus<-get_temp_clus(sym_dm,stable_centers_info$group_list,strategy = stable_centers_info$strategy,data=data,freq_table)
  cluster_result<-DAobs(data,temp_clus$temp_clus, method=class_method,K_knn=K_knn,depth=depth,dm0=dm0)
  return(list(temp_clus=temp_clus$temp_clus,cluster_result=cluster_result,group_list=temp_clus$group_list,stable_centers=temp_clus$stable_centers,strategy=stable_centers_info$strategy))
}

#change cluster results from list to vector form
cluster2cv<-function(data,temp.clus){
  current.label<-rep(0,nrow(data))
  for (i in 1:length(temp.clus)) {
    current.label[temp.clus[[i]]]<-i
  }
  return(current.label)
}
