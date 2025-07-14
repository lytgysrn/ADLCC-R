color_plot<-function(data,cluster_vector,local_center=c()){
  data_new=data%>%as.data.frame()
  colnames(data_new)<-c('V1','V2')
  cluster_vector<-factor(cluster_vector)
  p1<-ggplot(data_new, aes(x = V1, y = V2))+
    geom_point()+geom_point(data = data_new[cluster_vector!=0, ],
                            aes(x = V1, y = V2, color = cluster_vector[cluster_vector!=0]),
                            size = 2, alpha = 0.8)+
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))+guides(color = "none")+theme_void()
  
  if (length(local_center)>0){
    p2<-p1+  geom_point(
      data = data_new[local_center, ],
      aes(x = V1, y = V2, fill = cluster_vector[local_center]), 
      shape = 21, size = 5, color = "black", stroke = 1.5, alpha = 0.95
    )+ guides(color = "none", fill = "none")+theme_void() 
    
    return(list(p1=p1,p2=p2))
  } else {
    return(p1)
  }
}

plot_ILD_row_color<-function(ILD_mat, highest_ld_idx, target_index,nbr_list){
  row_data <- ILD_mat[target_index, ]
  df <- data.frame(
    Position = nbr_list,
    ILD = row_data,
    isCandidate = factor(highest_ld_idx[[target_index]]),
    Beta <- nbr_list / nrow(ILD_mat)
  )
  
  p <- ggplot(df, aes(x = Beta, y = ILD, color = isCandidate)) +
    geom_point(size = 3) +
    labs(title = paste("ILD Values for point", target_index),
         x = "Locality Level",
         y = "ILD Value",
         color = "Locally deep point") +
    theme_minimal()
  print(p)
}

library(cowplot)
combine_graph_generator<-function(p1,p2,p3,label){
  main_plot=plot_grid(p1,p2,p3,labels = c("A-DLCC-centroid", "A-DLCC-linkage", "DLCC-min"),nrow=1,align='h',label_size=14)
  final_plot <- ggdraw() +
    draw_label(label, x = 0.02, y = 0.5, angle = 90, vjust = 0.5, size = 16, fontface = "bold") +
    draw_plot(main_plot, x = 0.06, y = 0, width = 0.94, height = 1) +
    draw_line(
      x = c(0.06 + 0.94/3, 0.06 + 0.94/3),
      y = c(0, 1),
      linetype = 2,
      color = "grey40", size = 1, alpha = 0.3
    ) +
    draw_line(
      x = c(0.06 + 0.94*2/3, 0.06 + 0.94*2/3),
      y = c(0, 1),
      linetype = 2,
      color = "grey40", size = 1, alpha = 0.3
    )
}