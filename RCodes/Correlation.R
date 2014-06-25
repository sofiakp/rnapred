map<-read.table("Desktop/cel_name_map.txt",header=T)
CELFiles<-read.table("Desktop/RCodes/rma-sketch.summary.txt",header=T)
colnames(map)[1]<-"cel_name"
colnames(map)[2]<-"type"

type<-map$type
typeU<-unique(type)

temp<-matrix( ,nrow=length(typeU),ncol=400)

for( i in typeU)
{
  if('-' %in% typeU){
      print("find '-'")
     typeU<-typeU[-(match('-',typeU))]
  }
}

#Finding CEL files with the same type
print("Start finding CEL files with the same type")
for( k in 1:length(typeU)){
   index=1
   for( i in 1:nrow(map)){
      for( j in i+1:nrow(map)){
         if( j< 159){
           if((map[i,2]==typeU[k]) & (map[j,2]==typeU[k])){
              temp[k,index]<-toString(map[i,1])
              index<-index+1
              temp[k,index]<-toString(map[j,1])
              index<-index+1

           }
         }
      }    
   }
}

print("End of finding CEL files with the same type")

#remove .CEL from the end of cel file names
 for( m in 2:ncol(CELFiles)){
     string<-gsub(".CEL","", colnames(CELFiles)[m])
     colnames(CELFiles)[m]<-toString(string)
}

# Create a matrix to save all the computed correaltions
cols<-colnames(CELFiles)
cor<-matrix( ,nrow=159,ncol=159)
rownames(cor) <- cols[2:160]
colnames(cor) <- cols[2:160]
 
#Computing correlations
print("Start compuitng correlations")
cat("    CEL_name","CEL_name","Correlation",file="Desktop/Correlations.txt",sep="\t\t\t\t\t\t")
cat(file="Desktop/Correlations.txt",sep="\n",append=TRUE)

nplot<-1
for ( m in 1:nrow(temp))
{
    if(m<=nrow(temp)){
      U<-unique(temp[m, ])
       for(j in 1:length(U)){
          for(k in j+1:length(U)){
               if(k<=length(U)){
                  vector1<-toString(U[j])
                 
                  vector2<-toString(U[k])
                  
                  for(c in 2:ncol(CELFiles)){
                      
                     if(toString(colnames(CELFiles)[c])==vector1){
                        
                         
                           for(d in 1:ncol(CELFiles)){
                              if(toString(colnames(CELFiles)[d])==vector2 & d<=ncol(CELFiles)){
                               
                                 corr<-cor(CELFiles[,c],CELFiles[,d])
                                     
                                    for( Rindex in 1:nrow(cor)){
                                           if(rownames(cor)[Rindex]==vector1){
                                             for(Cindex in 1:ncol(cor)){
                                                 if(colnames(cor)[Cindex]==vector2){
                                                    cor[Rindex,Cindex]<-corr
                                                    cat(colnames(CELFiles)[c],colnames(CELFiles)[d],corr,file="Desktop/Correlations.txt",sep="\t",append=TRUE)
                                                    cat(file="Desktop/Correlations.txt",sep="\n",append=TRUE)
                                                    
                                                     mypath <- file.path("Desktop",paste("plot_",nplot, ".jpg", sep = ""))
                                                     nplot<-nplot+1
                                                     png(file=mypath,width=2000, height=1000) 
                                                     par(mai=c(3,3,3,3)) 
                                                     plot(CELFiles[ ,d],CELFiles[ ,c], xlab="", ylab="",cex=1, main=paste("The correlation value is: ",toString(corr),sep = ""),cex.axis=2,cex.main=2)
                                                     mtext(side = 1, text = colnames(CELFiles)[c], line = 5, cex=2)
                                                     mtext(side = 2, text = colnames(CELFiles)[d], line = 5, cex=2)
                                                     dev.off()

                                                 }
                                        }
                                        }
                                    }
                              }
                           }
                       }
                  }
               }
          }
      } 
  
}
}

print("End of compuitng correlations")



