library(calibrate)
library(plotrix)

png(file="MAD.png",width=10000, height=9800)         
qc<-read.table("RCodes/rma-sketch.report.txt",sep="\t",header=T)
par(mai=c(90,5,5,5))   
plot(1:159,qc$all_probeset_mad_residual_mean, ylim=c(0,0.7), xlab="", ylab="", main="Deviation of Residual from Median", cex.axis=8, cex.main=10, pch=19,type="p",ps=400,cex=10)

#axis(1, at = seq(1,159, by = 1), las=2)
mtext(side = 1, text = "Array", line = 12, cex=10)
mtext(side = 2, text = "Mean absolute deviation", line = 12, cex=10)
textxy(1:159,qc$all_probeset_mad_residual_mean,1:159,cex=8)
seq1 <- seq(1:159)
mat1 <- matrix(seq1,nrow=159,ncol=1)
mat2 <- cbind(qc,mat1)
colnames(mat2)[76]<-"index"
legend("bottom", legend=paste(mat2$index,mat2$cel_files,sep=" : "),xpd = NA, inset = c(0,-2.1),bty="n",ncol=3,pch=12,cex=7.5)


dev.off()                                                             
