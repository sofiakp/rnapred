
#!/bin/bash                                                                                                                                    
                                                                                                                                                

library_anno="/srv/gs1/projects/snyder/oursu/my_programs/apt-1.15.1-x86_64-intel-linux/Human_Exon_1.0_ST_Analysis/"
o="output"


/srv/gs1/projects/snyder/oursu/my_programs/apt-1.15.1-x86_64-intel-linux/bin/apt-probeset-summarize -a rma-sketch -p ${library_anno}HuEx-1_0-st
-v2.r2.pgf -c ${library_anno}HuEx-1_0-st-v2.r2.clf -m ${library_anno}HuEx-1_0-st-v2.r2.dt1.hg18.core.mps -qc-probesets ${library_anno}HuEx-1_0-
st-v2.r2.qcc -o ${o} /srv/gsfs0/projects/kundaje/users/sofiakp/roadmap/data/rna/exon_arrays/cel/*.CEL
