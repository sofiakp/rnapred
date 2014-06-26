#!/bin/bash

# Gets FASTA sequence from a BED file
#/srv/gsfs0/projects/kundaje/users/tsdavis/bin/homer/bin/homerTools extract /srv/gsfs0/projects/kundaje/users/sofiakp/data/gencode/v19/gencode.v19.annotation.noM.genes.firstTSS.1k.bed /srv/gsfs0/projects/kundaje/commonRepository/genomicSequence/encodeHg19Male -fa > /srv/gsfs0/projects/kundaje/users/sofiakp/data/gencode/v19/gencode.v19.annotation.noM.genes.firstTSS.1k.fa
/srv/gsfs0/projects/kundaje/users/tsdavis/bin/homer/bin/homerTools freq -format fasta -gc /srv/gsfs0/projects/kundaje/users/sofiakp/data/gencode/v19/gencode.v19.annotation.noM.genes.firstTSS.1k.gc -o /srv/gsfs0/projects/kundaje/users/sofiakp/data/gencode/v19/gencode.v19.annotation.noM.genes.firstTSS.1k_tmp.gc /srv/gsfs0/projects/kundaje/users/sofiakp/data/gencode/v19/gencode.v19.annotation.noM.genes.firstTSS.1k.fa

rm /srv/gsfs0/projects/kundaje/users/sofiakp/data/gencode/v19/gencode.v19.annotation.noM.genes.firstTSS.1k_tmp.gc