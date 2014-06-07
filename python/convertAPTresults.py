import sys

annot_file = '/srv/gs1/projects/snyder/oursu/my_programs/apt-1.15.1-x86_64-intel-linux/Human_Exon_1.0_ST_Analysis/HuEx-1_0-st-v2.na33.1.hg19.transcript.csv'

with open(annot_file, 'r') as infile:
    for line in infile:
        if line[0] == '#':
            continue
        fields = [s.strip('"') for s in line.strip().split(',')]
        transcript_id = fields[0]
        gene_info = fields[7].split('///')
        gene_ids = []
        for g in gene_info:
            if len(g.split('//')) > 1:
                gene_ids.append(g.split('//')[1])
        print transcript_id, gene_ids
