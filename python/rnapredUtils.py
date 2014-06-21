import sys
from numpyutils import *
import numpy as np
import os
import os.path

def add_feat(feat, idx, filename, gene_name_idx):
    """Reads signal from a file and adds it to the given column of feat.
    
    The input file should be BED-like. The 4th column should be of the form
    gene_i, where gene is the name of a gene. The _i part is ignored. 
    This column should be sorted by gene. The 5th column should contain the 
    average signal in the corresponding region. The signal for a gene will
    then be computed as the average across all regions corresponding to the
    gene.
    """
    
    with open(filename, 'r') as infile:
        last_gene = None
        last_bp = 0
        last_sig = 0.0
        
        for line in infile:
            fields = line.strip().split()
            gene = '_'.join(fields[3].split('_')[:-1])
            if not gene in gene_name_idx:
                # Ignore genes not in the gene expression matrix.
                continue
            
            start = int(fields[1])
            end = int(fields[2])
            # Total signal in the region
            sig = float(fields[4]) * (end - start) 
            
            if gene != last_gene and last_gene != None:
                feat[gene_name_idx[last_gene], idx] = last_sig / last_bp
                last_sig = sig
                last_bp = end - start
            else:
                last_sig += sig
                last_bp += (end - start)
            last_gene = gene
    
    if last_gene != None:
        feat[gene_name_idx[last_gene], idx] = last_sig / last_bp


def read_gene_len(gene_file, gene_name_idx):
    """Reads a BED file and extracts information about the length of genes."""
    
    gene_len = np.zeros((len(gene_name_idx), 1))
    
    with open(gene_file, 'r') as infile:
        for line in infile:
            fields = line.strip().split()
            gene = fields[6]
            if gene in gene_name_idx:
                start = int(fields[1])
                end = int(fields[2])
                gene_len[gene_name_idx[gene]] = end - start
            
    if np.any(gene_len == 0):
        raise ValueError('Some genes in the expression matrix are missing from the gene file')
    return gene_len


def get_gc_cont(gene_file, gene_gc_file, gene_name_idx):
    """Reads a file with CpG content information and computes the 
    normalized CpG content for each gene.
    
    Args:
    - gene_file: BED file with genes. 4th column should match the first
    column of the gene_gc_file. However, the 7th column is the actual gene
    name used.
    - gene_gc_file: File with the format of homerTools freq.
    """
    
    gc_cont = np.zeros((len(gene_name_idx), 1))
    
    # Map each gene id to its normalized CpG content.
    gc_map = {}
    with open(gene_gc_file, 'r') as infile:
        for line in infile:
            fields = line.strip().split()
            gene = fields[0]
            cpg = float(fields[1])
            gc = float(fields[2]) * 0.5
            gc_map[gene] = cpg / gc
    
    # Now map each gene name to its normalized CpG content.
    with open(gene_file, 'r') as infile:
        for line in infile:
            fields = line.strip().split()
            gene_name = fields[6]
            gene = fields[3]
            
            if gene_name in gene_name_idx and gene in gc_map:
                gc_cont[gene_name_idx[gene_name]] = gc_map[gene]
            
    return gc_cont


def read_exon_array(dat_file, gene_name_idx):
    """Reads a matrix of expression values from exon arrays and selects 
    a subset of genes.
    
    Args:
    - dat_file: npz file. Should contain an expression matrix (expr), 
    where rows are genes and columns are conditions, a 
    list of gene names (gene_names), and a list of condition names
    (expt_names).
    - gene_name_idx[g] is the index of gene g in the output expression
    matrix.
    """
    
    data = np.load(dat_file)
    expr = data['expr']
    gene_names = data['gene_names']
    expt_names = data['expt_names']
    
    expr_out = np.zeros((len(gene_name_idx), len(expt_names)))
    for i, g in enumerate(gene_names):
        if g in gene_name_idx:
            expr_out[gene_name_idx[g], :] = expr[i, :]
    return (expr_out, expt_names)


def read_rna_pred_data(expr_file, gene_name_file, train_expt_name_file, 
                       gc_cont_file, exon_array_file,
                       gene_file, expt_name_file, feat_names, signal_dirs,
                       out_feat_names,
                       out_dir, overwrite = False):
    """Read data for RNA-seq level prediction and creates a feature matrix for each cell line.
    
    Args:
    - expr_file: path to file with RNA-seq levels (for the training cell lines). It contains a matrix
    genes x cell lines.
    - gene_name_file: path to file with the names of the genes corresponding to the rows of the 
    above matrix.
    - train_expt_name_file: path to file with the names of the experiments in the RNA-seq expression
    matrix (i.e. training cell lines).
    - gc_cont_file: File with gc content (format as in HOMER's freq tool).
    - exon_array_file: Path to npz file with exon array data.
    - gene_file: File from where gene length information will be read.
    - expt_name_file: Path to file listing all the cell lines (training and testing).
    - out_dir:
    - feat_names, signal_dirs: For each cell line C, we will look for files 
    signal_dirs[i]/C-feat_names[i].bed
    - out_feat_names: Feature names to store in the output files.
    
    Return value:
    A tuple with the following:
    - gene_names
    - expt_names
    - all_expt_names: Training and testing experiment names
    """
       
    expr = np.loadtxt(expr_file, delimiter = '\t')
    
    with open(train_expt_name_file, 'r') as infile:
        expt_names = np.array([line.strip() for line in infile.readlines()])
    short_expt_names = [ex.split('_')[0] for ex in expt_names]
    nexpt = len(expt_names)
    assert(expr.shape[1] == nexpt)

    with open(gene_name_file, 'r') as infile:
        gene_names = np.array([line.strip().split()[0] for line in infile.readlines()])
    ngenes = len(gene_names)
    assert(expr.shape[0] == ngenes)
    assert(len(set(gene_names)) == ngenes) # Gene names should be unique
    
    # Map each gene name to its index in the expression table, so you can easily
    # reorder other datasets (like signal data) to match the order of the genes
    # in the gene expression matrix.
    gene_name_idx = {}
    for i, g in enumerate(gene_names):
        gene_name_idx[g] = i
    
    gene_len = read_gene_len(gene_file, gene_name_idx)
    (exon_expr, exon_expt_names) = read_exon_array(exon_array_file, gene_name_idx)
    gc_cont = get_gc_cont(gene_file, gc_cont_file, gene_name_idx)
    
    with open(expt_name_file, 'r') as infile:
        all_expt_names = np.array([line.strip() for line in infile.readlines()])
    assert(all([s in all_expt_names for s in short_expt_names]))
    
    out_feat_names.extend(['gene_len', 'array_expr', 'CpG'])
    nfeat = len(feat_names)
    
    ntrain = 0
    ntest = 0
    
    for ex in all_expt_names:
        exon_idx = np.argwhere(np.array(exon_expt_names) == ex)
        if len(exon_idx) == 0:
            print >> sys.stderr, 'Skipping', ex, '- Exon expression missing.'
            continue
        else:
            exon_idx = exon_idx[0][0]
            
        outfile = os.path.join(out_dir, ex + '_feat_mat.npz')
        if not overwrite and os.path.exists(outfile):
            continue
        
        filenames = [os.path.join(signal_dirs[i], ex + '-' + s + '.bed') for i, s in enumerate(feat_names)]
        if any([not os.path.exists(f) for f in filenames]): 
            print >> sys.stderr, 'Skipping', ex, '- Signal files missing.'
            continue
        
        feat = np.zeros((ngenes, nfeat))
        
        for idx, f in enumerate(filenames):
            add_feat(feat, idx, f, gene_name_idx)
    
        # Max-normalize the histone signal of each experiment.
        feat = feat / np.reshape(np.max(feat, axis = 0), (1, feat.shape[1]))
        feat = np.concatenate((feat, gene_len), axis = 1)
        feat = np.concatenate((feat, np.reshape(exon_expr[:, exon_idx], (ngenes, 1))), axis = 1)
        feat = np.concatenate((feat, gc_cont), axis = 1)
        
        if ex in short_expt_names:
            # Training condition
            ex_idx = np.argwhere(np.array(short_expt_names) == ex)[0][0]
            np.savez(outfile, feat = feat, gene_names = gene_names, 
                     feat_names = out_feat_names, y = np.arcsinh(expr[:, ex_idx]))
            ntrain += 1
        else:
            # Purely testing condition
            np.savez(outfile, feat = feat, gene_names = gene_names, 
                     feat_names = out_feat_names)
            ntest += 1
    
    print >> sys.stderr, 'Read', ntrain, 'training and', ntest, 'testing conditions.'        
    return (gene_names, short_expt_names, all_expt_names)
