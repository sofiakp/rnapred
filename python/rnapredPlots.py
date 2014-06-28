import sys
import os
import os.path
import numpy as np
import pandas as pd
import pandas.rpy.common as com
import rpy2.robjects.lib.ggplot2 as ggplot2
import rpy2.robjects as ro
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances
import pickle
from rnapredCV import read_model, read_cv_res, read_feat_mat

def r2_trained_on_avg_rest(expt_names, feat_mat_dir, model_dir, njobs = 1):
    """Compute for each cell type the average prediction when trained on 
    each of the rest of the cell types.
    
    For each of the specified experiments, it will compute the average 
    prediction of models trained on each of the rest of the experiments,
    and obtain the R-squared with the true values.
    
    Args:
    - expt_names
    - feat_mat_dir: Directory with feature matrices. For each experiment
    in expt_names, it will look for a file <feat_mat_dir>/<expt>_feat_mat.npz.
    - model_dir: Directory with learned models. For each experiment in 
    expt_names, it will look for a file <model_dir>/<expt>_model.pkl. 
    """
    
    models = []
    for ex in expt_names:
        with open(os.path.join(model_dir, ex + '_model.pkl'), 'rb') as infile:
            model = pickle.load(infile)
            #model.set_njobs(njobs)
            models.append(model)
    
    r2 = []
    for ex in expt_names:
        data = np.load(os.path.join(feat_mat_dir, ex + '_feat_mat.npz'))
        feat = data['feat']
        y = data['y']
        data.close()
        pred = np.zeros((y.shape[0],))
        for midx, m in enumerate(models):
            if expt_names[midx] == ex:
                continue
            tmp_pred = m.predict(feat)
            pred += m.predict(feat)
        pred = pred / (len(models) - 1)
        r2.append(metrics.r2_score(y, pred))
    return r2


def r2_trained_on_rest(expt_names, feat_mat_dir, model_dir, njobs = 1):
    """Compute for each cell type the prediction of a model trained on
    the rest of the cell types.
    The model excluding cell type <ex> should be in 
    <model_dir>/merged_no_<ex>_model.pkl.

    Args:
    - expt_names
    - feat_mat_dir: Directory with feature matrices. For each experiment
    in expt_names, it will look for a file <feat_mat_dir>/<expt>_feat_mat.npz.
    - model_dir: Directory with learned models. For each experiment in 
    expt_names, it will look for a file <model_dir>/merged_no_<expt>_model.pkl. 
    """
    
    r2 = []
    for ex in expt_names:
        with open(os.path.join(model_dir, 'merged_no_' + ex + '_model.pkl'), 'rb') as infile:
            model = pickle.load(infile)
            #model.set_njobs(njobs)
        data = np.load(os.path.join(feat_mat_dir, ex + '_feat_mat.npz'))
        feat = data['feat']
        y = data['y']
        data.close()
        r2.append(metrics.r2_score(y, model.predict(feat)))
    return r2


def r2_trained_on_most_corr(expt_names, feat_mat_dir, model_dir, njobs = 1):
    """Compute for each cell type the prediction of the model trained
    on the cell type with the most correlated promoter H3K4me3.

    Args:
    - expt_names
    - feat_mat_dir: Directory with feature matrices. For each experiment
    in expt_names, it will look for a file <feat_mat_dir>/<expt>_feat_mat.npz.
    - model_dir: Directory with learned models. For each experiment in 
    expt_names, it will look for a file <model_dir>/<expt>_model.pkl. 
    """

    # Compute correlations between cell types
    array_expr = None
    for ex in expt_names:
        data = np.load(os.path.join(feat_mat_dir, ex + '_feat_mat.npz'))
        feat = data['feat']
        ex_ind = np.argwhere(np.array(data['feat_names']) == 'H3K4me3.firstTSS.1k')[0][0]
        if array_expr is None:
            array_expr = np.reshape(feat[:, ex_ind], (feat.shape[0], 1))
        else:
            array_expr = np.concatenate((array_expr, np.reshape(feat[:, ex_ind], (feat.shape[0], 1))), axis = 1)
        data.close()
        
    cor = 1 - pairwise_distances(array_expr.T, metric = 'correlation')
    np.fill_diagonal(cor, 0)
    
    # Load all models
    models = []
    for ex in expt_names:
        with open(os.path.join(model_dir, ex + '_model.pkl'), 'rb') as infile:
            model = pickle.load(infile)
            #model.set_njobs(njobs)
            models.append(model)
    
    r2 = []
    # For each cell type, get the most correlated one and use that
    # for making predictions.
    for expt_ind, ex in enumerate(expt_names):
        data = np.load(os.path.join(feat_mat_dir, ex + '_feat_mat.npz'))
        feat = data['feat']
        train_ind = np.argmax(cor[expt_ind, :])
        r2.append(metrics.r2_score(data['y'], models[train_ind].predict(feat)))
        data.close()
        
    return r2


def plot_cv_r2(pandas_df, outfile, fsize = 10, height = 120, max_width = 50, xlab = 'Parameters'):
    """Makes boxplots of cross-validation results for different parameter settings"""

    ncv = len(set(list(pandas_df['title'])))
    r_df = com.convert_to_r_dataframe(pandas_df)
    
    gp = ggplot2.ggplot(r_df) + ggplot2.aes_string(x = 'factor(title)', y = 'r2') + \
        ggplot2.geom_boxplot() + ggplot2.scale_y_continuous('R-squared') + \
        ggplot2.scale_x_discrete(xlab) + ggplot2.theme_bw() + \
        ggplot2.theme(**{'axis.text.x':ggplot2.element_text(size = fsize, angle = 65, vjust = 1, hjust = 1),
                         'axis.text.y':ggplot2.element_text(size = fsize)})
    w = max(5 * ncv, max_width) 
    ro.r.ggsave(filename = outfile, plot = gp, width = w, height = height, unit = 'mm')


def plot_coef(feat_mat_dir, model_dir, expt_names, pref, outfile = None, height = 120, fsize = 12):
    
    for expt_idx, ex in enumerate(expt_names):
        feat_mat_file = os.path.join(feat_mat_dir, ex + '_feat_mat.npz')
        model_file = os.path.join(model_dir, pref + ex + '_model.pkl')
        model = read_model(model_file)
        (tmp_feat, tmp_y, tmp_feat_names, tmp_gene_names) = read_feat_mat(feat_mat_file)
        
        if expt_idx == 0:
            feat_names = tmp_feat_names
            clf_coef = model.clf_coef()
            reg_coef = model.reg_coef()
        else:
            assert(all(f[0] == f[1] for f in zip(feat_names, tmp_feat_names)))
            clf_coef = np.concatenate((clf_coef, model.clf_coef()), axis = 1)
            reg_coef = np.concatenate((reg_coef, model.reg_coef()), axis = 1)
    
    nexpt = expt_idx + 1
    
    # Now clf_coef has one row per coefficient and one column per experiment.
    # The reshape below will read the data row-first.
    df = pd.DataFrame({'feature':np.repeat(feat_names, nexpt),
                       'Classification':np.reshape(clf_coef, (clf_coef.size,)),
                       'Regression':np.reshape(reg_coef, (reg_coef.size,))})

    df2 = pd.melt(df, id_vars = 'feature', var_name = 'fun')
    r_df = com.convert_to_r_dataframe(df2)
    gp = ggplot2.ggplot(r_df) + ggplot2.aes_string(x = 'factor(feature)', y = 'value') + \
        ggplot2.facet_wrap('fun', scales = 'free_y') + \
        ggplot2.geom_boxplot() + ggplot2.scale_y_continuous('Importance') + \
        ggplot2.scale_x_discrete('') + ggplot2.theme_bw() + \
        ggplot2.theme(**{'axis.text.x':ggplot2.element_text(size = fsize, angle = 65, vjust = 1, hjust = 1),
                         'axis.text.y':ggplot2.element_text(size = fsize),
                         'strip.text.x':ggplot2.element_text(size = fsize + 1)})
    w = max(22 * nexpt, 80)
    if outfile is None:
        gp.plot()
    else:
        ro.r.ggsave(filename = outfile, plot = gp, width = w, height = height, unit = 'mm')
    return df


def r2_per_tree_size(model, feat, y, ntrees):
    """Compute R-squared on the given data for reduced versions of a RF-based TwoStepRegressor.
    
    Args:
    - model: A TwoStepRegressor
    - feat, y: Data on which to make predictions and true output.
    - ntrees: Number of trees to consider. Values smaller than the number of trees
    in the input model will be ignored.
    
    Return value:
    A tuple (ntrees, r2) where r2[i] is the R-squared when considering only the first
    ntrees[i] trees of the RFs of the TwoStepRegressor.
    """
    ntrees = [n for n in ntrees if n <= min(model.model1.n_estimators, model.model2.n_estimators)]
    r2 = []
    for n in ntrees:
        model.model1.n_estimators = n
        model.model2.n_estimators = n
        r2.append(metrics.r2_score(y, model.predict(feat)))
    return (ntrees, r2)
