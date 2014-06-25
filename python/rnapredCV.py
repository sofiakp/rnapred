import sys
import os
import os.path
import pandas as pd
import argparse
import numpy as np
from multiprocessing import Pool
from sklearn import linear_model
from sklearn import ensemble
from sklearn import cross_validation
from sklearn import metrics
#import rpy2.robjects.lib.ggplot2 as ggplot2
#import rpy2.robjects as ro
#from rpy2.robjects.packages import importr

class TwoStepRegressor():
    """Base class for a two-step regressor.
    
    A two-step regressor is a product of two models. A classification
    model (model1) that predicts whether a sample is > 0 or not and 
    a regression model (model2) trained on non-negative samples.
    The prediction for a new example is the product of the predictions
    of the two models.
    """
    
    def fit(self, X, y):
        ybin = np.array(y > 0, dtype = np.int)
        self.model1.fit(X, ybin)
        reg_ex = y > 0
        self.model2.fit(X[reg_ex, :], y[reg_ex])
        
    def predict(self, X):
        return self.model1.predict(X) * self.model2.predict(X)

    def clf_coef(self):
        """Return coefficients for the classification model.
        The meaning of these coefficients depends on the type of model.
        """
        return model1.coef_
    
    def reg_coef(self):
        return model2.coef_

    
class RFClassifierRFRegressor(TwoStepRegressor):
    def __init__(self, args_clf, args_reg):
        self.model1 = ensemble.RandomForestClassifier(**args_clf)
        self.model2 = ensemble.RandomForestRegressor(**args_reg)
     
    def clf_coef(self):
        return model1.feature_importances_
    
    def reg_coef(self):
        return model2.feature_importances_

    
class LogClassifierRidgeRegressor(TwoStepRegressor):
    def __init__(self, args_clf, args_reg):
        self.model1 = linear_model.LogisticRegression(**args_clf)
        self.model2 = linear_model.Ridge(**args_reg)


def cross_validate(cv, class_name, args_clf, args_reg, X, y):
    """Performs cross-validation of a TwoStepRegressor.
    
    Args:
    - cv: The output of a cross-validation generator (eg. ShuffleSplit).
    - class_name: The type of TwoStepRegressor.
    - args_clf: Dictionary of arguments for the classifier of the TwoStepRegressor.
    - args_reg: Dictionary of arguments for the regressor of the TwoStepRegressor.
    - X, y: feature matrix and output vector.
    
    Return value:
    R-squared across folds
    """
    
    r2 = []
    for train, test in cv:
        model = class_name(args_clf, args_reg)
        model.fit(X[train, :], y[train, :])
        pred = model.predict(X[test, :])
        r2.append(metrics.r2_score(y[test], pred))
    return r2


def cross_validate_star(args):
    return cross_validate(*args)


def cross_validate_grid(cv, class_name, args_clf_list, args_reg_list, X, y, 
                        zip_params = False, nproc = 8):
    """Perform cross-validation of a TwoStepRegressor in a grid of parameters.
    
    Args:
    - As in cross_validate. args_clf_list and args_reg_list are lists of dictionaries.
    It will perform one cross validation run for each combination of arguments.
    
    Return value:
    A list of tuples of the form (args_clf, args_reg, r2).
    """
    
    # Create parameter pairs
    param_pairs = []
    if zip_params:
        for args_clf, args_reg in zip(args_clf_list, args_reg_list):
            param_pairs.append((args_clf, args_reg))
    else:
        for args_clf in args_clf_list:
            for args_reg in args_reg_list:
                param_pairs.append((args_clf, args_reg))
                
    if nproc > 1:
        pool = Pool(nproc)
        tasks = []
        for param_pair in param_pairs:
            tasks.append((cv, class_name, param_pair[0], param_pair[1], X, y))
        r2 = pool.imap(cross_validate_star, tasks)
        pool.close()
        pool.join()
        r2 = [r for r in r2]
        
        # Create the tuples for the output. The results in r2 are 
        # ordered in the same order in which we created the models.
        res = []
        for param_pair in param_pairs:
            res.append((param_pair[0], param_pair[1], r2[len(res)]))
    else:
        res = []
        for param_pair in param_pairs:
            res.append((param_pair[0], param_pair[1],
                        cross_validate(cv, class_name, param_pair[0], param_pair[1], X, y)))
    return res


def get_forest_params(ntrees_vals, min_leaf_vals):
    """Creates cross validation parameters for a RandomForestClassifier or Regressor.
    One dictionary of parameters is created for each combination of the input arguments.
    """
    
    forest_params = {'max_features':'auto'}
    param_list = []
    for t in ntrees_vals:
        for n in min_leaf_vals:
            new_params = dict(forest_params)
            new_params['n_estimators'] = t
            new_params['min_samples_leaf'] = n
            param_list.append(new_params)
    return param_list


def forest_cv_to_pandas_df(res, cv = 10):
    clf_n_trees = []
    reg_n_trees = []
    clf_min_samples_leaf = []
    reg_min_samples_leaf = []
    r2 = []
    for r in res:
        clf_n_trees.extend([r[0]['n_estimators']] * cv)
        reg_n_trees.extend([r[1]['n_estimators']] * cv)
        clf_min_samples_leaf.extend([r[0]['min_samples_leaf']] * cv)
        reg_min_samples_leaf.extend([r[1]['min_samples_leaf']] * cv)
        r2.extend(r[2])
    
    df = pd.DataFrame({'clf.n.trees':clf_n_trees, 'reg.n.trees':reg_n_trees,
                       'clf.min.samples':clf_min_samples_leaf, 
                       'reg.min.samples':reg_min_samples_leaf,
                       'r2':r2})
    df['title'] = ['t{:d}-s{:d}/t{:d}-s{:d}'.format(s[0], s[1], s[2], s[3]) 
               for s in zip(df['clf.n.trees'], df['clf.min.samples'], df['reg.n.trees'], df['reg.min.samples'])]
    return df


def logLass_cv_to_pandas_df(res, cv = 10):
    logC = []
    ridgeA = []
    r2 = []
    for r in res:
        logC.extend([r[0]['C']] * cv)
        ridgeA.extend([r[1]['alpha']] * cv)
        r2.extend(r[2])
        
    df = pd.DataFrame({'C':logC, 'alpha':ridgeA, 'r2':r2})
    df['title'] = ['C{:g}-alpha{:g}'.format(s[0], s[1]) 
                   for s in zip(df['C'], df['alpha'])]
    return df


def concatenate_expt_feat_mat(infiles):
    feat = None
    y = None
    for infile in infiles:
        data = np.load(infile)
        if not 'y' in data:
            continue
        tmp_feat = data['feat']        
        tmp_y = data['y']
        tmp_y = tmp_y / np.max(tmp_y)
        tmp_feat_names = data['feat_names']
        ex_ind = np.argwhere(np.array(data['feat_names']) == 'array_expr')[0][0]
        tmp_feat[:, ex_ind] = tmp_feat[:, ex_ind] / np.max(tmp_feat[:, ex_ind])
        if feat is None:
            feat = tmp_feat
            y = tmp_y
            feat_names = tmp_feat_names
        else:
            feat = np.concatenate((feat, tmp_feat), axis = 0)
            y = np.concatenate((y, tmp_y), axis = 0)
            assert(len(feat_names) == len(tmp_feat_names))
            assert(all(f[0] == f[1] for f in zip(feat_names, tmp_feat_names)))
    return (feat, y)


def main():
    desc = """Performs cross-validation of a TwoStepRegressor in a grid of parameters."""
    parser = argparse.ArgumentParser(description = desc)
    parser.add_argument('infile', help = 'Input file or directory. If it is a directory, files will be concatenated')
    parser.add_argument('outfile', help = 'Path to output npz file')
    parser.add_argument('--method', '-m', default = 'rf', help = 'One of logLasso or rf')
    parser.add_argument('--ntrees', default = '10', 
                        help = 'Comma separated list of values for the number of trees')
    parser.add_argument('--leaf', default = '1',
                        help = 'Comma separated list of values for the min number of examples per leaf')
    parser.add_argument('--alphas', default = '1',
                        help = 'Comma separated list of values for the alpha parameter of Lasso regression')
    parser.add_argument('--cs', default = '1', 
                        help = 'Comma separated list of values for the C parameter of logistic regression')
    parser.add_argument('--nproc', type = int, default = 8, help = 'Number of processors to use')
    parser.add_argument('--nocv', action = 'store_true', default = False,
                        help = 'No cross-validation. Just run and return the model.')
    args = parser.parse_args()
    method = args.method
    nproc = args.nproc

    if os.path.isfile(args.infile):
        data = np.load(args.infile)
        if not 'y' in data:
            warning('File ' + args.infile + ' does not have y vector. Exiting.')
            return
        feat = data['feat']
        y = data['y']
        data.close()
    else:
        files = [os.path.join(args.infile, f) for f in os.listdir(args.infile)]
        files = [f for f in files if (os.path.exists(f) and not os.path.isdir(f) and f.endswith('_feat_mat.npz'))]
        (feat, y) = concatenate_expt_feat_mat(files)

    cv = cross_validation.ShuffleSplit(feat.shape[0], n_iter = 10, test_size = 0.1, random_state = 0) 
    if method == 'rf':
        ntrees_vals = [int(s) for s in args.ntrees.split(',')]
        min_leaf_vals = [int(s) for s in args.leaf.split(',')]
        params = get_forest_params(ntrees_vals, min_leaf_vals)
        clf_params = []
        for p in params:
            newp = dict(p)
            newp['criterion'] = 'entropy'
            clf_params.append(newp)

        if args.nocv:
            class_name = RFClassifierRFRegressor
        else:
            cv_res = cross_validate_grid(cv, RFClassifierRFRegressor, clf_params, params, feat, y, 
                                         zip_params = False, nproc = nproc)
    elif method == 'logLasso':
        clf_params = [{'penalty':'l2', 'C':float(c)} for c in args.cs.split(',')]
        params = [{'alpha':float(a)} for a in args.alphas.split(',')]
        if args.nocv:
            class_name = LogClassifierRidgeRegressor
        else:
            cv_res = cross_validate_grid(cv, LogClassifierRidgeRegressor, clf_params, params,
                                         feat, y, zip_params = False, nproc = nproc)

    if args.nocv:
        assert(len(clf_params) == 1)
        assert(len(params) == 1)
        model = class_name(clf_params[0], params[0])
        model.fit(feat, y)
        np.savez(args.outfile, model = model)
    else:
        np.savez(args.outfile, cv_res = cv_res)


if __name__ == '__main__':
    main()
