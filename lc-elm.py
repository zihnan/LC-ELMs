import argparse
import itertools
import json
import numpy as np
import random
import sys
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble.base import _partition_estimators
from sklearn.externals import joblib
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state, check_X_y, check_array, column_or_1d
from sklearn.utils.validation import check_is_fitted
from sklearn_extensions.extreme_learning_machines import ELMClassifier

__all__ = ["LCELMClassifier"]

                                              
class LCELMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=25, n_hidden=20, alpha=0.5, rbf_width=1.0,
                 activation_func='sigmoid', activation_args=None,
                 user_components=None, regressor=None,
                 binarizer=LabelBinarizer(-1, 1),
                 random_state=None, n_jobs=1, verbose=False):
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.random_state = random_state
        self.activation_func = activation_func
        self.activation_args = activation_args
        self.user_components = user_components
        self.rbf_width = rbf_width
        self.regressor = regressor
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.binarizer = binarizer
        self.verbose = verbose
        self._initialize_estimators()
        
    def _initialize_estimators(self):
        self.estimators_ = [ELMClassifier(n_hidden=self.n_hidden, alpha=self.alpha, rbf_width=self.rbf_width,
                 activation_func=self.activation_func, activation_args=self.activation_args,
                 user_components=self.user_components, regressor=self.regressor,
                 binarizer=self.binarizer,
                 random_state=self.random_state) for _ in range(self.n_estimators)]
        self.estimators_features_ = []
        
    def fit(self, X, y):
        
        n_samples, self.n_features_ = X.shape
        
        # train En estimators
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        y_bin = self.binarizer.fit_transform(y)
        self._initialize_estimators()
        for index, estimator in enumerate(self.estimators_):
            estimator.fit(X, y_bin)
            #estimator.fit(X, y)
        f = self._f(X)
        f_plus = np.linalg.pinv(f)
        self.v_ = f_plus.dot(y)
        return self
    
    def _f(self, X):    
        # calculate F(X)
        f = [[0] * self.n_estimators for _ in range(len(X))]
        for index1, data in enumerate(X):
            for index2, elm in enumerate(self.estimators_):
                try:
                    f[index1][index2] = elm.predict(np.array([data])).tolist()[0]
                except ValueError as e:
                    print data
                    print data.shape
        return np.array(f)
        
    def predict(self, X):
        check_is_fitted(self, 'classes_')
        X = check_array(X, accept_sparse=['csr', 'csc'])
        n_sample, n_features = X.shape
        
        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))
        '''
        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)
        all_decisions = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_decision_function)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X)
            for i in range(n_jobs))
        print 'predict_proba>>>>>>begin'
        print all_decisions
        print 'predict_proba>>>>>>end'
        '''
        f = self._f(X)
        result = f.dot(self.v_).reshape((n_sample,-1))
        return self.binarizer.inverse_transform(np.sign(result))
        '''
        predicted_probabilitiy = np.sign(self.predict_proba(X))
        '''
        '''
        class_predictions = self.binarizer.inverse_transform(predicted_probabilitiy)
        return class_predictions
        '''
        #return predicted_probabilitiy > 0
    """
    def predict_proba(self, X):
        check_is_fitted(self, 'classes_')
        X = check_array(X, accept_sparse=['csr', 'csc'])
        n_sample, n_features = X.shape
        
        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))
        '''
        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)
        all_decisions = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_decision_function)(
                self.estimators_[starts[i]:starts[i + 1]],
                self.estimators_features_[starts[i]:starts[i + 1]],
                X)
            for i in range(n_jobs))
        print 'predict_proba>>>>>>begin'
        print all_decisions
        print 'predict_proba>>>>>>end'
        '''
        f = self._f(X)
        result = f.dot(self.v_).reshape((n_sample,-1))
        return result
    """
    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))

def _h(data, elms, ensemble_size):
    f = [[0] * ensemble_size for _ in range(len(data))]
    for index1, data in enumerate(data):
        for index2, elm in enumerate(elms):
            f[index1][index2] = elm.predict(np.array([data])).tolist()[0]
    return f

def get_data(file_path, length=-1, percent=1.0, enable_shuffle=False):
    temp = []
    with open(file_path, 'r') as f:
        json_list = f.readlines()
        for row, index in itertools.izip(json_list, range(len(json_list))):
            if length > 0 and index > length - 1:
                break
            if len(row.rstrip()) > 0:
                json_text = json.loads(row.rstrip())
                if isinstance(json_text[-4], list):
                    temp.append(json_text[:-4] + json_text[-3:])
                else:
                    temp.append(json_text)
    if enable_shuffle:
        random.shuffle(temp)
    print 'Get Data Size: {size}'.format(size=int(len(temp)*percent))
    return temp[:int(len(temp)*percent)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('-s', '--save-model', dest='save_model', nargs='?', help='input file', type=str, action='store', default=None)
    parser.add_argument('-l', '--load-model', dest='load_model', nargs='?', help='input file', type=str, action='store', default=None)
    parser.add_argument('-i', '--input_file', nargs=2, help='input file', type=str, action='store')
    parser.add_argument('--activation_func', nargs='?', help='activation function of elm', type=str, choices=['sigmoid', 'tanh', ''], default='sigmoid', action='store')
    parser.add_argument('--hidden_size', nargs='?', help='number of hidden nodes of elm', type=int, choices=range(5,21), default=10, action='store')
    parser.add_argument('--ensemble_size', nargs='?', help='number of elm of ensemble-based elm', type=int, choices=range(5,31,5), default=25, action='store')
    args = parser.parse_args()
    print args.input_file
    input_file = args.input_file
    
    dataset1 = get_data(input_file[0], enable_shuffle=True)
    dataset2 = get_data(input_file[1], enable_shuffle=True)
        
    if len(dataset1) > len(dataset2):
        dataset1 = dataset1[:len(dataset2)]
    elif len(dataset1) < len(dataset2):
        dataset2 = dataset2[:len(dataset1)]
        
    clsset1 = [0] * len(dataset1)
    clsset2 = [1] * len(dataset2)
    print '>>>>>>>>>>>>>.test'
    if args.load_model:
        print 'Load Model: {}'.format(args.load_model)
        test_lc = joblib.load(args.load_model)
    else:
        test_lc = LCELMClassifier()
        test_lc.fit(np.array(dataset1 + dataset2), np.array(clsset1 + clsset2))
    print 'Total: ',
    print len(dataset1 + dataset2)
    print 'TP: ',
    print sum(test_lc.predict(np.array(dataset1)))
    print 'TN: ',
    print sum(test_lc.predict(np.array(dataset2)))
    score = cross_val_score(test_lc, np.array(dataset1 + dataset2), np.array(clsset1 + clsset2), scoring='f1')
    print 'avg:{:.2f}%\tstd:{:.2f}%'.format(score.mean() * 100, score.std() * 100)
    print '>>>>>>>>>>>>>.test'
    if args.save_model:
        joblib.dump(test_lc, args.save_model)
    """
    avg = []
    std = []
    # train base classfiers
    elms = []
    for i in range(args.ensemble_size):
        elm = ELMClassifier(n_hidden=args.hidden_size, activation_func=args.activation_func)
        elms.append(elm.fit(np.array(dataset1 + dataset2), np.array(clsset1 + clsset2)))
        score = cross_val_score(elm, np.array(dataset1 + dataset2), np.array(clsset1 + clsset2), cv=10)
        avg.append(score.mean()*100)
        std.append(score.std()*100)
        #print 'avg:{:.2f}%\tstd:{:.2f}%'.format(score.mean()*100, score.std()*100)
        
    print 'avg:{:.2f}%\tstd:{:.2f}%'.format(np.array(avg).mean(), np.array(std).mean())
    
    # ensemble calculate
    f = _h(np.array(dataset1 + dataset2), elms, args.ensemble_size)
    print 'pseudo-inverse matrix'
    f_plus = np.linalg.pinv(np.array(f))
    v = f_plus.dot(np.array(clsset1 + clsset2))
    print v.shape
    print np.array([dataset1[0]])
    for index, i in enumerate(dataset1):
        if index > 100:
            break
        test = _h(np.array([i]), elms, args.ensemble_size)
        print np.array(test).dot(v)
    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
    for index, i in enumerate(dataset2):
        if index > 100:
            break
        test = _h(np.array([i]), elms, args.ensemble_size)
        result = np.array(test).dot(v)
    for f in args.input_file:
        f.close()
    """
