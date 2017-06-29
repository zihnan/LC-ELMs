import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble.base import _partition_estimators
from sklearn.externals import joblib
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state, check_X_y, check_array, column_or_1d
from sklearn.utils.validation import check_is_fitted
from sklearn_extensions.extreme_learning_machines import ELMClassifier
                                             
class LCELMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=25, n_hidden=20, alpha=0.5, rbf_width=1.0,
                 activation_func='sigmoid', activation_args=None,
                 user_components=None, regressor=None,
                 binarizer=LabelBinarizer(-1, 1),
                 random_state=None, verbose=False):
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.random_state = random_state
        self.activation_func = activation_func
        self.activation_args = activation_args
        self.user_components = user_components
        self.rbf_width = rbf_width
        self.regressor = regressor
        self.n_estimators = n_estimators
        #self.n_jobs = n_jobs
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
        X, y = check_X_y(X, y, ['csr', 'csc'])
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
        return self.binarizer.inverse_transform(self.predict_proba(X))
    
    def predict_proba(self, X):
        check_is_fitted(self, 'classes_')
        X = check_array(X, accept_sparse=['csr', 'csc'])
        n_sample, n_features = X.shape
        
        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))
        ''' not yet
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
    
    def decision_function(self, X):
        return self.predict_proba(X)/(self.binarizer.pos_label - self.binarizer.neg_label)
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
