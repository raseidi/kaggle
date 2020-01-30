import scipy.stats as stats
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV

clf = SVR()
param_dist = {'C': stats.uniform(1, 10e4),
              'kernel': ['rbf', 'sigmoid'],
              'gamma': stats.expon(scale=.1)(1e-5, 1e0)}

n_iter_search = 100
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)