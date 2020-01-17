import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

import preproc

def feat_imp(scores, feature_names, save_path=None):
    '''
        scores: list of feature importances (if scores are retrieved
        from cross validation, pass scores like:
            [fi.feature_importances_ for fi in scores['estimator']]
        )
        feature_names: columns of original dataframe
        save_path: None is you do not wish to save it, the path otherwise
        returns feature importances (mean for cv) as dataframe
    '''
    scores = np.array(scores)
    df = pd.DataFrame(scores if scores.ndim == 1 else scores[0], index=feature_names, columns=['Importance'])
    if scores.ndim > 1:
        for estimator in scores[1:]:
            df = (df + pd.DataFrame(estimator,
                                    index=feature_names,
                                    columns=['Importance']))

    df = df/scores.ndim
    df = df.sort_values(by='Importance', ascending=False)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Feature'}, inplace=True)

    return df

def plot_feat_imp(df, x, y, n_max=5):
    df = df.head(n=n_max)
    g = sns.barplot(x=x, y=y, data=df)
    g.set_xlabel('Importance')
    g.set_ylabel('Feature')
    g.set_title('Feature importance')
    # g.figure.savefig('{}.png'.format(y), bbox_inches="tight")
    plt.show()

df_train = preproc.pre_process_v2('train.csv')
df_test = preproc.pre_process_v2('test.csv')
# df_train = preproc.pre_process_v1('train.csv')
# df_test = preproc.pre_process_v1('test.csv')

#............. feature selection .............#
df_train = df_train.sample(frac=1) # shuffling dataset

def validating(X_train, y_train):
    clf = RandomForestClassifier()
    return cross_validate(clf, X_train, y_train, cv=5)

X_train = df_train.drop('Survived', axis=1)
y_train = df_train.Survived
rf_fi = RandomForestClassifier()
rf_fi.fit(X_train, y_train)
rf_fi.feature_importances_

# withoug feature selection
cv_results = validating(X_train, y_train)

fi = feat_imp(rf_fi.feature_importances_, X_train.columns)
plot_feat_imp(fi, 'Importance', 'Feature')

# with feature selection
X_train = X_train[fi.head(6).Feature.values]
cv_results = validating(X_train, y_train)
cv_results

#............. tuning .............#
import scipy.stats as stats
from sklearn.utils.fixes import loguniform

clf = SVC()
param_dist = {'C': stats.uniform(1, 10e4),
              'kernel': ['rbf', 'sigmoid'],
              'gamma': loguniform(1e-5, 1e0)}

n_iter_search = 100
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

rs = random_search.fit(X_train, y_train)
# Current 80% accuracy on kaggle: 
# {'C': 7110.060148342983, 'gamma': 0.01177348670454452, 'kernel': 'rbf'}
params = rs.best_params_

# validating params foun by random search
clf = SVC(C=params['C'], gamma=params['gamma'], kernel=params['kernel'])
cv_results = cross_validate(clf, X_train, y_train, cv=5)
clf.fit(X_train, y_train)

# Final test with the model employed
final_pred = df_test.copy()
final_pred['Survived'] = clf.predict(df_test)
final_pred.reset_index(inplace=True)
final_pred[['PassengerId', 'Survived']].to_csv('predictions/predictions_v4.csv', index=False)
