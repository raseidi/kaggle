import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 20
os.chdir('house_prices/')

train = pd.read_csv('data/train.csv')
train['Dataset'] = 'train'
y = train.SalePrice
train.drop('SalePrice', axis=1, inplace=True)

test = pd.read_csv('data/test.csv')
test['Dataset'] = 'test'

df = pd.concat([train, test])

df.describe()
df.info()
df.set_index(['Id', 'Dataset'], inplace=True)

''' dropping all columns that have missing values '''
missing = df.isnull().sum()
missing[missing > 0].index
df.drop(missing[missing > 0].index, axis=1, inplace=True)

''' encoding categorical variables '''
categorical = [f for f in df.columns if df.dtypes[f] == 'object']

for c in categorical:
    df['cat_' + c] = LabelEncoder().fit_transform(df[c])
    
df.drop(categorical, axis=1, inplace=True)

selector = VarianceThreshold(0.15)
selector.fit(df)
# get support returns true for columns with var > 0.15
df.drop(df.columns[~selector.get_support()], axis=1, inplace=True)

#................ imporving categorical data ................#
''' plotting boxplots of categorical variables
def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=65)

categorical = [c for c in df.columns if c.startswith('cat_')]
# l = sorted(df.index.__dir__())
# print(*l, sep='\n')
tmp = df[df.index.get_level_values(1) == 'train'].copy()
tmp.loc[:, 'SalePrice'] = y.values
f = pd.melt(tmp, id_vars=['SalePrice'], value_vars=categorical)
plt.figure(dpi=100, figsize=(16,8))
g = sns.FacetGrid(f, col="variable",  col_wrap=4, sharex=False, sharey=False)
g = g.map(boxplot, "value", "SalePrice")
g.savefig('plots/dist_categorical_v2.png')
'''

# conclusions toke from data visualization
# drop neighborhood
# PavedDrive = [0,1], [2]
# HeatingQC = [0], [1,2,3,4]
# HouseStyle = [0,1,4,6,7], [2, 3,5]
# LotConfig = [1, 2, 4], [2, 3]
# LotShape = [0,1,3], [2]
# RoofStyle = [1,2,4,5], [2,3]
# RoofMatl = [0,2,3,4,5,6], [1,7]
to_binary = ['cat_PavedDrive', 'cat_HeatingQC', 'cat_HouseStyle', 'cat_LotConfig',
             'cat_LotShape', 'cat_RoofMatl', 'cat_RoofStyle']

df.loc[:, 'cat_PavedDrive'] = np.where(
    df.loc[:, 'cat_PavedDrive'].isin([0, 1]), 0, 1
)

df.loc[:, 'cat_HeatingQC'] = np.where(
    df.loc[:, 'cat_HeatingQC'].isin([1, 2, 3, 4]), 0, 1
)

df.loc[:, 'cat_HouseStyle'] = np.where(
    df.loc[:, 'cat_HouseStyle'].isin([0,1,4,6,7]), 0, 1
)

df.loc[:, 'cat_LotConfig'] = np.where(
    df.loc[:, 'cat_LotConfig'].isin([1, 2, 4]), 0, 1
)

df.loc[:, 'cat_LotShape'] = np.where(
    df.loc[:, 'cat_LotShape'].isin([0,1,3]), 0, 1
)

df.loc[:, 'cat_RoofStyle'] = np.where(
    df.loc[:, 'cat_RoofStyle'].isin([1,2,4,5]), 0, 1
)

df.loc[:, 'cat_RoofMatl'] = np.where(
    df.loc[:, 'cat_RoofMatl'].isin([0,2,3,4,5,6]), 0, 1
)

selector = VarianceThreshold(0.1)
selector.fit(df)
# get support returns true for columns with var > 0.15
df.drop(df.columns[~selector.get_support()], axis=1, inplace=True)


# BldgType = {0: [1,2,3], 1: [4], 2: [0]}
# ExternalQual = {0: 0, 1: 1, 2: 2, 3: 3}
# ExternalCond = {0: [1, 3], 1: [0], 2: [2, 4]}
# SaleCondition = {0: [0,1,2,3], 1: [4], 2: [5]}
# Condition1 = {0: [1, 5, 6], 1: [0, 3, 4, 6, 7, 8], 2: [2]}
# LandContour = {0: [0], 1: [1, 2], 2: [3]}
# Foundation = {0: [0,1], 1: [3,4,5], 2: [2]}
one_hot_encoding = ['cat_BldgType', 'cat_ExterQual', 'cat_ExterCond',
                    'cat_SaleCondition', 'cat_Condition1', 'cat_LandContour',
                    'cat_Foundation']

df.loc[:, 'cat_BldgType'] = df.loc[:, 'cat_BldgType'].map({
    1: 0, 2: 0, 3: 0,
    4: 1,
    0: 2
})

df.loc[:, 'cat_ExterCond'] = df.loc[:, 'cat_ExterCond'].map({
    1: 0, 3: 0,
    0: 1,
    2: 2, 4: 2
})

df.loc[:, 'cat_SaleCondition'] = df.loc[:, 'cat_SaleCondition'].map({
    0: 0, 1: 0, 2: 0, 3: 0,
    4: 1,
    5: 2
})

df.loc[:, 'cat_Condition1'] = df.loc[:, 'cat_Condition1'].map({
    1: 0, 5: 0, 6: 0,
    0: 1, 3: 1, 4: 1, 6: 1, 7: 1, 8: 1,
    2: 2
})

df.loc[:, 'cat_LandContour'] = df.loc[:, 'cat_LandContour'].map({
    0: 0,
    1: 1, 2: 1,
    3: 2
})

df.loc[:, 'cat_Foundation'] = df.loc[:, 'cat_Foundation'].map({
    0: 0, 1: 0,
    3: 1, 4: 1, 5: 1,
    2: 2
})

df = pd.get_dummies(df, columns=one_hot_encoding)
#................ imporving categorical data ................#

#................ handling numerical data ................#

# numerical = [c for c in df.columns if not c.startswith('cat_')]
# tmp['YearBuilt_interval'] = pd.cut(df.YearBuilt, 5)
# tmp[['YearBuilt_interval', 'SalePrice']].groupby('YearBuilt_interval').mean().sort_values(by='SalePrice')
# df.YearBuilt.plot.hist()
# plt.show()
#................ handling numerical data ................#



#................ splitting into train/test ................#
df.reset_index(inplace=True)
train = df[df.Dataset == 'train'].copy()
train['SalePrice'] = y.copy()
test = df[df.Dataset == 'test'].copy()

train.drop('Dataset', axis=1, inplace=True)
train.set_index('Id', inplace=True)
test.drop('Dataset', axis=1, inplace=True)
test.set_index('Id', inplace=True)

train.loc[:, 'SalePrice'] = train.SalePrice.apply(np.log)
X = train.drop('SalePrice', axis=1)
y = train.SalePrice
#................ splitting into train/test ................#

#................ tuning ................#
import scipy.stats as stats
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

gb_params = {
    'loss': ['ls', 'lad', 'huber'],
    'n_estimators': [100, 150, 200, 300],
    'min_samples_split': [2, 4, 6, 10],
    'max_features': ['auto', 'sqrt', 'log2', None]
}

rf_params = {
    'n_estimators': [100, 150, 200, 300]
}

svr_params = {
    'C': stats.uniform(1, 10e4),
    'kernel': ['rbf', 'sigmoid'],
    'gamma': stats.loguniform(1e-5, 1e0)
}
lasso_params = {
    'alpha': stats.loguniform(1e-5, 1e0)
}

n_iter_search = 30
rf_rs = RandomizedSearchCV(
    RandomForestRegressor(),
    param_distributions=rf_params,
    n_iter=n_iter_search, scoring='neg_mean_squared_error'
)

svr_rs = RandomizedSearchCV(
    SVR(),
    param_distributions=svr_params,
    n_iter=n_iter_search, scoring='neg_mean_squared_error'
)

lasso_rs = RandomizedSearchCV(
    Lasso(),
    param_distributions=lasso_params,
    n_iter=n_iter_search, scoring='neg_mean_squared_error'
)

gb_rs = RandomizedSearchCV(
    GradientBoostingRegressor(),
    param_distributions=gb_params,
    n_iter=n_iter_search, scoring='neg_mean_squared_error'
)

rf_rs = rf_rs.fit(X, y)
svr_rs = svr_rs.fit(X, y)
lasso_rs = lasso_rs.fit(X, y)
# gb_rs = gb_rs.fit(X, y)

# rf_rs.best_params_
rf_bestparams = {'n_estimators': 200}
# svr_rs.best_params_
svr_bestparams = {
    'C': 10512.841195085914,
    'gamma': 4.109049702149218e-05,
    'kernel': 'rbf'
}
# lasso_rs.best_params_
lasso_bestparams = {'alpha': 6.34105969106034e-05}
# gb_rs.best_params_
gb_bestparams = {
    'n_estimators': 300,
    'min_samples_split': 6,
    'max_features': 'log2',
    'loss': 'huber'
}

#................ tuning ................#

#................ validating ................#
from sklearn.model_selection import cross_validate

rf_cv = cross_validate(
    RandomForestRegressor(
        n_estimators=rf_bestparams['n_estimators']
    ),
    X, y,
    scoring='neg_mean_squared_error', cv=5,
)
svr_cv = cross_validate(
    SVR(**svr_bestparams),
    X, y,
    scoring='neg_mean_squared_error', cv=5,
)
lasso_cv = cross_validate(
    Lasso(
        alpha=lasso_bestparams['alpha']   
    ),
    X, y,
    scoring='neg_mean_squared_error', cv=5,
)
gb_cv = cross_validate(
    GradientBoostingRegressor(**gb_bestparams),
    X, y,
    scoring='neg_mean_squared_error', cv=5,
)
scores = {
    'rf': rf_cv['test_score'].mean() ** 2,
    'svr': svr_cv['test_score'].mean() ** 2,
    'lasso': lasso_cv['test_score'].mean() ** 2,
    'gb': gb_cv['test_score'].mean() ** 2
}
pd.DataFrame([scores]).loc[0].sort_values()
#................ validating ................#

#................ model evaluation ................#
# reg = Lasso(alpha=lasso_bestparams['alpha'])
reg = RandomForestRegressor(**rf_bestparams)
# reg = GradientBoostingRegressor(**gb_bestparams)
reg.fit(X, y)

test_ = test.copy()
test_['SalePrice'] = reg.predict(test)
# test_['SalePrice'] = test_['SalePrice'].apply(lambda x: round(np.e ** x))
test_['SalePrice'] = test_['SalePrice'].apply(lambda x: np.floor(np.exp(x)))
test_.reset_index(inplace=True)
test_[['Id', 'SalePrice']].to_csv('predictions_v2.csv', index=False)




#................ model evaluation ................#
