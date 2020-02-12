import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
plt.style.use('ggplot')

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 20
os.chdir('house_prices/')

train = pd.read_csv('data/train.csv')
train_ = train.copy()
train['Dataset'] = 'train'
y = train.SalePrice
train.drop('SalePrice', axis=1, inplace=True)

test = pd.read_csv('data/test.csv')
test['Dataset'] = 'test'

df = pd.concat([train, test])

df.describe()
df.info()
df.set_index(['Id', 'Dataset'], inplace=True)

#................ missing values ................#
missing = df.isnull().sum()
missing[missing > 0].sort_values(ascending=False)
df.drop(missing[missing > 1000].index, axis=1, inplace=True)

# numerical
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
df['GarageCars'] = df.GarageCars.fillna(0)
df['GarageArea'] = df.GarageArea.fillna(0)
df.drop('GarageYrBlt', axis=1, inplace=True)
df['MasVnrArea'] = df.MasVnrArea.fillna(0)
df['MasVnrArea'] = np.where(
    df.MasVnrArea == 0, 0, 1
)
bsmt_num = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
for i in bsmt_num:
    df[i] = df[i].fillna(0)

# categorical
df['GarageType'] = df.GarageType.fillna('NA')
df['GarageFinish'] = np.where(
    df.GarageFinish == 'Unf', 0, 1
)
bsmt_cat = ['BsmtQual',
       'BsmtCond',
       'BsmtExposure',
       'BsmtFinType1',
       'BsmtFinType2'
]
df.loc[:, bsm] = df[bsmt_cat].fillna('NA')
df.drop(['MasVnrType', 'GarageQual', 'GarageCond'], axis=1, inplace=True)

few_missing_values = ['MSZoning', 'Utilities', 'Exterior1st',
                      'Exterior2nd', 'Electrical',
                      'KitchenQual', 'Functional', 'SaleType']

for f in few_missing_values:
    df[f] = df[f].fillna(df[f].mode()[0])

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing

'''
# apgar daqui pra baixo 
categorical = [f for f in df.columns if df.dtypes[f] == 'object'
               and f in missing.index]
numerical = [f for f in df.columns if df.dtypes[f] != 'object'
             and f in missing.index]
missing[numerical]
missing[categorical]


# from the numerical approaches, we obtain some insights for categorical:
# MasVnrArea was replace by a binary column, thus MasVnrType can be dropped;
# GarageType/Finsih/Qual/Cond we already know that the missing values are
# the same for the same rows;
# The Bsmt columns we can evaluate later, because I think we can drop some of them.

print(*sorted(df.columns), sep='\n')

train = df[df.index.get_level_values(1) == 'train'].copy()
train.reset_index(inplace=True)
train['SalePrice'] = y
train.set_index(['Id', 'Dataset'], inplace=True)
train.corr()['SalePrice'].sort_values()

# Bsmt 
df.BsmtQual.value_counts()
df.BsmtCond.value_counts()
df.BsmtExposure.value_counts()
df.BsmtFinType1.value_counts()
df.BsmtFinType2.value_counts()

df.loc[:, bsm] = df[bsmt_cat].fillna('NA')
df.isna().sum()

# Garages categorical 
df.GarageFinish.value_counts() # transform to finished or not
df.GarageQual.value_counts() # drop
df.GarageCond.value_counts() # drop
df.GarageType.value_counts() # fillna by 'NA'
# train.corr()['SalePrice'].sort_values()
# sns.scatterplot(x='GarageType2', y='SalePrice', data=train)
# plt.show()
# Bsmt* 
len(df[df.BsmtFinSF1 == 0])
len(df[df.BsmtFinSF2 == 0])
len(df[df.BsmtUnfSF == 0])
len(df[df.TotalBsmtSF == 0])
len(df[df.BsmtFullBath == 0])
len(df[df.BsmtHalfBath == 0])


# GaregeArea 
df[df['GarageArea'].isna()].index
df.loc[(2577, 'test'), ['GarageArea', 'GarageYrBlt', 'GarageArea']]

#MasVnrArea: Masonry veneer area in square feet 
df.MasVnrArea.describe()
train.corr()['MasVnrArea'].sort_values()
# df.MasVnrArea.plot.hist()
# plt.show()

train.corr()['SalePrice'].sort_values()
# sns.scatterplot(x='MasVnrArea', y='SalePrice', hue='OverallQual', data=train)
# plt.show()
# Although this column presents 0.5 correlation rate to SalePrice,
# it has a lot of zeros and missing values. An alternative would be
# replacing this column by a binary one: if it has Masonry veneer or not
df.MasVnrArea.isnull().sum()
df.MasVnrArea[df.MasVnrArea == 0].value_counts()
df.MasVnrArea = df.MasVnrArea.fillna(0)
df.MasVnrArea = np.where(
    df.MasVnrArea == 0, 0, 1
)
df.MasVnrArea.value_counts()

# GarageYrBlt: Year garage was built  
df.GarageYrBlt
train.corr()['GarageYrBlt'].sort_values()
# most of YearBuilt and GarageYrBlt values are the same, so we can trop GarageYrBlt
# instead of replacing the missing values
df.loc[:, ['GarageYrBlt', 'YearBuilt']]

# LotFrontage: Linear feet of street connected to property  
# from data description, we could assume this feature directly 
# related to neighborhood or LotArea featueres;
# thus, we could replace null values by the LotFrontage mean,
# or the mean grouped by one of the mentioned featueres

# evaluating neighborhood, replacing null values by median
train['Neighborhood2'] = LabelEncoder().fit_transform(train.Neighborhood)
train['missing'] = train['LotFrontage'].isna()

train.corr()['LotFrontage'].sort_values()

fig, ax = plt.subplots(1, 3)
sns.scatterplot(x='Neighborhood2', y='LotFrontage',
                hue='Neighborhood',
                data=train, ax=ax[0])

train['LotFrontage2'] = train.LotFrontage.fillna(train.LotFrontage.mean())
sns.scatterplot(x='Neighborhood2', y='LotFrontage2',
                hue='missing',
                data=train,
                ax=ax[1])
train['LotFrontage2'] = train.groupby('Neighborhood2')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
sns.scatterplot(x='Neighborhood2', y='LotFrontage2',
                hue='missing',
                data=train,
                ax=ax[2])
# plt.show()

# # evaluating LotArea, replacing null values by median
# # before replacing null values by grouping LotArea, we need to transform this
# # feature into a categorical one to enable us to group it. 
# train.LotArea.describe()
# train.LotArea = train.LotArea.apply(np.log)
# train['LotArea2'] = pd.cut(train.LotArea, 8, labels=range(8))
# # plotting this we can notice this column is not very related to SalePrice
# # however, it has high correlation rate with LotFrontage
# sns.scatterplot('LotArea2', 'SalePrice', data=train)
# plt.show()
# train.corr()['LotArea'].sort_values()

# fig, ax = plt.subplots(1, 3)
# sns.scatterplot(x='LotArea2', y='LotFrontage',
#                 hue='LotArea2',
#                 data=train, ax=ax[0])
# ax[0].set_title('Grouped by')

# train['LotFrontage3'] = train.LotFrontage.fillna(train.LotFrontage.mean())
# sns.scatterplot(x='LotArea2', y='LotFrontage3',
#                 hue='missing',
#                 data=train,
#                 ax=ax[1])
# ax[1].set_title('Replacing NaNs by the column median')

# train['LotFrontage3'] = train.groupby('LotArea2')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
# sns.scatterplot(x='LotArea2', y='LotFrontage3',
#                 hue='missing',
#                 data=train,
#                 ax=ax[2])
# ax[2].set_title('Replacing NaNs by group medians')

# plt.show()
# train.corr()['SalePrice'].sort_values()
# Both strategies guaranted the same corelation rate to SalePrice
# Lets use the neighborhood then, because it is simpler
'''
#................ missing values ................#

#................ encoding categorical variables ................#
categorical = [f for f in df.columns if df.dtypes[f] == 'object']

for c in categorical:
    df['cat_' + c] = LabelEncoder().fit_transform(df[c])
    
df.drop(categorical, axis=1, inplace=True)

# selector = VarianceThreshold(0.15)
# selector.fit(df)
# # get support returns true for columns with var > 0.15
# df.drop(df.columns[~selector.get_support()], axis=1, inplace=True)
#................ encoding categorical variables ................#

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

# selector = VarianceThreshold(0.1)
# selector.fit(df)
# # get support returns true for columns with var > 0.15
# df.drop(df.columns[~selector.get_support()], axis=1, inplace=True)


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

numerical = [c for c in df.columns if not c.startswith('cat_')]
# tmp['YearBuilt_interval'] = pd.cut(df.YearBuilt, 5)
# tmp[['YearBuilt_interval', 'SalePrice']].groupby('YearBuilt_interval').mean().sort_values(by='SalePrice')
# df.YearBuilt.plot.hist()
# plt.show()
# df.loc[(1, 'train'), numerical]
# df.loc[:, 'LotArea'] = df.loc[:, 'LotArea'].apply(np.log)
# df.loc[:, 'GrLivArea'] = df.loc[:, 'GrLivArea'].apply(np.log)
# numerical.remove('LotArea')
# numerical.remove('GrLivArea')
# df.drop(['YearBuilt', 'YearRemodAdd', 'YrSold'], axis=1, inplace=True)

df.GrLivArea = df.GrLivArea.apply(np.log)
df.TotalBsmtSF = df.TotalBsmtSF.apply(lambda x: np.log(x) if x else x)
df.loc[:, 'OverallQual'] = pd.cut(df.OverallQual, 3, labels=[0, 1, 2])
df = pd.get_dummies(df, columns=['OverallQual'])
df.loc[:, 'TotRmsAbvGrd'] = pd.cut(df.TotRmsAbvGrd, 4, labels=[0, 1, 2, 3])
df.loc[:, 'YearRemodAdd'] = pd.cut(df.YearRemodAdd, 3, labels=[0, 1, 2]).astype('int')
df = pd.get_dummies(df, columns=['TotRmsAbvGrd'])
df.loc[:, '1stFlrSF'] = df['1stFlrSF'].apply(lambda x: np.log(x) if x else x)
df.loc[:, '2ndFlrSF'] = df['2ndFlrSF'].apply(lambda x: np.log(x) if x else x)
# Analyzing YearBuilt we noticed that values > 1982 are very
# higher than other. Thus, we will transform this column
# into binary
# train_.loc[:, 'Year_tmp'] = pd.cut(train_[var], 10) #, labels=range(10))
df.loc[:, 'YearBuilt'] = np.where(
    df.loc[:, 'YearBuilt'] > 1982, 1, 0
)
# From here, Neighborhood column looks really useless. Drop it.
df.drop(['MSSubClass', 'OpenPorchSF', 'BedroomAbvGr',
         'LotArea', 'YrSold'], axis=1, inplace=True)
# cat_Neighborhood
# numerical = [c for c in df.columns if not c.startswith('cat_')]
# df.loc[(1, 'train'), :].sort_values()
# sns.heatmap(df[numerical].corr())
# var='YrSold'
# df[var].describe()
# sns.swarmplot(x='Neighborhood', y='SalePrice', hue='Neighborhood',
#                 data=train_)
# plt.show()
# print(*df.columns.sort_values(), sep='\n')

# selector = VarianceThreshold(0.15)
# selector.fit(df)
# df.drop(df.columns[~selector.get_support()], axis=1, inplace=True)
#................ handling numerical data ................#

#................ skewness, outliers ................#
from scipy.stats import skew

# cols = [c for c in df.columns if df[c].dtype == 'object']
sks = df.apply(lambda x: skew(x))
sks = pd.DataFrame({'Skew': sks})
sks.sort_values(by='Skew', ascending=True)
sks = sks[abs(sks) > 0.75]

from scipy.special import boxcox1p
skewed_features = sks.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    df[feat] = boxcox1p(df[feat], lam)
    

#................ skewness, outliers ................#

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

#................ feature selection ................#
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
reg = GradientBoostingRegressor(**gb_bestparams)
reg.fit(X, y)
reg.feature_importances_
fi = pd.DataFrame({
    'Features': X.columns,
    'Importance': reg.feature_importances_
})
fi.sort_values(by='Importance')
# X = X[fi[fi.Importance > 0.01].Features.values]

#................ feature selection ................#

#................ tuning ................#
import scipy.stats as stats
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

gb_params = {
    'loss': ['ls', 'lad', 'huber'],
    'n_estimators': [100, 150, 200, 300, 500],
    'min_samples_split': [2, 4, 6, 10, 16, 20, 50],
    'max_features': ['auto', 'sqrt', 'log2', None]
}

lasso_params = {
    'alpha': stats.loguniform(1e-5, 1e0)
}

n_iter_search = 30
lasso_rs = RandomizedSearchCV(
    Lasso(),
    param_distributions=lasso_params,
    n_iter=n_iter_search, scoring='neg_mean_squared_error'
)

gb_rs = GridSearchCV(
    GradientBoostingRegressor(),
    param_grid=gb_params,
    scoring='neg_mean_squared_error',
    cv=5
)

# rf_rs = rf_rs.fit(X, y)
# svr_rs = svr_rs.fit(X, y)
lasso_rs = lasso_rs.fit(X, y)
gb_rs = gb_rs.fit(X, y)

# rf_rs.best_params_
rf_bestparams = {
    'max_depth': 150,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 300
}

# svr_rs.best_params_
svr_bestparams = {
    'C': 10512.841195085914,
    'gamma': 4.109049702149218e-05,
    'kernel': 'rbf'
}
# lasso_rs.best_params_
lasso_bestparams = {'alpha': 0.00017981640047254707}
# gb_rs.best_params_
# gb_bestparams = {
#     'n_estimators': 300,
#     'min_samples_split': 6,
#     'max_features': 'log2',
#     'loss': 'huber'
# }
gb_bestparams = {
    'loss': 'huber',
    'max_features': 'log2',
    'min_samples_split': 10,
    'n_estimators': 500
}

#................ tuning ................#

#................ validating ................#
from sklearn.svm import SVR
import scipy.stats as stats
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

rf_bestparams = {
    'max_depth': 150,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 300
}

# svr_rs.best_params_
svr_bestparams = {
    'C': 10512.841195085914,
    'gamma': 4.109049702149218e-05,
    'kernel': 'rbf'
}
# lasso_rs.best_params_
lasso_bestparams = {'alpha': 6.34105969106034e-05}
# gb_rs.best_params_
# gb_bestparams = {
#     'n_estimators': 300,
#     'min_samples_split': 6,
#     'max_features': 'log2',
#     'loss': 'huber'
# }
gb_bestparams = {
    'loss': 'huber',
    'max_features': 'sqrt',
    'min_samples_split': 10,
    'n_estimators': 3000,
    'learning_rate': 0.05
}

rf_cv = cross_validate(
    RandomForestRegressor(**rf_bestparams),
    X, y,
    scoring='neg_mean_squared_error', cv=5,
)
# svr_cv = cross_validate(
#     SVR(**svr_bestparams),
#     X, y,
#     scoring='neg_mean_squared_error', cv=5,
# )
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
    'rf': np.sqrt(abs(rf_cv['test_score'].mean())),
    # 'svr': svr_cv['test_score'].mean(),
    'lasso': np.sqrt(abs(lasso_cv['test_score'].mean())),
    'gb': np.sqrt(abs(gb_cv['test_score'].mean()))
}
pd.DataFrame([scores]).loc[0].sort_values()
#................ validating ................#

#................ model evaluation ................#
from xgboost import XGBRegressor
# reg = Lasso(alpha=lasso_bestparams['alpha'])
# reg = GradientBoostingRegressor(**gb_bestparams)
# reg = RandomForestRegressor(**rf_bestparams)
reg = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                    learning_rate=0.05, max_depth=3, 
                    min_child_weight=1.7817, n_estimators=2200,
                    reg_alpha=0.4640, reg_lambda=0.8571,
                    subsample=0.5213, silent=1,
                    random_state =7, nthread = -1)
reg = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10, 
                                loss='huber', random_state=5)

reg.fit(X, y)

test_ = test.copy()
# test_ = test[fi[fi.Importance > 0.01].Features.values].copy()
test_['SalePrice'] = reg.predict(test_)
# test_['SalePrice'] = test_['SalePrice'].apply(lambda x: round(np.e ** x))
test_['SalePrice'] = test_['SalePrice'].apply(lambda x: np.floor(np.exp(x)))
test_.reset_index(inplace=True)
test_[['Id', 'SalePrice']].to_csv('predictions.csv', index=False)

#................ model evaluation ................#
