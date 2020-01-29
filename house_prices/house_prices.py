import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.display.max_rows = 81

os.chdir('house_prices/')

df_ = pd.read_csv('data/train.csv')
df = df_.copy()
df.describe()
df.info()

#................... handling missing values ...................#

# dropping columns with many missing values;
# except for FireplaceQu columns, which can be replaced
# by true/false (there is a fireplace or not)
df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
df.FireplaceQu.fillna(0, inplace=True)
df.loc[:,'FireplaceQu'] = df.FireplaceQu.apply(lambda x: x if x == 0 else 1)

print(*df.columns[df.isnull().any()], sep='\n')
# first attempt just dropping all lines with missing values;
# this way we lose 366 samples.
df.dropna(axis=0, inplace=True)

#................... handling missing values ...................#

#................... handling categorical values ...................#

def catplot(x='', y='SalePrice', data=None, kind='box'):
    sns.catplot(x=x, y=y, hue=x,
    data=data, kind=kind)
    plt.show()

drop_columns = []
df.loc[0,:]

df.BsmtExposure.value_counts()
# catplot(x='BsmtExposure', data=df, kind='swarm')
df.loc[:, 'BsmtExposure'] = np.where(
    df.BsmtExposure.str.contains('Gd'), 1,0
)

df.BsmtCond.value_counts()
# catplot(x='BsmtCond', data=df)
drop_columns.append('BsmtCond')

# catplot show that TA and Fa are similar, then we 
# can do an one hot encode (TA+Fa, Gd, Ex)
df.BsmtQual.value_counts()
# catplot(x='BsmtQual', data=df, kind='swarm')
df.loc[:, 'BsmtQual'] = df.BsmtQual.map({
    'TA': 0,
    'Fa': 0,
    'Gd': 1,
    'Ex': 2
})
df = pd.get_dummies(df, columns=['BsmtQual'])

df.Foundation.value_counts()
# catplot(x='Foundation', data=df)
df.loc[:, 'Foundation'] = np.where(
    df.Foundation.str.contains('PConc'), 1, 0
)

# through the catplot we can note ExterCond
# has no impact over the sale price
df.ExterCond.value_counts()
# catplot(x='ExterCond', data=df,kind='box')
drop_columns.append('ExterCond')

# Gd and Ex have similar SalePrice, as well as TA and Fa.
# Thus we just binarize both
df.ExterQual.value_counts()
# catplot(x='ExterQual', data=df,kind='box')
df.loc[:, 'ExterQual'] = np.where(
    df.ExterQual.str.contains('Gd|Ex', regex=True),
    1, 0
)

df.MasVnrType.value_counts()
df.loc[:, 'MasVnrType'] = np.where(
    df.MasVnrType.str.contains('None'), 1, 0
)

# Similar to Neighborhood
# ToDo
df.Exterior2nd.value_counts()
drop_columns.append('Exterior1st')
drop_columns.append('Exterior2nd')

df.RoofMatl.value_counts()
drop_columns.append('RoofMatl')

df.RoofStyle.value_counts()
# catplot(x='RoofStyle', data=df, kind='swarm')
df.loc[:, 'RoofStyle'] = np.where(
    df.RoofStyle.str.contains('Gable'), 1, 0
)

# 2Story has the highest variance. An alternative
# is to transform into a binary columns
# true if it is 2Story false otherwise
df.HouseStyle.value_counts()
# catplot(x='HouseStyle', data=df, kind='swarm')
df.loc[:, 'HouseStyle'] = np.where(
    df.HouseStyle.str.contains('2Story'), 1, 0
)

df.BldgType.value_counts()
# catplot(x='BldgType', data=df)
df.loc[:, 'BldgType'] = np.where(
    df.BldgType.str.contains('1Fam'), 1, 0
)

df.Condition2.value_counts()
drop_columns.append('Condition2')
df.Condition1.value_counts()
# catplot(x='Condition1', data=df)
df.loc[:, 'Condition1'] = np.where(
    df.Condition1.str.contains('Norm'), 1, 0
)

# still dont know how to handle this, so ill drop it
# ToDo
df.Neighborhood.value_counts()
#df.groupby('Neighborhood').SalePrice.max().sort_values()
drop_columns.append('Neighborhood')

df.LandSlope.value_counts()
drop_columns.append('LandSlope')

# very high variance rate
df.LotConfig.value_counts()
cat_to_num = dict(zip(sorted(df.LotConfig.unique()), range(5)))
df.loc[:, 'LotConfig'] = df.LotConfig.map(cat_to_num)
#df.groupby('LotConfig').SalePrice.mean().sort_values()
#df.groupby('LotConfig').SalePrice.std().sort_values()

df.Utilities.value_counts()
drop_columns.append('Utilities')

# droping because there is a majority value (Lvl)
df.LandContour.value_counts()
#df.groupby('LandContour').SalePrice.min().sort_values()
#df.groupby('LandContour').SalePrice.max().sort_values()
drop_columns.append('LandContour')

df.LotShape.value_counts()
#df.groupby('LotShape')['SalePrice'].mean().sort_values()
#df.groupby('LotShape')['SalePrice'].std().sort_values()
''' ive just find a easier way to do this
df.loc[:, 'LotShape'] = df.LotShape.map({
    'Reg': 0,
    'IR1': 1,
    'IR2': 1,
    'IR3': 1 # IRs are very similar, then we assume they are equivalent
})'''
df.loc[:, 'LotShape'] = np.where(
    df.LotShape.str.contains('IR'), 1, 0
)

# this can be dropped, there are 1090 pave and only 4 grvl
df.Street.value_counts()
drop_columns.append('Street')

# RL has high variance in SalePrice;
# it has the most expansive and second most cheap house;
# maybe it can be dropped.
df.MSZoning.value_counts()
#df.groupby(['MSZoning'])['SalePrice'].std().sort_values()
drop_columns.append('MSZoning')

df.drop(drop_columns, axis=1, inplace=True)
df.set_index('Id', inplace=True)
# corr = df.corr()
# sns.heatmap(corr, 
#         xticklabels=corr.columns,
#         yticklabels=corr.columns)
# plt.show()
df.drop(columns=df.select_dtypes('object').columns, axis=1, inplace=True)
#................... handling categorical values ...................#


#................... validation ...................#
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor

final_df = df.copy()
X = final_df.drop('SalePrice', axis=1).copy()
y = final_df.loc[:, 'SalePrice']

cv_scores = cross_validate(RandomForestRegressor(),
                           X, y,
                           scoring='r2')
# print(cross_validate.__doc__)
#................... validation ...................#

#................... employing final model  ...................#
#................... employing final model  ...................#

train = preproc('data/train.csv') 
test = preproc('data/test.csv')

reg = RandomForestRegressor()
reg.fit(train.drop('SalePrice', axis=1), train.SalePrice)
reg.predict(test)
test['SalePrice'] = reg.predict(test)
test.loc[:,'SalePrice'].to_csv('predictions.csv')

