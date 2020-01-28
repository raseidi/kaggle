import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_rows = 81

os.chdir('house_prices/')

df = pd.read_csv('data/train.csv')
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

drop_columns = []
df.loc[0,:]

# continue from here;
df.LandContour.value_counts()
df.groupby('LandContour').SalePrice.mean()

df.LotShape.value_counts()
df.groupby('LotShape')['SalePrice'].mean().sort_values()
df.groupby('LotShape')['SalePrice'].std().sort_values()
df.loc[:, 'LotShape'] = df.LotShape.map({
    'Reg': 0,
    'IR1': 1,
    'IR2': 1,
    'IR3': 1 # IRs are very similar, then we assume they are equivalent
})

# this can be dropped, there are 1090 pave and only 4 grvl
df.Street.value_counts()
drop_columns.append('Street')

# RL has high variance in SalePrice;
# it has the most expansive and second most cheap house;
# maybe it can be dropped.
df.MSZoning.value_counts()
df.groupby(['MSZoning'])['SalePrice'].std().sort_values()
drop_columns.append('MSZoning')

#................... handling categorical values ...................#



#................... employing the model ...................#
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate

df_ = df.copy()
df_.set_index('Id', inplace=True)
X = df_.drop('SalePrice', axis=1).copy()
y = df.loc[:, 'SalePrice']

cv_scores = cross_validate(RandomForestRegressor(),
    X, y, scoring='neg_mean_squared_error')
# print(cross_validate.__doc__)






