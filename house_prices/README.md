# Kaggle: house prices challenge
## Categorial encoding

Label encoding: convert each value in a column to a number. Example hard coding:
```
cat_to_num = dict(zip(sorted(df.LotConfig.unique()), range(5)))df.loc[:, 'LotConfig'] = df.LotConfig.map(cat_to_num)
```
Example using pandas:
```
df.loc[:, 'LotConfig'] = df.LotConfig.astype('category').cat.codes
```
Example using scikilearn:
```
from sklearn.preprocessing import LabelEncoder
df.loc[:, 'LotConfig'] = LabelEncoder().fit_transform(df.LotConfig)
```
Note that last two transformations are being made to the train dataset and do not guarantee a specific order. It is important to ensure that the categorical values will be replaced by the same numerical values in the test dataset. Using hard code in this situation may not be a bad idea.

There are also situations where you wish to convert categorical values into binary values. That is the case of LotShape column, which has four categorical values and it may be econded as binary. Hard code example:
```
df.loc[:, 'LotShape'] = df.LotShape.map({
    'Reg': 0,
    'IR1': 1,
    'IR2': 1,
    'IR3': 1 # IRs are very similar, then we assume they are equivalent
})
```
Using np.where, it will replace by 1 the values that satify the condition, and 0 otherwise:
```
df.loc[:, 'LotShape'] = np.where(
    df.LotShape.str.contains('IR'), 1, 0
)
```

One Hot Encoding: each value into a new binary column (see pandas.get_dummies)

```
df.loc[:, 'ExterQual'] = df.ExterQual.map({
    'TA': 0,
    'Gd': 1,
    'Ex': 2,
    'Fa': 2
})
df = pd.get_dummies(df, columns=['ExterQual'])
```