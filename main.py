import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

original_train = pd.read_csv('train.csv')
original_test = pd.read_csv('test.csv')

original_train.info()
original_train['FareInterval'] = pd.cut(original_train.Fare, 3)
original_train[['FareInterval', 'Survived']].groupby(['FareInterval'], as_index=False).mean().sort_values(by='FareInterval', ascending=True)
original_train[original_train.Fare > 340].Survived

tmp = original_train.copy()
tmp['AgeInterval'] = pd.cut(original_train.Fare, 3)
tmp[['AgeInterval', 'Survived']].groupby(['AgeInterval'], as_index=False).mean().sort_values(by='AgeInterval', ascending=True)

pd.set_option("display.max_columns", len(original_train.columns))
pd.set_option("display.max_rows", 30)

original_train.head()
original_train.SibSp.value_counts()
original_train.Parch.value_counts()
original_train.Pclass.value_counts()

original_train.groupby(['Pclass', 'Survived']).value_counts()

sns.catplot(data=original_train, kind='bar',\
    x='Pclass', y='Sex', hue='Survived')
plt.show()

# Probably this Cabin column can be dropped
original_train[~original_train.Cabin.isna()].Survived.value_counts()

sns.distplot(df.Age[(~df.Age.isna()) & (df.Survived == 0)])
sns.distplot(df.Age[(~df.Age.isna()) & (df.Survived == 1)])
plt.show()

# applying log to normalize Fare column
sns.distplot(df.Fare)
sns.distplot(df.Fare.apply(lambda x: np.log(x) if x != 0 else x))
plt.show()
df.Fare = df.Fare.apply(lambda x: np.log(x) if x != 0 else x)
plt.show()

# df = df.sample(frac=1) # shuffling dataset

final_pred = X_test[['y_pred']].copy()
final_pred.reset_index(inplace=True)
final_pred.rename({'y_pred': 'Survived'}, axis=1, inplace=True)
final_pred.to_csv('predictions_v2.csv', index=False)
