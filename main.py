import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

original_df = pd.read_csv('train.csv')
original_test = pd.read_csv('test.csv')

pd.set_option("display.max_columns", len(original_df.columns))
pd.set_option("display.max_rows", 30)

original_df.head()
original_df.SibSp.value_counts()
original_df.Parch.value_counts()
original_df.info()

# Probably this Cabin column can be dropped
original_df[~original_df.Cabin.isna()].Survived.value_counts()

df = original_df.copy()

# investigating if there is any relevance on
# Ticket column
tmp = df.copy()
aux = tmp.Ticket.apply(lambda x: x[:5])
print(*sorted(aux), sep='\n')
tmp.sort_values(by=['Ticket'], inplace=True)
tmp[tmp.Ticket == '110152']

# plot a heat map showing that most survivers
# have paid higher fares
tmp.loc[:, ['Fare', 'Survived']].sort_values(by=['Fare'])

#.............. pre processing ..............#
df.info()
df.Sex = df.Sex.apply(lambda x: 0 if x == 'female' else 1)

# there are only two missing values, so we fill with
# majority value
df.Embarked.value_counts()
df.Embarked.fillna('S', inplace=True)

# dropping Cabin because there are 
# a lot of missing values
df.drop('Cabin', axis=1, inplace=True)

df.info()

# It looks like that who is old (age > 60)
# was left behind. This must be discritized (ToDo)
df.Age.describe()
df.Age = df.Age.round()
df[df.Age > 60].Survived.value_counts()
df[df.Age < 1].Survived.value_counts()
sns.distplot(df.Age[(~df.Age.isna()) & (df.Survived == 0)])
sns.distplot(df.Age[(~df.Age.isna()) & (df.Survived == 1)])
plt.show()
df[df.Age.isna()].Survived.value_counts()
df[~df.Age.isna()].Survived.value_counts()
# it is not suitable to replace nan values by
# the mean of each class, because both classes
# have similar age means. By now, we drop it
df.drop(['Ticket', 'Name'], axis=1, inplace=True)
df.info()
df.head()

# applying log to normalize Fare column
sns.distplot(df.Fare)
sns.distplot(df.Fare.apply(lambda x: np.log(x) if x != 0 else x))
plt.show()
df.Fare = df.Fare.apply(lambda x: np.log(x) if x != 0 else x)

df.info()

# handling age column
# df.set_index('PassengerId', inplace=True)
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()
# sex, pclass and fare are the most correlated columns
# to survived; thus, we can consider pclass as a
# second target (not fare because it is continuous)
# and replace missing values by the mean of each
# group (Survived, Sex, Pclass)
tmp = df.groupby(by=['Survived', 'Sex', 'Pclass']).transform(lambda x: x.fillna(x.mean()))
df.Age = tmp.Age
df.Age = df.Age.round()

sns.distplot(df.Age)
plt.show()

# trasforming these columns into binaries
df.Embarked.value_counts()
df.Pclass.value_counts()

# df['Embarked_S'] = df.Embarked.apply(lambda x: 1 if x == 'S' else 0)
# df['Embarked_C'] = df.Embarked.apply(lambda x: 1 if x == 'C' else 0)
# df['Embarked_Q'] = df.Embarked.apply(lambda x: 1 if x == 'Q' else 0)

# df['Pclass_1'] = df.Pclass.apply(lambda x: 1 if x == '1' else 0)
# df['Pclass_2'] = df.Pclass.apply(lambda x: 1 if x == '2' else 0)
# df['Pclass_3'] = df.Pclass.apply(lambda x: 1 if x == '3' else 0)

# df.drop(['Embarked', 'Pclass'], axis=1, inplace=True)

df.info()

#.............. validating ..............#
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

# use accuracy_score for cross_validate
df.set_index('PassengerId', inplace=True)
df = df.sample(frac=1) # shuffling dataset
X_train = df.drop('Survived', axis=1)
y_train = df.Survived
clf = RandomForestClassifier(n_estimators=100)

results = cross_validate(clf, X_train, y_train, cv=5)
results['test_score'].mean()
 
# #.............. testing ..............#
df_test = pd.read_csv('test.csv')
df_test.info()
df_test.Fare = df_test.Fare.fillna(df_test.Fare.mean())
df_test.Sex = df_test.Sex.apply(lambda x: 0 if x == 'female' else 1)

df_test.drop('Cabin', axis=1, inplace=True)

df_test.drop(['Ticket', 'Name'], axis=1, inplace=True)

df_test.Fare = df_test.Fare.apply(lambda x: np.log(x) if x != 0 else x)

df_test.info()

tmp = df_test.groupby(by=['Sex', 'Pclass']).transform(lambda x: x.fillna(x.mean()))
df_test.Age = tmp.Age
df_test.Age = df_test.Age.round()

# df_test['Embarked_S'] = df_test.Embarked.apply(lambda x: 1 if x == 'S' else 0)
# df_test['Embarked_C'] = df_test.Embarked.apply(lambda x: 1 if x == 'C' else 0)
# df_test['Embarked_Q'] = df_test.Embarked.apply(lambda x: 1 if x == 'Q' else 0)

# df_test['Pclass_1'] = df_test.Pclass.apply(lambda x: 1 if x == '1' else 0)
# df_test['Pclass_2'] = df_test.Pclass.apply(lambda x: 1 if x == '2' else 0)
# df_test['Pclass_3'] = df_test.Pclass.apply(lambda x: 1 if x == '3' else 0)

# df_test.drop(['Embarked', 'Pclass'], axis=1, inplace=True)

# draft #

df_test['Embarked'] = df_test.Embarked.apply(lambda x: 0 if x == 'S' else x)
df_test['Embarked'] = df_test.Embarked.apply(lambda x: 1 if x == 'C' else x)
df_test['Embarked'] = df_test.Embarked.apply(lambda x: 2 if x == 'Q' else x)

df_test['Pclass'] = df_test.Pclass.apply(lambda x: 0 if x == '1' else x)
df_test['Pclass'] = df_test.Pclass.apply(lambda x: 1 if x == '2' else x)
df_test['Pclass'] = df_test.Pclass.apply(lambda x: 2 if x == '3' else x)

df['Embarked'] = df.Embarked.apply(lambda x: 0 if x == 'S' else x)
df['Embarked'] = df.Embarked.apply(lambda x: 1 if x == 'C' else x)
df['Embarked'] = df.Embarked.apply(lambda x: 2 if x == 'Q' else x)

df['Pclass'] = df.Pclass.apply(lambda x: 0 if x == '1' else x)
df['Pclass'] = df.Pclass.apply(lambda x: 1 if x == '2' else x)
df['Pclass'] = df.Pclass.apply(lambda x: 2 if x == '3' else x)

# end draft #


df_test.set_index('PassengerId', inplace=True)
X_test = df_test
X_train = df.drop('Survived', axis=1)
y_train = df.Survived

clf_final = RandomForestClassifier()
clf_final.fit(X_train, y_train)
X_test['y_pred'] = clf_final.predict(X_test)

final_pred = X_test[['y_pred']].copy()
final_pred.reset_index(inplace=True)
final_pred.rename({'y_pred': 'Survived'}, axis=1, inplace=True)
final_pred.to_csv('predictions_v2.csv', index=False)
