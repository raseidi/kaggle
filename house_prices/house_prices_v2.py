import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 20

df = pd.read_csv('data/train.csv')

numerical = [f for f in df.columns if df.dtypes[f] != 'object']
numerical.remove('SalePrice')
numerical.remove('Id')
categorical = [f for f in df.columns if df.dtypes[f] == 'object']
numerical = sorted(numerical)
categorical = sorted(categorical)


f = pd.melt(df, value_vars=numerical)
plt.figure(dpi=100, figsize=(16,8))
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")
g.savefig('plots/dist_numerical.png')

def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=65)

f = pd.melt(df, id_vars=['SalePrice'], value_vars=categorical)
plt.figure(dpi=100, figsize=(16,8))
g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value", "SalePrice")
g.savefig('plots/dist_categorical.png')