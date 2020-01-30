# kaggle_titanic

Project in construction.

Model employed currently [reaching 80% on kaggle](https://www.kaggle.com/raseidi/competitions) (top 10%).

## Dependencies

Currently, this project uses the following packages: *scikit-learn*, *pandas*, *numpy*, *matplotlib*, and *seaborn*. Also, some custom methods were employed for data visualization, and they are avaible in [here](https://github.com/raseidi/data_visualization).

## Preprocessing
-
## Pipeline

After the preprocessing step, an SVM was tunned via the Random Search strategy, with 100 iterations. The best parameters found were the following:
 
```
'C': 7110.060148342983,
'gamma': 0.01177348670454452,
'kernel': 'rbf'
```

Before submiting the final test, the model employed was validated through a 5-fold cros validation.
