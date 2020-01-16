import numpy as np
import pandas as pd

def handle_age(df):
    # Ideas for missing values:
    # 1. Replace missing values by mean of multi-targets;
    # find most relevant columns, then deal as if it
    # grouping by.
    # 2. Replace by mean of each class

    # Subsequently, classify ages in young, adult, old
    pass

def pre_process(df):   
    # normalizing Fare column
    df.Fare = df.Fare.apply(lambda x: np.log(x) if x != 0 else x)

    # transforming Sex column into binary
    df.Sex = df.Sex.apply(lambda x: 0 if x == 'female' else 1)
    
    # categorical to binary columns
    df['Embarked_S'] = df.Embarked.apply(lambda x: 1 if x == 'S' else 0)
    df['Embarked_C'] = df.Embarked.apply(lambda x: 1 if x == 'C' else 0)
    df['Embarked_Q'] = df.Embarked.apply(lambda x: 1 if x == 'Q' else 0)
    df['Pclass_1'] = df.Pclass.apply(lambda x: 1 if x == '1' else 0)
    df['Pclass_2'] = df.Pclass.apply(lambda x: 1 if x == '2' else 0)
    df['Pclass_3'] = df.Pclass.apply(lambda x: 1 if x == '3' else 0)

    # Name and Ticket are irrelevant
    # Cabin column has a lot of missing values
    # Embarked and Pclass were replaced by binary columns
    df.drop(['Name', 'Cabin', 'Ticket',\
        'Embarked', 'Pclass'], inplace=True)


