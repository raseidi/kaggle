import numpy as np
import pandas as pd

def pre_process_v1(file_path='train.csv'):
    df = pd.read_csv(file_path)
    df.Embarked.fillna('S', inplace=True)
    df.Fare.fillna(df.Fare.median(), inplace=True)

    # normalizing Fare column
    # df.Fare = df.Fare.apply(lambda x: np.log(x) if x != 0 else x)
    # transforming Sex column into binary
    df.Sex = df.Sex.apply(lambda x: 0 if x == 'female' else 1)
    # df.Sext.map('female': 0, 'male': 1)
    
    # categorical to binary columns
    df['Embarked_S'] = df.Embarked.apply(lambda x: 1 if x == 'S' else 0)
    df['Embarked_C'] = df.Embarked.apply(lambda x: 1 if x == 'C' else 0)
    df['Embarked_Q'] = df.Embarked.apply(lambda x: 1 if x == 'Q' else 0)
    df['Pclass_1'] = df.Pclass.apply(lambda x: 1 if x == '1' else 0)
    df['Pclass_2'] = df.Pclass.apply(lambda x: 1 if x == '2' else 0)
    df['Pclass_3'] = df.Pclass.apply(lambda x: 1 if x == '3' else 0)

    # Replacing missing values by the mean
    # of the columns (groups) most correlated
    # to the target (Survived)
    tmp = df.groupby(by=['Sex', 'Pclass']).transform(lambda x: x.fillna(x.mean()))
    df.Age = tmp.Age.round()
    # cutting into 3 intervals is enough, because it automatically
    # ranks youngest ones as the most survivors, and the
    # olddest as the who most died
    
    # next three steps were necessary only to set
    # the age intervals
    # tmp = df.copy()
    # tmp['AgeInterval'] = pd.cut(tmp.Age, 3)
    # tmp[['AgeInterval', 'Survived']].groupby(['AgeInterval'], as_index=False).mean().sort_values(by='AgeInterval', ascending=True)
    df['Age_0'] = df.Age.apply(lambda x: 1 if x <= 27 else 0)
    df['Age_1'] = df.Age.apply(lambda x: 1 if 27 < x <= 53 else 0)
    df['Age_2'] = df.Age.apply(lambda x: 1 if x > 53 else 0)

    # Here it is an intereseting insight, the higher
    # the Fare, most chance of surviving
    # tmp['AgeInterval'] = pd.cut(tmp.Fare, 3) # copying and pasting, column name is irrelevant
    # tmp[['AgeInterval', 'Survived']].groupby(['AgeInterval'], as_index=False).mean().sort_values(by='AgeInterval', ascending=True)
    df['Fare_0'] = df.Fare.apply(lambda x: 1 if x <= 171 else 0)
    df['Fare_1'] = df.Fare.apply(lambda x: 1 if 171 < x <= 341 else 0)
    df['Fare_2'] = df.Fare.apply(lambda x: 1 if x > 341 else 0)

    # Name and Ticket are irrelevant
    # Cabin column has a lot of missing values
    # Embarked, Pclass, Fare, and Age were replaced by 
    # binary columns
    df.drop(['Name', 'Cabin', 'Ticket',\
        'Embarked', 'Pclass', 'Age', 'Fare'],
        axis=1, inplace=True)

    df.set_index('PassengerId', inplace=True)
    return df

def pre_process_v2(file_path='train.csv'):
    df = pd.read_csv(file_path)
    df.Embarked.fillna('S', inplace=True)
    df.Fare.fillna(df.Fare.median(), inplace=True)

    # normalizing Fare column
    # df.Fare = df.Fare.apply(lambda x: np.log(x) if x != 0 else x)
    # transforming Sex column into binary
    # df.Sex.map({'female': 0, 'male': 1})
    df.Sex = df.Sex.apply(lambda x: 0 if x == 'female' else 1)

    df.Embarked = df.Embarked.apply(lambda x: 0 if x == 'S' else x)
    df.Embarked = df.Embarked.apply(lambda x: 1 if x == 'C' else x)
    df.Embarked = df.Embarked.apply(lambda x: 2 if x == 'Q' else x)
    df.Pclass = df.Pclass.apply(lambda x: 0 if x == '1' else x)
    df.Pclass = df.Pclass.apply(lambda x: 1 if x == '2' else x)
    df.Pclass = df.Pclass.apply(lambda x: 2 if x == '3' else x)

    tmp = df.groupby(by=['Sex', 'Pclass']).transform(lambda x: x.fillna(x.mean()))
    df.Age = tmp.Age.round()

    df.Age = df.Age.apply(lambda x: 0 if x <= 27 else x)
    df.Age = df.Age.apply(lambda x: 1 if 27 < x <= 53 else x)
    df.Age = df.Age.apply(lambda x: 2 if x > 53 else x)

    df.Fare = df.Fare.apply(lambda x: 0 if x <= 171 else x)
    df.Fare = df.Fare.apply(lambda x: 1 if 171 < x <= 341 else x)
    df.Fare = df.Fare.apply(lambda x: 2 if x > 341 else x)

    df = df.astype({'Age': 'int32', 'Fare': 'int64'})

    df.drop(['Name', 'Cabin', 'Ticket'],
        axis=1, inplace=True)

    df.set_index('PassengerId', inplace=True)
    return df