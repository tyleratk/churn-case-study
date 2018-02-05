import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt


# ------------------------- notes ----------------------------
'''
    converted last_trip_date to datetime to get if they are an active user
        - need to test
    dropped a whole bunch of columns that should be changed to dummies
    cross_val_score of ~70%
    use cross_val on many different models
    signup_date within 30 days of june 1st
'''


def get_returning_cust(date):
    '''
    input: string of date from dataframe
    output: 0 or 1 whether the date was after june 1st
    '''
    d = datetime(2014,6,1)
    return 1 if date >= d else 0


def clean_df(df):
    '''
    input: dateframe
    output: cleaned dateframe
    
    NEED TO CHANGE -> fillna, add dummies
    '''
    df.fillna(0, inplace=True)
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['active'] = df['last_trip_date'].apply(get_returning_cust)
    df = pd.get_dummies(df, columns=['city', 'phone'])
    df.drop(['last_trip_date', 'signup_date'], axis=1, inplace=True)

    return df
    
    




if __name__ == '__main__':

    train = pd.read_csv('train.csv')
    # test = pd.read_csv('test.csv')

    clean_train = clean_df(train)
    # clean_test = clean_df(test)

    y = clean_train.pop('active').values
    # clean_train.drop('last_trip_date', axis=1, inplace=True)
    # clean_train.drop('signup_date', axis=1, inplace=True)
    # clean_train.drop('phone', axis=1, inplace=True)
    # clean_train.drop('city', axis=1, inplace=True)

    X = clean_train.values
    # 
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # 
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

