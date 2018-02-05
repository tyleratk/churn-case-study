from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv('train.csv')
df = df.fillna(df.mean())
df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['active'] = (df['last_trip_date'].dt.month >= 6).astype(int)
df = pd.get_dummies(df, columns=['city', 'phone'])

df.drop(['last_trip_date', 'signup_date'], axis=1, inplace=True)

y = df.pop('active').values
X = df.values

X_train, X_test, y_train, y_test = train_test_split(X, y)

original_params = {}

params = [('First', {'n_estimators': 1000, 
           'learning_rate':.1,                          
           'max_depth':3, 'max_features':'sqrt'}),
          ('Second', {'n_estimators': 1250, 'learning_rate':.01,                            
                      'max_depth':3, 'max_features':'sqrt'})]
plt.figure()

for label, setting in params:
    params = dict(original_params)
    params.update(setting)
    
    model = GradientBoostingClassifier(**params).fit(X_train, y_train)
    
    test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)
    
    for i, y_pred in enumerate(model.staged_decision_function(X_test)):
        # clf.loss_ assumes that y_test[i] in {0, 1}
        test_deviance[i] = model.loss_(y_test, y_pred)
        
    plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5],
              '-', label=label)
plt.show()