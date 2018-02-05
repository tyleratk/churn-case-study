from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd



df = pd.read_csv('train.csv')
df = df.fillna(df.mean())
df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['active'] = df['last_trip_date'].dt.month >= 6
df = pd.get_dummies(df, columns=['city', 'phone'])

df.drop(['last_trip_date', 'signup_date'], axis=1, inplace=True)

y = df.pop('active').values
X = df.values

print('Loaded x, y')

tuned_parameters = {'loss': ['deviance', 'exponential'],
                    'learning_rate': [.001, .01, .1],
                    'n_estimators': [100, 250, 500],
                    'max_depth': [2, 3, 4, 5]}
                
model = GradientBoostingClassifier()  
print('Running grid-search...')  
gs = GridSearchCV(model, tuned_parameters, verbose=True)
gs.fit(X, y)

print(gs.cv_results_)







