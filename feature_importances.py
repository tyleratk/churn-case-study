import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def get_x(file_name):
    npz = np.load(file_name)
    X_filled_knn = npz['X_filled_knn']
    return X_filled_knn
    
df = pd.read_csv('train.csv')
df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
df['active'] = (df['last_trip_date'].dt.month >= 6).astype(int)
cols = df.columns
y = df.pop('active').values
X = get_x('X_filled_knn.npz')

# X_train, X_test, y_train, y_test = train_test_split(X, y)

rf = RandomForestClassifier().fit(X, y)
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:\n")

for f in range(X.shape[1]):
    print(f+1, ': {} - {:.3f}'.format(cols[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="b", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()







