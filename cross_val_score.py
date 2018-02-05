from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# BOOSTING
from sklearn.ensemble import GradientBoostingClassifier
# from xgboost import XGBClassifier



df = pd.read_csv('train.csv')
df = df.fillna(df.mean())
df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['active'] = df['last_trip_date'].dt.month >= 6
df = pd.get_dummies(df, columns=['city', 'phone'])

df.drop(['last_trip_date', 'signup_date'], axis=1, inplace=True)

y = df.pop('active').values
X = df.values

kfold = KFold(n_splits=10, random_state=22) #split data into 10 parts
xyz=[]
accuracy=[]
std=[]

# classifiers=['Logistic Regression','KNN','Decision \
#               Tree','Naive Bayes','Random Forest']
classifiers = ['GradientBoostingClassifier', 'XGBClassifier']

# models = [LogisticRegression(), KNeighborsClassifier(n_neighbors=9), 
#           DecisionTreeClassifier(), GaussianNB(), 
#           RandomForestClassifier()]

models = [GradientBoostingClassifier(), XGBClassifier]

for idx, i in enumerate(models):
    model = i
    cv_result = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
    print('Done with {}'.format(classifiers[idx]))
 
new_models_dataframe2 = pd.DataFrame({'CVMean':xyz,'Std':std},
                                      index=classifiers)
print(new_models_dataframe2)









