#base mode
import pandas as pd
import numpy as np

#split, norm, metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import confusion_matrix, precision_score, recall_score
# from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
# from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import xgboost as xgb
# import lightgbm as lgb
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
# import scipy.stats as st
# from scipy import interp
import matplotlib.pyplot as plt
# from itertools import cycle
# from sklearn import svm, datasets
# from sklearn.metrics import roc_curve, auc
# from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix



def confu_cv(model):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # confu = cross_val_score(model, X_train, y_t, scoring="accuracy", cv=kf)
    return(confu)




if __name__ == '__main__':
    df = pd.read_csv('train.csv')
    df = df.fillna(df.mean())
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['active'] = df['last_trip_date'].dt.month >= 6
    df = pd.get_dummies(df, columns=['city', 'phone'])
    df.drop(['last_trip_date', 'signup_date'], axis=1, inplace=True)
    y_train = df.pop('active').values
    
    df = pd.read_csv('test.csv')
    df = df.fillna(df.mean())
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['active'] = df['last_trip_date'].dt.month >= 6
    df = pd.get_dummies(df, columns=['city', 'phone'])
    df.drop(['last_trip_date', 'signup_date'], axis=1, inplace=True)
    y_test = df.pop('active').values
    
    npz = np.load('X_filled_mice.npz')
    X_train = npz['X_filled_mice']
    npz = np.load('X_test.npz')
    X_test = npz['X_filled_mice']


    model = XGBClassifier(colsample_bytree=0.55, gamma=3,
                                 learning_rate=0.05, max_depth=3,
                                 min_child_weight=1.5, n_estimators=2200,
                                 reg_alpha=0.4640, reg_lambda=0.8571,
                                 subsample=0.5213, silent=1, nthread = -1, seed=5)
    model.fit(X_train, y_train)
    print(np.mean(cross_val_score(model, X_test, y_test)))

    #                              n_estimators,
    #                              reg_alpha=2,
                                #  reg_lambda=1,
    #                              subsample=0.75)
    # e1 = confu_cv(model)
    # print(np.mean(e1))
    # y_pred = model.fit(X_train, y_train).predict(X_test)
    # 
    # 
    # print(confusion_matrix(y_test, y_pred))
    # tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()


# ROC Curve
    # Run classifier with cross-validation and plot ROC curves
    # cv = StratifiedKFold(n_splits=6)
    # classifier = model
    #
    # tprs = []
    # aucs = []
    # mean_fpr = np.linspace(0, 1, 100)
    #
    # i = 0
    # for train, test in cv.split(X_filled_mice, y):
    #     probas_ = classifier.fit(X_filled_mice[train], y[train]).predict_proba(X_filled_mice[test])
    #     # Compute ROC curve and area the curve
    #     fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    #     tprs.append(interp(mean_fpr, fpr, tpr))
    #     tprs[-1][0] = 0.0
    #     roc_auc = auc(fpr, tpr)
    #     aucs.append(roc_auc)
    #     plt.plot(fpr, tpr, lw=1, alpha=0.3,
    #              label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    #
    #     i += 1
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
    #          label='Luck', alpha=.8)
    #
    # mean_tpr = np.mean(tprs, axis=0)
    # mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = np.std(aucs)
    # plt.plot(mean_fpr, mean_tpr, color='b',
    #          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
    #          lw=2, alpha=.8)
    #
    # std_tpr = np.std(tprs, axis=0)
    # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
    #                  label=r'$\pm$ 1 std. dev.')
    #
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()
