import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fancyimpute import MICE
from xgboost import XGBClassifier # need to install xgboost / pip install
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


    
def get_x(file_name):
    npz = np.load(file_name)
    X_filled_knn = npz['X_filled_knn']
    return X_filled_knn


def rss(data, labels):
    rss = []
    for label in np.unique(labels):
        filter_data = data[labels == label]
        rss.append(((filter_data - filter_data.mean(axis = 0))**2).sum())
    return sum(rss)


def make_elbow_plot(data, ks, plotname=None):
    rsss = []
    for k in ks:
        print('Running with k', k)
        km = KMeans(n_clusters=k)
        km.fit(data)
        rsss.append(rss(data, km.labels_))
    fig, ax = plt.subplots()
    ax.plot(ks, rsss)
    ax.set_xlabel('k')
    ax.set_ylabel('RSS')
    ax.set_title('Elbow Plot')
    if plotname:
        plt.savefig(plotname)
    else:
        plt.show()
                                      
                                      
def get_clustering():                              
    df = pd.read_csv('train.csv')
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['active'] = (df['last_trip_date'].dt.month >= 6).astype(int)
    n_samples, n_features = df.shape
    n_clusters = 3
    X = get_x('X_filled_knn.npz')
    
    reduced_data = PCA(n_components=2).fit_transform(X)
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=50)
    kmeans.fit(reduced_data)
    
    h = .02
    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering (PCA-reduced data)\n')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()    


def get_feature_importances():
    X, y = get_data('train.csv')
    df = pd.read_csv('train.csv')
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['active'] = (df['last_trip_date'].dt.month >= 6).astype(int)
    cols = df.columns
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


def get_data(filename, from_pickle=False):
    '''
    Input: filename (csv if from_pickle=False,
                     pickfile if from_pickle=True)
                     
    Output: scaled X, y
    '''
    if from_pickle:
        df = pd.read_csv('train.csv')
        df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
        df['active'] = (df['last_trip_date'].dt.month >= 6).astype(int)
    
        y = df.pop('active').values
        npz = np.load(filename + '.npz')
        X_filled = npz[filename]
        return X_filled, y
        
    else:
        df = pd.read_csv(filename)
        df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
        df['signup_date'] = pd.to_datetime(df['signup_date'])
        df['active'] = (df['last_trip_date'].dt.month >= 6).astype(int)
        df = pd.get_dummies(df, columns=['city', 'phone'])
        df.drop(['last_trip_date', 'signup_date'], axis=1, inplace=True)

        y = df.pop('active').values
        X = df.values.astype(float)

        scaler = StandardScaler()
        X_scaled = MICE(n_imputations=6690).complete(X)
        X_scaled = scaler.fit_transform(X)
        return X, y
    
    
def get_model():

    model = XGBClassifier(colsample_bytree=0.55, gamma=3,
                          learning_rate=0.05, max_depth=3,
                          min_child_weight=1.5, n_estimators=2200,
                          reg_alpha=0.4640, reg_lambda=0.8571,
                          subsample=0.5213, silent=1, nthread = -1, seed=5)
    return model




if __name__ == '__main__':
    
    X, y = get_data('train.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # model = get_model()
    
    # get_clustering() # will plot kmeans model - you can change parameters above
    # get_feature_importances(): # bar graph of feature importances








