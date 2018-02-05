from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from time import time
from sklearn import metrics
from sklearn.decomposition import PCA


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
        

def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))
                                      
                                      




if __name__ == '__main__':
    
    df = pd.read_csv('train.csv')
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['active'] = (df['last_trip_date'].dt.month >= 6).astype(int)
    cols = df.columns
    n_samples, n_features = df.shape
    n_clusters = 3
    sample_size = 300
    y = df.pop('active').values
    labels = y
    X = get_x('X_filled_knn.npz')
    # print('Loaded data\n')
    
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    # make_elbow_plot(X, range(1, 15))
    
    # bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
    #               name="k-means++", data=X)
    # 
    # bench_k_means(KMeans(init='random', n_clusters=n_clusters, n_init=10),
    #               name="random", data=X)
    
    # pca = PCA(n_components=n_clusters).fit(X)
    # bench_k_means(KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1),
    #               name="PCA-based",
    #               data=X)
    
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







