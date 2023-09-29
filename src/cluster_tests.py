import clustering as cl
import numpy as np
import time
from sklearn import metrics

def test():
    from sklearn.ensemble import IsolationForest
    db = np.load("data/nn_vectors/OUT_ERRORS_DEFINITIVE.npy")
    print(db.shape)
    isolation_forest = IsolationForest(n_estimators=200, max_samples=db.shape[0]//50, random_state=0)
    out = isolation_forest.fit_predict(db)
    print(out.shape)
    print(out[out == -1].shape)

def test_kmeans(db):
    print("----------------kmeans")
    t0 = time.time()
    cl.kmeans.fit( db )
    t_final = time.time() - t0

    print("time:", t_final)
    print("inertia:", cl.kmeans.inertia_)
    # Investigar más, pero encuentra pocos centroides útiles

def test_aff_prop(db):
    print("----------------affinity propagation")
    cl.affinity_prop.fit( db )
    cluster_centers_indices = cl.affinity_prop.cluster_centers_indices_
    labels = cl.affinity_prop.labels_

    n_clusters_ = len(cluster_centers_indices)

    print("Estimated number of clusters: %d" % n_clusters_)
    print(
        "Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(db, labels, metric="sqeuclidean")
    )
    # No hay suficiente memoria para usar nuestros datos, ni de cerca,

def test_mean_shift(db):
    print("----------------mean shift")
    bandwidth = cl.estimate_bandwidth2(db)
    print(bandwidth)

    n_instances = 64

    while n_instances < db.shape[0]:
        print("n_instances:",n_instances)
        indexes = np.random.choice(db.shape[0], n_instances, replace=False)
        small_db = db[indexes]
        cl.mean_shift.fit(small_db)
        labels = cl.mean_shift.labels_
        # cluster_centers = ms.cluster_centers_

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)

        print("number of estimated clusters : %d" % n_clusters_)

        n_instances *= 2

def test_DBSCAN_1(db):
    print("----------------DBSCAN")
    # Test 1: number of points
    n_instances = 64

    while n_instances < 100000:
        print("n_instances:",n_instances)
        indexes = np.random.choice(db.shape[0], n_instances, replace=False)
        small_db = db[indexes]
        cl.dbscan.fit(small_db)
        labels = cl.dbscan.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        n_instances *= 2

def test_DBSCAN_2(db):
    # Test 2: variability
    print("----------------DBSCAN")
    from sklearn.cluster import DBSCAN
    n_instances = 65536
    # n_instances = 131072

    indexes = np.random.choice(db.shape[0], n_instances, replace=False)
    small_db = db[indexes]

    for eps in np.linspace(0.2,0.8,10):

        print("EPS =",eps)

        dbscan = DBSCAN(eps=eps, min_samples=10)
        dbscan.fit(small_db)
        labels = dbscan.labels_

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        print('-'*50)
        print()

def test_birch_1(db):
    print("----------------Birch")

    n_instances = 256
    print('-'*50)

    while n_instances < db.shape[0]:
        indexes = np.random.choice(db.shape[0], n_instances, replace=False)
        small_db = db[indexes]#[:,:1]

        birch = Birch(threshold=0.5,branching_factor=50,n_clusters=None)
        print("n_instances:",n_instances)
        t = time()
        birch.fit(small_db)
        t = time()-t

        # Plot result
        labels = birch.labels_
        centroids = birch.subcluster_centers_
        n_clusters = np.unique(labels).size
        print("n_clusters : %d" % n_clusters)
        print("time :",t)
        print('-'*50)


def test_birch_2(db):
    from time import time
    from sklearn.cluster import Birch
    n_instances = int(1e6)
    # n_instances = int(1e3)
    print('-'*50)

    for branching_factor in np.geomspace(5,10000,10, dtype=int):
        print("branching_factor :",branching_factor)
        indexes = np.random.choice(db.shape[0], n_instances, replace=False)
        small_db = db[indexes]#[:,:1]

        birch = Birch(threshold=0.5,branching_factor=branching_factor,n_clusters=None)
        t = time()
        birch.fit(small_db)
        t = time()-t

        # Plot result
        labels = birch.labels_
        centroids = birch.subcluster_centers_
        n_clusters = np.unique(labels).size
        print("n_clusters : %d" % n_clusters)
        print("time :",t)
        print('-'*50)

def test_birch_3(db):
    from time import time
    from sklearn.cluster import Birch
    n_instances = int(1e6)
    branching_factor = 300
    # n_instances = int(1e3)
    print('-'*50)

    for threshold in np.linspace(0.5,5,10):
        print("threshold :",threshold)
        indexes = np.random.choice(db.shape[0], n_instances, replace=False)
        small_db = db[indexes]#[:,:1]

        birch = Birch(threshold=threshold,branching_factor=branching_factor,n_clusters=None)
        t = time()
        # birch.fit(small_db)
        birch.fit(db)
        t = time()-t

        # Plot result
        labels = birch.labels_
        centroids = birch.subcluster_centers_
        n_clusters = np.unique(labels).size
        print("n_clusters : %d" % n_clusters)
        print("time :",t)
        print('-'*50)
