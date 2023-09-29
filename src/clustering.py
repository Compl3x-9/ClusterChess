import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_samples, silhouette_score

# ANÁLISIS DE DATOS PREVIO
def varianzas_datos(db):
    return np.var(db, axis=0)

def medias_datos(db):
    return np.mean(db, axis=0)

def valores_0_datos(db):
    return np.sum( db == 0 , axis=0)

def preproc_remove_zero_value_dims(db, model):
    c0s = {
        "dif2vec":[ 0,  1,  2,  4,  6,  7,  8,  9, 10, 11, 13, 14, 15, 16, 17, 19, 20,
       21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 34, 36, 37, 38, 39, 40, 41,
       43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60,
       61, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80,
       81, 82, 83, 84, 85, 86, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99],
        "mov2vec":[  0,   1,   2,   4,   6,   7,   8,   9,  10,  11,  13,  14,  15,
        16,  17,  19,  20,  21,  22,  24,  25,  26,  27,  28,  30,  31,
        32,  34,  36,  37,  38,  39,  40,  41,  43,  44,  45,  46,  47,
        48,  49,  50,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,
        63,  64,  65,  66,  67,  69,  70,  71,  72,  73,  74,  75,  76,
        77,  79,  80,  81,  82,  83,  84,  85,  86,  88,  89,  90,  91,
        92,  94,  95,  96,  97,  98,  99, 100, 101, 102, 104, 106, 107,
       108, 109, 110, 111, 113, 114, 115, 116, 117, 119, 120, 121, 122,
       124, 125, 126, 127, 128, 130, 131, 132, 134, 136, 137, 138, 139,
       140, 141, 143, 144, 145, 146, 147, 148, 149, 150, 152, 153, 154,
       155, 156, 157, 158, 159, 160, 161, 163, 164, 165, 166, 167, 169,
       170, 171, 172, 173, 174, 175, 176, 177, 179, 180, 181, 182, 183,
       184, 185, 186, 188, 189, 190, 191, 192, 194, 195, 196, 197, 198,
       199],
        "pos2vec1":[ 1, 14, 15, 30, 33, 38, 45, 46, 48, 73, 78, 83, 86, 91, 93],
        "pos2vec2":[ 1,  2,  3,  4,  6,  9, 10, 13, 14, 15, 17, 18, 20, 21, 24, 26, 31,
       34, 35, 36, 38, 39, 44, 46, 48, 49, 51, 52, 54, 58, 60, 61, 62, 67,
       68, 69, 70, 71, 72, 73, 76, 77, 78, 79, 81, 83, 85, 89, 90, 92, 96,
       97]
           }
    ret = np.delete(db, c0s[model], axis=1)
    return ret

def preproc_accum_low_density_dims(db, model):
    orders = {
        "mov2vec":[[ 2, 5,14],[17, 20, 29]],
        "dif2vec":[[ 2, 5,14]],
        "pos2vec1":[[21,67,34,73,74,35,71,2,39,60,20,7,62,77],[9,8,18,44,51,79,52,6,13,49,
        61,65,59,33,69,41,84,12,26,30,48]],
        "pos2vec2":[[6,33,24,36,37,11,45,5,15,18,26,2,32,0,16,41,39,30,23,22,25,7,20,4,
        46,17,21,31,12,10,8,40,28]]
    }

    order = orders[model]
    base = np.delete(db, sum(order,[]), axis=1)
    # print(db.shape)

    c = np.zeros((db.shape[0]))

    for l in order:
        for i in l:
            mask = db[:,i] != 0
            for j in range(db.shape[0]):
                if db[j,i] != 0:
                    c[j] = db[j,i]
        base = np.c_[base, c]
        c = np.zeros((db.shape[0]))

    return np.asarray(base)

def train_save_kmeans(inp_file, n_clusters, out_model, out_labels=""):
    kmeans = KMeans(
        n_clusters=n_clusters,init='k-means++',n_init="auto",max_iter=200,tol=0
    )
    db = np.load(inp_file)

    cluster_labels = kmeans.fit_predict(db)

    cluster_sizes = [(i, cluster_labels[cluster_labels==i].size) for i in range(n_clusters)]
    print(cluster_sizes)

    from joblib import dump

    dump(kmeans, out_model)
    if out_labels != "":
        np.savetxt(out_labels, cluster_labels, fmt='%4.0f')

def test():
    fname = "OUT_ERRORS_DEFINITIVE.npy"
    db_init = np.load("data/nn_vectors/" + fname)

    print(db_init.shape)

    isolation_forest = IsolationForest(n_estimators=200, max_samples=db.shape[0]//50,
                                       contamination=0.05, random_state=0)

    out = isolation_forest.fit_predict(db_init)

    print(out[out == -1].shape)

    db = db_init[out == 1]

    if False:
        silhouette_scores = []
        X = list(range(5,100))
        for i in X:
            print(i,"clusters")
            kmeans = KMeans(
                n_clusters=i,init='k-means++',n_init="auto",max_iter=50,tol=0.0000001
            )

            # inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
            # fit k-means
            cluster_labels = kmeans.fit_predict(db)

            # silhouette_avg = silhouette_score(db, cluster_labels)
            silhouette_avg = kmeans.inertia_
            print(
                "For n_clusters =",
                i,
                "The inertia is :",
                silhouette_avg,
            )
            silhouette_scores.append(silhouette_avg)
        # print()
        plt.title("Curva de la inercia de k-medias según el número de clusters")
        plt.xlabel("Número de clusters")
        plt.ylabel("Inercia del algoritmo")
        plt.plot(X, silhouette_scores, linestyle='--', marker='o', color='b')
        plt.show()

    n_clusters_final = 50
    kmeans = KMeans(
        n_clusters=n_clusters_final,init='k-means++',n_init="auto",max_iter=200,tol=0
    )

    cluster_labels = kmeans.fit_predict(db)
    silhouette_avg = kmeans.inertia_

    print(silhouette_avg)
    cluster_sizes = [cluster_labels[cluster_labels==i].size for i in range(n_clusters_final)]
    cluster_sizes.sort()
    print(cluster_sizes)

    final_labels = kmeans.predict(db_init)

    np.savetxt('CLUSTERING_LABELS.txt', final_labels, fmt='%4.0f')
