import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import preproceso
from sklearn.metrics import accuracy_score, f1_score, precision_score

def label_to_album(cluster_labels, y):
    # Initializing
    reference_labels = {}
    # For loop to run through each label of cluster label
    y = np.array(y[:, 0])
    for i in range(len(np.unique(kmeans.labels_))):
        index = np.where(cluster_labels == i, 1, 0)
        num = np.bincount(y[index == 1]).argmax()

        reference_labels[i] = num
    return reference_labels

dfLyrics = pd.read_csv("taylor-songs.csv")
df = preproceso.preprocess(dfLyrics)
X = df[["Lyrics_Embeddings"]]
X = pd.DataFrame(X.Lyrics_Embeddings.tolist(), index=X.index)
Y = np.array(df[["Album_num"]])

kmeans = KMeans(n_clusters=10).fit(X)
labels = kmeans.predict(X)
df['Cluster'] = labels
cluster_album = label_to_album(labels, Y)
album_labels = np.random.rand(len(kmeans.labels_))
for i in range(len(kmeans.labels_)):
    album_labels[i] = int(cluster_album[kmeans.labels_[i]])
df['PredictedAlbum'] = album_labels

print("La accuracy es:" , accuracy_score(album_labels,Y))
print("La precision es:" , precision_score(album_labels,Y, average='weighted'))
print("El f1 score es:" , f1_score(album_labels,Y, average='weighted'))

dfResultados = pd.DataFrame()
dfResultados['Title'] = df['Title']
dfResultados['Album'] = df['Album']
dfResultados['PredictedAlbum'] = df['PredictedAlbum']
dfResultados.to_csv("results.csv", index=False)