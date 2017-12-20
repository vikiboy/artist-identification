import numpy as np
import h5py
import os
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import normalize
import pandas as pd
import csv
from sklearn import svm
from sklearn.manifold import TSNE
from skdata.mnist.views import OfficialImageClassification
from tsne import bh_sne
from sklearn.decomposition import PCA

directory = "data/"


df = pd.read_csv('train_info.csv')
uniqueArtists = np.unique(df['artist'])

artistHash = {}
for i in range(len(uniqueArtists)):
    artistHash[uniqueArtists[i]] = i

labels = {}
X = []
Y = []

for index,row in df.iterrows():
    labels[row['filename']] = artistHash[row['artist']]

"""
Building the dataset from gram matrices
"""

for filename in os.listdir(directory):
    if filename.endswith(".h5"):
        file = os.path.join(directory, filename)
        if filename.endswith('.h5'):
            filename = filename[:-3]
            filename = filename + ".jpg"
        f = h5py.File(file,'r')

        # List all groups
        for i in range(len(f.keys())):
            if i == 3:
                a_group_key = list(f.keys())[i]
                data = list(f[a_group_key])
                for item in data:
                    # style = normalize(item, axis=1, norm='l1')
                    # style = style.flatten()
                    # X.append(style)
                    Y.append(labels[filename])

print np.unique(Y).shape[0]
"""
Zero mean data
"""
X = np.load("pcaoutput.npy")
print X.shape
arrM = X.transpose().astype('float64')

# arrM = np.array(X).transpose().astype('float64')
meanM = np.mean(arrM,axis=1)
stdM = np.mean(arrM,axis=1)
finalM = (arrM - meanM.reshape(meanM.shape[0],1))/stdM.reshape(stdM.shape[0],1)
mTsne = finalM.transpose()
print mTsne.shape

# X_embedded = bh_sne(mTsne)
#
# np.save('tsne512.npy',X_embedded)

# y_data = Y

# vis_x = vis_data[:, 0]
# vis_y = vis_data[:, 1]

# plt.scatter(vis_x, vis_y, c=y_data, cmap=plt.cm.get_cmap("jet", np.unique(Y).shape[0]))
# plt.colorbar(ticks=range(100))
# plt.clim(-0.5, 9.5)
# plt.show()

# X_embedded = TSNE(n_components=2).fit_transform(mTsne)
X_data = list(mTsne)
print len(X_data)
"""
Training and Test Data
"""
validLabels = np.unique(Y)

dataCount = len(X_data)
count = int(round(0.8*dataCount))

X_trainingData = X_data[:count]
Y_trainingData = Y[:count]

X_testData = X_data[count:]
Y_testData = Y[count:]
y_test = np.array(Y_testData)

"""
svm
"""
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X_trainingData, Y_trainingData)

y_pred = clf.predict(X_testData)

np.save('finalResults512.npy',y_pred)

print y_pred
print sum(y_test==y_pred)
