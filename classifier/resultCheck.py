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

directory = "data/"


df = pd.read_csv('train_info.csv')
uniqueArtists = np.unique(df['artist'])

artistHash = {}
for i in range(len(uniqueArtists)):
    artistHash[uniqueArtists[i]] = i

labels = {}
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
            if i == 0:
                a_group_key = list(f.keys())[i]
                data = list(f[a_group_key])
                for item in data:
                    Y.append(labels[filename])

print np.unique(Y).shape[0]
X_embedded= np.load('tsne64.npy')

X_data = list(X_embedded)
print len(X_data)
"""
Training and Test Data
"""
validLabels = np.unique(Y)

dataCount = len(X_data)
count = int(round(0.8*dataCount))

X_trainingData = X_data[:count]
Y_trainingData = Y[:count]
print len(X_trainingData)

X_testData = X_data[count:]
print len(X_testData)

Y_testData = Y[count:]
y_test = np.array(Y_testData)

"""
svm
"""
# clf = svm.SVC(decision_function_shape='ovo')
# clf.fit(X_trainingData, Y_trainingData)
#
# y_pred = clf.predict(X_testData)

y_pred=np.load('finalResults.npy')

print y_pred
print sum(y_test==y_pred)
