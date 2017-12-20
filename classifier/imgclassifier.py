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

directory = "dataImgs/"

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

for filename in os.listdir(directory):
    if filename.endswith(".png"):
        file = os.path.join(directory, filename)
        if filename.endswith('_500.png'):
            filename = filename[:-8]
            filename = filename + ".jpg"
        Y.append(labels[filename])
        X.append(filename)

for x,y in zip(X,Y):
    print x,y
