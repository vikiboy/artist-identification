import numpy as np
import h5py
import os
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import normalize
import pandas as pd
import csv

directory = "."


df = pd.read_csv('train_info.csv')
uniqueArtists = np.unique(df['artist'])

artistHash = {}
for i in range(len(uniqueArtists)):
    artistHash[uniqueArtists[i]] = i

labels = {}

for index,row in df.iterrows():
    labels[row['filename']] = artistHash[row['artist']]

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
                    style = normalize(item, axis=1, norm='l1')
                    print filename
