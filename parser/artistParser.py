import numpy as np
import pandas as pd
import csv

df = pd.read_csv('train_info.csv')
uniqueArtists = np.unique(df['artist'])

artistHash = {}
for i in range(len(uniqueArtists)):
    artistHash[uniqueArtists[i]] = i
#
# for key,value in artistHash.items():
#     print key,value

labels = {}

for index,row in df.iterrows():
    labels[row['filename']] = artistHash[row['artist']]
