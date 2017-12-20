import os
import numpy as np
import pandas as pd
import cPickle as pickle

directory = "../data/"
filePath = 'train_info.csv'

def returnLabels(filePath,save=False):
    """
    Assigning labels to each file
    """
    df = pd.read_csv(filePath)
    uniqueArtists = np.unique(df['artist'])
    uniqueStyle = np.unique(df['style'])

    styleHash = {}
    styleCount = 0

    for i in range(len(uniqueStyle)):
        if str(uniqueStyle[i]) == 'nan':
            styleHash['nan'] = 0
        else:
            styleCount +=1
            styleHash[uniqueStyle[i]] = styleCount

    artistHash = {}

    for i in range(len(uniqueArtists)):
        artistHash[uniqueArtists[i]] = i

    artistLabels = {}
    styleLabels = {}
    X = []
    Y = []

    for index,row in df.iterrows():
        artistLabels[row['filename']] = artistHash[row['artist']]
        if str(row['style'])=='nan':
            styleLabels[row['filename']] = styleHash[str('nan')]
        else:
            styleLabels[row['filename']] = styleHash[row['style']]

    if save==True:
        pickle.dump(artistLabels,open("artistLabels.p","wb"))
        pickle.dump(styleLabels,open("styleLabels.p","wb"))

    return artistLabels,styleLabels

def readImages(dirPath,artistLabels,styleLabels,save=False):
    """
    Read images from dataset
    returns imageList -
    imageList[i]['filePath']['artist']
    imageList[i]['filePath']['style']
    """
    imageList = []
    for filename in os.listdir(dirPath):
        if filename.endswith(".png"):
            file = os.path.join(dirPath,filename)

            if filename.endswith("_500.png"):
                filename = filename[:-8]
                filename = filename + ".jpg"


            # Retrieve style and artist labels
            artist = artistLabels[filename]
            style = styleLabels[filename]

            # Save style in a dict
            label = {}
            label['artist']=artist
            label['style']=style

            # Save file name and labels in a dict
            fileLabelList = {}
            fileLabelList[file]=label

            # Save it to a list
            imageList.append(fileLabelList)

    if save==True:
        pickle.dump(imageList,open("imageList.p","wb"))

    return imageList


def main():
    artistLabels,styleLabels = returnLabels(filePath,save=False)
    imageList = readImages(directory,artistLabels,styleLabels,save=False)

if __name__ == '__main__':
    main()
