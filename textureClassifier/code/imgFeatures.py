import os
import numpy as np
import pandas as pd
import cPickle as pickle
from skimage import feature
from matplotlib import pyplot as plt
import cv2
from parser import readImages,returnLabels
from localbinarypatterns import LocalBinaryPatterns
from sklearn import mixture
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

directory = "../data/"
filePath = 'train_info.csv'

def getImg(imgListRow):
    """
    Read an image from the image list
    """
    imgPath = ""
    artistLabel = 0
    styleLabel = 0

    for key,value in imgListRow.items():
        imgPath = key
        artistLabel = value['artist']
        styleLabel = value['style']

    img = cv2.imread(imgPath)
    return img,artistLabel,styleLabel

def getLBP(img):
    """
    Compute LBP for an image
    """
    desc = LocalBinaryPatterns(24, 8)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    LBPhist = desc.describe(grayImg)
    return LBPhist

def getColorHistograms(img):
    """
    Compute color histograms for an image
    """
    color = ('b','g','r')
    histList = []
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        histarray = np.asarray(histr).reshape(1,256)
        histList.append(histarray/np.sum(histarray))
    return histList

def buildData(imgList,save=True):
    """
    Build Training Data array for GMM - [256R+256G+256B+24LBP] dimensional array
    returns list - {idx,feature,artist,style}
    """
    data = []

    for i in range(len(imgList)):
        print "Processing Image : ",i,"/",len(imgList)

        currentImgRow = imgList[i]
        currentImg,artistLabel,styleLabel = getImg(currentImgRow)
        lbpImg = getLBP(currentImg)
        histImg = getColorHistograms(currentImg)
        f = []
        for j in range(len(histImg)):
            a = histImg[j][0]
            a = a.tolist()
            f += a
        f += lbpImg.tolist()
        featureVec = np.array(f).reshape(1,794)
        currentImgData = {}
        currentImgData['idx']=i
        currentImgData['feature']=featureVec
        currentImgData['style']=styleLabel
        currentImgData['artist']=artistLabel
        data.append(currentImgData)

    if save==True:
        pickle.dump(data,open("dataImg.p","wb"))

    return data

def initGMM(noClusters):
    """
    Initialize a GMM with a specified number of clusters
    """
    gmm = mixture.BayesianGaussianMixture(n_components=noClusters, covariance_type='full',max_iter=1000)
    return gmm

def createSchool(Y_artist,Y_style,artistSchoolDiv,styleSchoolDiv):
    """
    Create Label Groups
    """
    # Dividing artist labels
    count = -1
    artistDict = {}
    for i in range(0,1583,artistSchoolDiv):
        count+=1
        if i+artistSchoolDiv<1582:
            for j in range(i,i+artistSchoolDiv):
                artistDict[j]=count
        else:
            for j in range(i,1583):
                artistDict[j]=count

    # Dividing style labels
    count = -1
    styleDict = {}
    for i in range(0,136,styleSchoolDiv):
        count+=1
        if i+styleSchoolDiv<135:
            for j in range(i,i+styleSchoolDiv):
                styleDict[j]=count
        else:
            for j in range(i,136):
                styleDict[j]=count

    Y_artist_school = []
    Y_style_school = []

    # Update the Y labels

    for i in range(len(Y_artist)):
        Y_artist_school.append(artistDict[Y_artist[i]])
        Y_style_school.append(styleDict[Y_style[i]])

    return Y_artist_school,Y_style_school

def GMMData(data,trainCount,artistSchoolDiv,styleSchoolDiv):

    """
    Preparing training and test data
    """
    X = []
    Y_artist = []
    Y_style = []
    X_train = []
    X_test = []
    Y_train_artist = []
    Y_train_style = []
    Y_test_artist = []
    Y_test_style = []


    for i in range(len(data)):
        X.append(data[i]['feature'])
        Y_artist.append(data[i]['artist'])
        Y_style.append(data[i]['style'])

    X = np.array(X)
    X = X.reshape(X.shape[0],X.shape[2]) # 7538 x 794 numpy array

    X_train = X[0:trainCount,:]
    X_test = X[trainCount:len(X),:]

    # Creating art and style schools - reducing number of labels to groups of labels
    Y_artist_school,Y_style_school = createSchool(Y_artist,Y_style,artistSchoolDiv,styleSchoolDiv)

    artistClusters = np.max(np.unique(Y_artist_school))+1
    styleClusters = np.max(np.unique(Y_style_school))+1

    Y_artist = np.array(Y_artist_school)
    Y_style = np.array(Y_style_school)

    Y_train_artist = Y_artist[0:trainCount,]
    Y_test_artist = Y_artist[trainCount:len(X),]

    Y_train_style = Y_style[0:trainCount,]
    Y_test_style = Y_style[trainCount:len(X),]

    print "Data Initialized"

    pca = PCA(n_components=300)
    xTrain = pca.fit_transform(X_train)
    # xTrain = X_train
    print xTrain.shape
    xTest = pca.fit_transform(X_test)
    # xTest = X_test
    print xTest.shape

    return xTrain,xTest,Y_train_artist,Y_test_artist,Y_train_style,Y_test_style,artistClusters,styleClusters


def fitGMM(xTrain,artistClusters,styleClusters,save=True):
    """
    Fit a GMM model with data
    """

    """
    Initializing GMM model
    """

    print "GMM initializing"
    gmm_artist = initGMM(artistClusters)
    gmm_artist.fit(xTrain)

    gmm_style = initGMM(styleClusters)
    gmm_style.fit(xTrain)

    print "GMM Model trained"

    if save==True:
        filePath_1 = "GMM_artist_trained_" + str(artistClusters)
        pickle.dump(gmm_artist,open(filePath_1,"wb"))

        filePath_2 = "GMM_style_trained_" + str(styleClusters)
        pickle.dump(gmm_style,open(filePath_2,"wb"))

    """
    Prediction on test data
    """

    return gmm_artist,gmm_style

def assignClusterLabels(xTrain,yTrain,noClusters,trainCount,model):
    """
    Assign Labels to Clusters
    """
    predLabels = model.predict(xTrain)
    trueClusterLabels = np.zeros([noClusters,noClusters])

    for i in range(noClusters):
        for k in range(trainCount):
            if predLabels[k] == i:
                trueClusterLabels[i][yTrain[k]] = trueClusterLabels[i][yTrain[k]] + 1

    clusterLabels = np.argmax(trueClusterLabels,axis=1)
    uniqueLabels = np.unique(clusterLabels).shape[0]
    return clusterLabels,uniqueLabels


def predictGMM(testData,model,clusterLabels,testLabels):
    """
    Returns predicted labels, accuracy
    """
    predData = model.predict(testData)

    predLabels = np.zeros(testLabels.shape)

    for i in range(predData.shape[0]):
        predLabels[i] = clusterLabels[predData[i]]

    testCount = testLabels.shape[0]
    accuracy = np.sum(predLabels==testLabels)/float(testCount)
    return accuracy

def saveOutputGMM(artistAccuracy,styleAccuracy,artistClusters,styleClusters,uniqueArtistLabels,uniqueStyleLabels):

    artistFilePath = "artistAccuracy_" + str(artistClusters) +".txt"
    artistResults = "Artist Accuracy : " + str(artistAccuracy) + ", Number of Clusters : " + str(artistClusters) + ", Number of Unique Labels : " + str(uniqueArtistLabels)

    tA = open(artistFilePath,"w")
    tA.write(artistResults)
    tA.close()

    styleFilePath = "styleAccuracy_" + str(styleClusters) +".txt"
    styleResults = "Style Accuracy : " + str(styleAccuracy) + ", Number of Clusters : " + str(styleClusters) + ", Number of Unique Labels : " + str(uniqueStyleLabels)

    sA = open(styleFilePath,"w")
    sA.write(styleResults)
    sA.close()

    print "Results Written to File"

def otherClassifiers(xTrain,xTest,Y_train_artist,Y_test_artist,Y_train_style,Y_test_style,classifier = 'svm'):
    """
    Write other classifiers
    """
    if classifier == 'svm':
        clf_artist = svm.SVC(decision_function_shape='ovo')
        clf_style = svm.SVC(decision_function_shape='ovo')

    elif classifier == 'adaboost':
        clf_artist =  AdaBoostClassifier()
        clf_style =  AdaBoostClassifier()

    elif classifier == 'nb':
        clf_artist = GaussianNB()
        clf_style = GaussianNB()

    clf_artist.fit(xTrain,Y_train_artist)
    artistPred = clf_artist.predict(xTest)
    accuracyArtist = np.sum(artistPred==Y_test_artist)/float(xTest.shape[0])

    clf_style.fit(xTrain,Y_train_style)
    stylePred = clf_style.predict(xTest)
    accuracyStyle = np.sum(stylePred==Y_test_style)/float(xTest.shape[0])

    return accuracyArtist,accuracyStyle

def saveOutputOtherClassifiers(artistAccuracy,styleAccuracy,classifierName,artistClusters,styleClusters):

    artistFilePath = "artistAccuracy_" + str(classifierName) + str(artistClusters) +".txt"
    artistResults = "Artist Accuracy : " + str(artistAccuracy) + ", Number of Clusters : " + str(artistClusters)

    tA = open(artistFilePath,"w")
    tA.write(artistResults)
    tA.close()

    styleFilePath = "styleAccuracy_" + str(classifierName) + str(styleClusters) +".txt"
    styleResults = "Style Accuracy : " + str(styleAccuracy) + ", Number of Clusters : " + str(styleClusters)

    sA = open(styleFilePath,"w")
    sA.write(styleResults)
    sA.close()

    print "Results Written to File"


def main():

    # artistLabels,styleLabels = returnLabels(filePath,save=False)
    # imageList = readImages(directory,artistLabels,styleLabels,save=False)
    # data = buildData(imageList,save=True)
    trainCount = 6500

    artistSet = [10,20,30,40,50,60,70,80,90,100,150]
    styleSet = [5,10,15,20,30]
    classifiers = ['svm','adaboost','nb']

    data = pickle.load(open("dataImg.p","rb"))

    for styleSchoolDiv in styleSet:
        for artistSchoolDiv in artistSet:
            print "artistCount : ",artistSchoolDiv," styleCount : ",styleSchoolDiv
            xTrain,xTest,Y_train_artist,Y_test_artist,Y_train_style,Y_test_style,artistClusters,styleClusters = GMMData(data,trainCount,artistSchoolDiv,styleSchoolDiv)

            for classifier in classifiers:
                accuracyArtist,accuracyStyle=otherClassifiers(xTrain,xTest,Y_train_artist,Y_test_artist,Y_train_style,Y_test_style,classifier)
                print str(classifier)," Accuracy Artist : ",accuracyArtist
                print str(classifier)," Accuracy Style : ",accuracyStyle
                saveOutputOtherClassifiers(accuracyArtist,accuracyStyle,classifier,artistClusters,styleClusters)

            # GMM
            gmm_artist,gmm_style = fitGMM(xTrain,artistClusters,styleClusters,save=False)

            clusterLabelsArtist,uniqueArtistLabels = assignClusterLabels(xTrain,Y_train_artist,artistClusters,trainCount,gmm_artist)

            accuracyArtist = predictGMM(xTest,gmm_artist,clusterLabelsArtist,Y_test_artist)

            clusterLabelsStyle,uniqueStyleLabels = assignClusterLabels(xTrain,Y_train_style,styleClusters,trainCount,gmm_style)

            accuracyStyle = predictGMM(xTest,gmm_style,clusterLabelsStyle,Y_test_style)

            saveOutputGMM(accuracyArtist,accuracyStyle,artistClusters,styleClusters,uniqueArtistLabels,uniqueStyleLabels)

            print "GMM Accuracy Artist :",accuracyArtist
            print "GMM Accuracy style :",accuracyStyle

if __name__ == '__main__':
    main()
