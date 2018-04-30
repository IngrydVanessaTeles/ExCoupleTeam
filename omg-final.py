import numpy
import os
import datetime
import librosa
import cv2
import csv
from pandas import DataFrame
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam, Adagrad
from keras.models import load_model
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Reshape
from keras.models import Model
from KEF.Metrics import metrics
from keras.utils import np_utils
from scipy.stats import pearsonr
import panda
import statistics
from collections import Counter
import matplotlib.pyplot as plt
import keras

numberOfAugmentedSamples = 10
framesPerSecond = 25
stride = 25


def slice_signal(signal, sliceSize, stride=0.5):
        """ Return windows of the given signal by sweeping in stride fractions
            of window
        """
        #print "Dimension:", signal.shape
        #print "Audio:", signal
        #print "Audio:", signal.shape

        sliceSize = 16000 * sliceSize
        slices = []
        currentFrame = 0

        # print "------"
        # print "Total Signal Size:", len(signal)
        # print "In Seconds: ", len(signal) / sliceSize
        # print "Each Slide: ", sliceSize
        # print "Number of Slices: ", len(signal) / sliceSize*stride
        # print "------"

        while currentFrame+sliceSize < len(signal):
            currentSlide = signal[currentFrame:int(currentFrame+sliceSize)]
            slices.append(currentSlide)
            currentFrame = int(currentFrame+sliceSize*stride)
            #print "Shape Current slide:", len(currentSlide)

        #print "Shape Slices:", len(slices)
        #raw_input("here")
        return numpy.array(slices)


        assert signal.ndim == 1, signal.ndim
        n_samples = signal.shape[0]
        offset = int(window_size * stride)
        slices = []
        for beg_i, end_i in zip(range(0, n_samples, offset),
                                range(window_size, n_samples + offset,
                                      offset)):
            if end_i - beg_i < window_size:
                break
            slice_ = signal[beg_i:end_i]
            if slice_.shape[0] == window_size:
                slices.append(slice_)
        return numpy.array(slices, dtype=numpy.int32)

def orderClassesFolder(folder):
    classes = os.listdir(folder)
    return classes

def preProcess(dataLocation, augment=False):

        #assert (
        #not dataLocation == None or not self.preProcessingProperties == ""), "No data or parameter sent to preprocess!"

        #assert (len(self.preProcessingProperties[
        #                0]) == 2), "The preprocess have the wrong shape. Right shape: ([imgSize,imgSize])."


        #videoClip = VideoFileClip(dataLocation)
        #print "Audio:", dataLocation

        fftsize = 1024
        hop_length = 512

        wav_data, sr = librosa.load(dataLocation, mono=True, sr=16000)

        signals = slice_signal(wav_data, 1, 1)
        signals2 = []
        for wav_data in signals:
            D = librosa.stft(wav_data, fftsize, hop_length=hop_length)
            magD = numpy.abs(D)
            #print "SHape:", magD.shape
            #print "Max:", magD.max()
            #print "Min:", magD.min()
            #raw_input("here")

            magD = numpy.array(cv2.resize(magD, (96, 520)))
            magD = numpy.expand_dims(magD, axis=0)
            #magD /= 256
            #print "Shape:", magD


            signals2.append(magD)

        #raw_input("here")
        return numpy.array(signals2)

def loadData(dataFolder, augment):

        def shuffle_unison(a, b):
            assert len(a) == len(b)
            p = numpy.random.permutation(len(a))
            return a[p], b[p]

        assert (not dataFolder == None or not dataFolder == ""), "Empty Data Folder!"

        dataX = []
        dataLabels = []
        classesDictionary = []
        dataArousal = []
        dataValence = []

        emotions = orderClassesFolder(dataFolder + "/")
        print("Emotions reading order: " + str(emotions))

        lastImage = None

        numberOfVideos = 0
        emotionNumber = 0
        for e in emotions:

            emotionNumber = emotionNumber+1
            loadedDataPoints = 0
            classesDictionary.append("'" + str(emotionNumber) + "':'" + str(e) + "',")


            time = datetime.datetime.now()

            #print dataFolder + "/" + v+"/"+dataPointLocation

            for audio in os.listdir(dataFolder + "/" + e+"/"):

             #if numberOfVideos < 30:
                dataPoint = preProcess(dataFolder + "/" + e+"/"+audio)

                for audio2 in dataPoint:
                    #print "SHape:", audio.shape
                    dataX.append(audio2)
                    dataLabels.append(e+"-"+audio)
                    loadedDataPoints = loadedDataPoints + 1
                numberOfVideos = numberOfVideos + 1


            print(
                "--- Emotion: " + str(e) + "(" + str(loadedDataPoints) + " Data points - " + str(
                    (datetime.datetime.now() - time).total_seconds()) + " seconds" + ")")


        #print "Labels before:", dataLabels
        #dataLabels = np_utils.to_categorical(dataLabels, emotionNumber)
        #print "Labels After:", dataLabels
        dataX = numpy.array(dataX)
        #dataX = numpy.swapaxes(dataX, 1, 2)

        # print "Shape Labels:", dataLabels.shape
        # print "Shape DataX:", dataX.shape
        #dataLabels = numpy.array(dataLabels).reshape((len(dataLabels), 2))

#        dataX = dataX.astype('float32')


        #dataX, dataLabels = shuffle_unison(dataX,dataLabels)

#        print "dataX shape:", numpy.array(dataX).shape
        #raw_input("here")

        dataX = numpy.array(dataX)
        dataLabels = numpy.array(dataLabels)

        return dataX, dataLabels


trainX, trainArquivo = loadData("data/OMG/trainAudiopp", augment=False)
testX, testArquivo = loadData("data/OMG/testAudiopp", augment=False)
finalX, finalArquivo = loadData("data/OMG/testFinalAudiopp", augment=False)

trainArousal = []
trainValence = []
trainEmotion = []
testArousal = []
testValence = []
testEmotion = []

csvFileTrain = "data/OMG/omg_TrainVideos.csv"
baseTrain = []
csvFileTest = "data/OMG/omg_ValidationVideos.csv"
baseTest = []
csvFileFinal = "data/OMG/omg_FinalVideos.csv"
baseFinal = []

with open(csvFileTrain, 'rt') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        try:
            temp1 = row[3]
            temp2 = row[4]
            temp3 = row[5]
            temp4 = row[6]
            temp5 = row[7]

            baseTrain.append([temp1, temp2, temp3, temp4, temp5])
        except IndexError:
            pass

with open(csvFileTest, 'rt') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        try:
            temp1 = row[3]
            temp2 = row[4]
            temp3 = row[5]
            temp4 = row[6]
            temp5 = row[7]

            baseTest.append([temp1, temp2, temp3, temp4, temp5])
        except IndexError:
            pass

with open(csvFileFinal, 'rt') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        try:
            temp1 = row[3]
            temp2 = row[4]
            temp3 = 0
            temp4 = 0
            temp5 = 0

            baseFinal.append([temp1, temp2, temp3, temp4, temp5])
        except IndexError:
            pass

for k in trainArquivo:
    for j in baseTrain:
        if ((j[0]==k[:k.find('-')]) and (j[1]==k[k.find('-')+1:len(k)-4])):
            trainArousal.append(float(j[2]))
            trainValence.append(float(j[3]))
            #trainEmotion.append(int(j[4]))

print("Finish collect of Arousal and Valence - Train")

for k in testArquivo:
    for j in baseTest:
        if ((j[0]==k[:k.find('-')]) and (j[1]==k[k.find('-')+1:len(k)-4])):
            testArousal.append(float(j[2]))
            testValence.append(float(j[3]))
            #testEmotion.append(int(j[4]))
print("Finish collect of Arousal and Valence - Test")

#trainEmotion = np_utils.to_categorical(trainEmotion, 7)
#testEmotion = np_utils.to_categorical(testEmotion, 7)

######################### Modelo ###########################################

autoencoder2 = load_model('autoencoder-IEMOCAP.h5')


modelArousal = load_model('hyperas-omg2/modelValence(20iteracoes-1).h5')
modelValence = load_model('hyperas-omg2/modelValence(20iteracoes-1).h5')

modelArousal.get_layer("encoder/L1/Conv1").trainable = False
modelArousal.get_layer("encoder/L1/Conv2").trainable = False
modelArousal.get_layer("encoder/L2/Conv1").trainable = False
modelArousal.get_layer("encoder/L2/Conv2").trainable = False
modelArousal.get_layer("encoder/L3/Conv1").trainable = False
modelArousal.get_layer("encoder/L3/Conv2").trainable = False
modelArousal.get_layer("encoder/L4/Conv1").trainable = False
modelArousal.get_layer("encoder/L4/Conv2").trainable = False

modelValence.get_layer("encoder/L1/Conv1").trainable = False
modelValence.get_layer("encoder/L1/Conv2").trainable = False
modelValence.get_layer("encoder/L2/Conv1").trainable = False
modelValence.get_layer("encoder/L2/Conv2").trainable = False
modelValence.get_layer("encoder/L3/Conv1").trainable = False
modelValence.get_layer("encoder/L3/Conv2").trainable = False
modelValence.get_layer("encoder/L4/Conv1").trainable = False
modelValence.get_layer("encoder/L4/Conv2").trainable = False

optimizer = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

from keras.losses import logcosh
modelArousal.compile(loss="mean_absolute_error",optimizer=Adam(), metrics=['mse', metrics.ccc])
modelValence.compile(loss="mean_absolute_error",optimizer=Adam(), metrics=['mse', metrics.ccc])

#adaGradOptimizer = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0001)
#modelEmotion.compile(optimizer=adaGradOptimizer, loss="categorical_crossentropy",
#    metrics=['accuracy', metrics.fbeta_score, metrics.recall, metrics.precision])


print("Training")
#modelArousal.fit(x=trainX, y=trainArousal, batch_size=128, epochs=100)
#modelValence.fit(x=trainX, y=trainValence, batch_size=128, epochs=100)
#modelEmotion.fit(x=trainX, y=trainEmotion, batch_size=64, epochs=100)

print("Evaluating")
testeA = modelArousal.evaluate(x=testX, y=testArousal, batch_size=64)
testeV = modelValence.evaluate(x=testX, y=testValence, batch_size=64)
#testeE = modelEmotion.evaluate(x=testX, y=testEmotion, batch_size=64)

print("Metrics Arousal Dividido: ", testeA)
print("Metrics Valence Dividido: ", testeV)
#print("Metrics Emotion: ", testeE)

testPredictArousal = modelArousal.predict(testX)
testPredictValence = modelValence.predict(testX)

finalPredictArousal = modelArousal.predict(finalX)
finalPredictValence = modelValence.predict(finalX)

testPredAset = []
testPredVset = []
testRealA = []
testRealV = []

finalPredAset = []
finalPredVset = []
finalVideo = []
finalUtterance = []

for j in baseTest:
    temp = []
    jk = 0
    while(jk<len(testArquivo)):
        if ((j[0] == testArquivo[jk][:testArquivo[jk].find('-')]) and (j[1] == testArquivo[jk][testArquivo[jk].find('-') + 1:len(testArquivo[jk]) - 4])):
            temp.append(testPredictArousal[jk][0])
            #print("entrou A ", testPredictArousal[jk][0])
        jk = jk+1
    testPredAset.append(temp)
    testRealA.append(j[2])

for j in baseTest:
    temp = []
    jk = 0
    while (jk < len(testArquivo)):
        if ((j[0] == testArquivo[jk][:testArquivo[jk].find('-')]) and (
                j[1] == testArquivo[jk][testArquivo[jk].find('-') + 1:len(testArquivo[jk]) - 4])):
            temp.append(testPredictValence[jk][0])
            #print("entrou V ", testPredictValence[jk][0])
        jk = jk + 1
    testPredVset.append(temp)
    testRealV.append(j[3])

for j in baseFinal:
    temp = []
    temp2 = ""
    temp3 = ""
    jk = 0
    while(jk<len(finalArquivo)):
        if ((j[0] == finalArquivo[jk][:finalArquivo[jk].find('-')]) and (j[1] == finalArquivo[jk][finalArquivo[jk].find('-') + 1:len(finalArquivo[jk]) - 4])):
            temp.append(finalPredictArousal[jk][0])
            temp2 = finalArquivo[jk][:finalArquivo[jk].find('-')]
            temp3 = finalArquivo[jk][finalArquivo[jk].find('-') + 1:len(finalArquivo[jk]) - 4]
            #print("entrou A ", testPredictArousal[jk][0])
        jk = jk+1
    finalPredAset.append(temp)
    finalVideo.append(temp2)
    finalUtterance.append(temp3)

contA = 0
contV = 0
for i in finalPredAset:
    contA = contA + len(i)
    if (len(i)==0):
        print("Arousal -", i)

conty = 0
for i in finalPredVset:
    contV = contV + len(i)
    if (len(i)==0):
        print("Valence -", i)
        print(baseFinal[conty][0])
        print(baseFinal[conty][1])
    conty = conty+1

testArousalMediana = []
testValenceMediana = []
testArousalModa = []
testValenceModa = []
testArousalMedia = []
testValenceMedia = []
testArousalReal = []
testValenceReal = []

finalArousalMedia = []
finalValenceMedia = []
finalArousalMediana = []
finalValenceMediana = []

cont = 0
for r in finalPredAset:
    if (len(r)!=0):
        finalArousalMediana.append(statistics.median(r))
        finalArousalMedia.append(numpy.mean(r))
    if (len(r)==0):
        finalArousalMediana.append(0)
        finalArousalMedia.append(0)
    cont = cont +1

cont = 0
for r in finalPredVset:
    if (len(r)!=0):
        finalValenceMediana.append(statistics.median(r))
        finalValenceMedia.append(numpy.mean(r))
    if (len(r)==0):
        finalValenceMediana.append(0)
        finalValenceMedia.append(0)
    cont = cont +1

cont = 0
for r in testPredAset:
    if (len(r)!=0):
        testArousalReal.append(baseTest[cont][2])
        testArousalMediana.append(statistics.median(r))
        data = Counter(r)
        mostCommon = data.most_common(1)
        testArousalModa.append(mostCommon[0][0])
        testArousalMedia.append(numpy.mean(r))
    cont = cont +1

cont = 0
for r in testPredVset:
    if(len(r)!=0):
        testValenceReal.append(baseTest[cont][3])
        testValenceMediana.append(statistics.median(r))
        data = Counter(r)
        mostCommon = data.most_common(1)
        testValenceModa.append(mostCommon[0][0])
        testValenceMedia.append(numpy.mean(r))
    cont = cont+1

path = "resultadosModa-exe09.csv"
with open(path, "wb") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(["link", "start", "end", "video", "utterance", "arousal", "valence"])
    print("moda")
    for i in range(len(testArousalMedia)):
        writer.writerow(["", "", "", "", "", testArousalModa[i], testValenceModa[i]])

path = "resultadosMediana-exe12.csv"
with open(path, "wb") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(["link","start","end","video","utterance","arousal","valence"])
    print("mediana")
    for i in range(len(testArousalMedia)):
        writer.writerow(["", "", "", "", "", testArousalMediana[i], testValenceMediana[i]])


path = "resultadosMedia-exe12.csv"
with open(path, "wb") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(["link", "start", "end", "video", "utterance", "arousal", "valence"])
    print("media")
    for i in range(len(testArousalMedia)):
        writer.writerow(["", "", "", "", "", testArousalMedia[i], testValenceMedia[i]])

path = "real-exe9.csv"
with open(path, "wb") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(["link", "start", "end", "video", "utterance", "arousal", "valence"])
    print("real")
    for i in range(len(testArousalMedia)):
        writer.writerow(["", "", "", "", "", testArousalReal[i], testValenceReal[i]])

print(len(baseFinal))
print(len(finalArousalMediana))
print(len(finalValenceMediana))

finalFile = {'video': [], 'utterance':[], 'arousal': [], 'valence': []}
print(finalArquivo[:3])
finalFile = {'video': [], 'utterance':[], 'arousal': [], 'valence': []}
for j in baseFinal:
    tempA = []
    tempV = []
    jk = 0
    finalFile['video'].append(j[0])
    finalFile['utterance'].append(j[1])

    while(jk<len(finalArquivo)):
        if ((j[0] == finalArquivo[jk][:finalArquivo[jk].find('-')]) and (j[1] == finalArquivo[jk][finalArquivo[jk].find('-') + 1:len(finalArquivo[jk]) - 4])):
            print("entre")
            tempA.append(finalPredictArousal[jk][0])
            tempV.append(finalPredictValence[jk][0])

        jk = jk+1
    finalPredAset.append(tempA)
    if(len(tempA)==0):
        finalFile['arousal'].append(0)
    else:
        finalFile['arousal'].append(numpy.median(tempA))
    if(len(tempV)==0):
        finalFile['valence'].append(0)
    else:
        finalFile['valence'].append(numpy.median(tempV))
    #finalRealA.append(j[2])


dataf = DataFrame.from_dict(finalFile)
dataf.to_csv('resultado.csv')