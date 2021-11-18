# Python Standard Library
import os
import threading
import time
import tkinter

# Project Dependencies
import networkLayout
import processImage
import settings
import trainModel

###############
# Main Window
###############
GUI = tkinter.Tk(className=" Image Dehazer")
GUI.geometry("256x350")

##################
# Parameter Entry
##################
# Batch Size
lBatchSize = tkinter.Label(text="Batch Size")
eBatchSize = tkinter.Entry()
eBatchSize.insert(0, settings.batchSize)
lBatchSize.pack()
eBatchSize.pack()
# Training Ratio
lTrainingRatio = tkinter.Label(text="TrainingRatio")
eTrainingRatio = tkinter.Entry()
eTrainingRatio.insert(0, settings.trainingRatio)
lTrainingRatio.pack()
eTrainingRatio.pack()
# Clip Value
lClipValue = tkinter.Label(text="Clip Value")
eClipValue = tkinter.Entry()
eClipValue.insert(0, settings.clipValue)
lClipValue.pack()
eClipValue.pack()
# Epoch Number
lEpochNumber = tkinter.Label(text="Epoch Number")
eEpochNumber = tkinter.Entry()
eEpochNumber.insert(0, settings.epochNum)
lEpochNumber.pack()
eEpochNumber.pack()
# Learning Rate
lLearningRate = tkinter.Label(text="Learning Rate")
eLearningRate = tkinter.Entry()
eLearningRate.insert(0, settings.learningRate)
lLearningRate.pack()
eLearningRate.pack()
# Weight Decay
lWeightDecay = tkinter.Label(text="Weight Decay")
eWeightDecay = tkinter.Entry()
eWeightDecay.insert(0, settings.weightDecay)
lWeightDecay.pack()
eWeightDecay.pack()

##########
# Buttons
##########
# Definition
Train = tkinter.Button(text="Train")
Dehaze = tkinter.Button(text="Dehaze")
# Functions
def trainingThread():
    trainModel.train(bS=int(eBatchSize.get()),
                     tR=float(eTrainingRatio.get()),
                     cV=float(eClipValue.get()),
                     eN=int(eEpochNumber.get()),
                     lR=float(eLearningRate.get()),
                     wD=float(eWeightDecay.get()))
    Train['text'] = "Train Complete"
def Trainer(event):
    networkLayout.setup()
    threading.Thread(target=trainingThread).start()
    Train['text'] = "Training..."
def dehazingThread():
    for image in os.listdir(settings.inputDirectory):
        processImage.process(img=image)
    Dehaze['text'] = "Dehaze Complete"
def Dehazer(event):
    networkLayout.setup()
    threading.Thread(target=dehazingThread).start()
    Dehaze['text'] = "Dehazing..."
# Binding and Packing
Train.bind("<Button-1>", Trainer)
Dehaze.bind("<Button-1>", Dehazer)
Train.pack()
Dehaze.pack()

r''' Optional Training Parameters

lHazyInput = tkinter.Label(text="Hazy Input")
eHazyInput = tkinter.Entry()
lClearDirectory = tkinter.Label(text="Clear Directory")
eClearDirectory = tkinter.Entry()
eClearDirectory.insert(0, settings.clearDirectory)
lHazyDirectory = tkinter.Label(text="Hazy Directory")
eHazyDirectory = tkinter.Entry()
eHazyDirectory.insert(0, settings.hazyDirectory)
lHazyInput.pack()
eHazyInput.pack()
lClearDirectory.pack()
eClearDirectory.pack()
lHazyDirectory.pack()
eHazyDirectory.pack()'''

GUI.mainloop()