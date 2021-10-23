# Adjust the following parameters to change program behaviour

# directories
clearDirectory = "clearNYU2/"
hazyDirectory = "hazyNYU2/"
testingDirectory = "test/"
modelDirectory = "models/"
logDirectory = "logs/"
inputDirectory = "input/"
outputDirectory = "output/"

# training values
learningRate = 0.0001
weightDecay = 0.0001
workerNum = 4
epochNum = 10           # train model for x epochs
clipValue = 0.1         # clip normal gradient value
batchSize = 2           # size of batches to process
trainingRatio = 0.9     # percentage of data used for training, as opposed to testing

# program behaviour
displayPeriod = 100      # display loss every x iterations
savePeriod = 10000         # save model every x iterations (model is saved at completion regardless)
comparisonMode = 2     # 0: Output Image | 1: Output Comparison | 2: Output Image and Comparison
progressDisplay = 0     # toggles whether or not training progress images are generated when processing

