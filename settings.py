# Adjust the following parameters to change program behaviour

# directories
clearDirectory = "clear/"
hazyDirectory = "hazy/"
testingDirectory = "test/"
modelDirectory = "models/"
inputDirectory = "input/"
outputDirectory = "output/"

# training values
learningRate = 0.0001
weightDecay = 0.0001
workerNum = 4
epochNum = 10       # train model for x epochs
clipValue = 0.1     # clip normal gradient value
displayPeriod = 60  # display loss every x iterations
savePeriod = 60     # save model every x iterations
batchSize = 8       # size of batches to process
trainingRatio = 0.9 # percentage of data used for training, as opposed to testing

# testing values
progressDisplay = 1  # toggles whether or not training progress images are generated
