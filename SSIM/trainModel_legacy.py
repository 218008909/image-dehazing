# add external dependencies
import math
import numpy
import os
import PIL.Image
import piq
import random
import torch
import torch.nn
import torch.optim
import torch.utils.data
import time
import torchvision

# add custom libraries
import settings
import networkLayout

def train(bS=settings.batchSize, tR=settings.trainingRatio, cV=settings.clipValue, eN=settings.epochNum, lR=settings.learningRate, wD=settings.weightDecay, exp=""):
    print(exp)
    # Create network object and set to training mode
    N = networkLayout.network().cuda()
    N.train()
    # Get dataset objects and convert to dataloaders
    datasetTraining, datasetTesting = compilePhotos(tR=tR)
    loaderTraining = torch.utils.data.DataLoader(datasetTraining, batch_size=bS, shuffle = True, num_workers=settings.workerNum, pin_memory=True)
    loaderTesting = torch.utils.data.DataLoader(datasetTesting, batch_size=bS, shuffle = True, num_workers=settings.workerNum, pin_memory=True)
    # Define model evaluation criterion and optimizer (comment in chosen loss function)
    criterion = torch.nn.MSELoss().cuda()
        # criterion = torch.nn.HuberLoss().cuda()
        # criterion = piq.SSIMLoss().cuda()
    optimizer = torch.optim.Adam(N.parameters(), lr=lR, weight_decay=wD)
    # Loop through epochs
    count = 0
    log = open(settings.logDirectory + exp + time.strftime("%Y.%m.%d_%H.%M") + "_Log.txt", 'w')
    for epoch in range(eN):
        # Loop through images, training
        lineEpoch = "\nEpoch " + str(epoch)
        print(lineEpoch)
        log.write(lineEpoch)
        for (clearImg, hazyImg) in loaderTraining:
            count = count + 1
            clearImg = clearImg.cuda()
            hazyImg = hazyImg.cuda()
            # Dehaze, then adjust based on loss
            dehazedImg = N(hazyImg)
            # loss = customSSIMLoss(dehazedImg, clearImg).cuda()
            loss = criterion(dehazedImg, clearImg).cuda()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(N.parameters(), cV)
            optimizer.step()
            # Display and Save, based on user settings
            if ((count % settings.displayPeriod) == 0):
                lineIteration = "\nIteration: " + str(count) + " Loss: " + str(loss.item()) + " SSIM: " + str(piq.ssim(dehazedImg, clearImg, data_range = 1).item())
                print(lineIteration)
                #log.write(lineIteration)
            if ((count % settings.savePeriod) == 0):
                torch.save(N.state_dict(), settings.modelDirectory + exp + "incompleteModel" + str(round((count/settings.savePeriod))) + ".pth")
        # Loop through images, testing
        countB = 0
        SSIMTotal = 0
        MSETotal = 0
        # Process training set
        for (clearImg, hazyImg) in loaderTesting:
            countB = countB + 1
            clearImg = clearImg.cuda()
            hazyImg = hazyImg.cuda()
            dehazedImg = N(hazyImg)
            r'''    Optionally output testing set after each epoch
            import torchvision
            torchvision.utils.save_image(torch.cat((hazyImg, dehazedImg, clearImg), 0), settings.testingDirectory+str(count) + "SSIM__" + str(piq.ssim(dehazedImg, clearImg, data_range = 1).item()) + ".jpg")
            '''
            SSIMTotal = SSIMTotal + piq.ssim(dehazedImg, clearImg, data_range = 1).item()
            MSETotal = MSETotal + criterion(dehazedImg, clearImg).item()
        lineSSIM = "\nSSIM: " + str(SSIMTotal/countB)
        print(lineSSIM)
        log.write(lineSSIM)
        lineMSE = "\nMSE: " + str(MSETotal/countB)
        log.write(lineMSE)
        print(lineMSE)
        linePSNR = "\nPSNR: " + str(10*math.log10(1/(MSETotal/countB)))
        print(linePSNR)
        log.write(linePSNR)
        print("---------")
        log.write("\n---------")
    # Save final model and output to log file
    torch.save(N.state_dict(), settings.modelDirectory + exp + "completeModel.pth")
    log.close()


def compilePhotos(tR):
    # Create lists
    listTraining = []
    listTesting = []
    masterList = {}

    # Retrieve training and testing data
    clearList = os.listdir(settings.clearDirectory)
    hazyList = os.listdir(settings.hazyDirectory)
    for clearImg in clearList:
        masterList[clearImg] = []
    for hazyImg in hazyList:
        ID = hazyImg.split("_")[0] + "_" + hazyImg.split("_")[1] + settings.fileExtension
        masterList[ID].append(hazyImg)

    # Sort data into lists
    i = 0
    for key in masterList.keys():
        for img in masterList[key]:
            i = i + 1
            if i < (len(hazyList)*tR):
                listTraining.append([key, img])
            else:
                listTesting.append([key, img])

    # Shuffle lists
    random.shuffle(listTraining)
    random.shuffle(listTesting)

    # Create dataset objects and return them
    datasetTraining = datasetPhotos(listTraining)
    datasetTesting = datasetPhotos(listTesting)
    return datasetTraining, datasetTesting


class datasetPhotos(torch.utils.data.Dataset):
    # Initialiser
    def __init__(self, pL):
        self.photoList = pL

    # Length Retrieval Function
    def __len__(self):
        return len(self.photoList)

    # Item Retrieval Function
    def __getitem__(self, item):
        clear, hazy = self.photoList[item]
        clearImg = (torch.from_numpy((numpy.asarray(PIL.Image.open(settings.clearDirectory + "\\" + clear).resize((480, 640), PIL.Image.ANTIALIAS))/255.0))).float().permute(2, 0, 1)
        hazyImg = (torch.from_numpy((numpy.asarray(PIL.Image.open(settings.hazyDirectory + "\\" + hazy).resize((480, 640),PIL.Image.ANTIALIAS)) / 255.0))).float().permute(2, 0, 1)
        return clearImg, hazyImg


if __name__ == "__main__":
    # Set up directories and train with parameters from settings.py
    networkLayout.setup()
    train()
