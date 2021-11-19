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

# add custom libraries
import settings
import networkLayout


def test():
    # Create network object and set to training mode
    N = networkLayout.network().cuda()
    N.eval()
    # Get dataset objects and convert to dataloaders
    datasetTraining, datasetTesting = compilePhotos()
    # loaderTraining = torch.utils.data.DataLoader(datasetTraining, batch_size = settings.batchSize, shuffle = True, num_workers = settings.workerNum, pin_memory = True)
    loaderTesting = torch.utils.data.DataLoader(datasetTesting, batch_size = settings.batchSize, shuffle = True, num_workers = settings.workerNum, pin_memory = True)
    # Define model evaluation criterion and optimizer (comment in chosen loss function)
    criterion = torch.nn.MSELoss().cuda()
    # criterion = piq.SSIMLoss().cuda()
    # Loop through epochs
    count = 0
    models = os.listdir(settings.modelDirectory)
    for model in models:
        title = model + time.strftime("%Y.%m.%d_%H.%M") + "_Log___Test.txt"
        print("\n", title)
        log = open(settings.logDirectory + title, 'w')
        count = 0
        SSIMTotal = 0
        MSETotal = 0
        # Process training set
        N.load_state_dict(torch.load(settings.modelDirectory + model))
        for (clearImg, hazyImg) in loaderTesting:
            count = count + 1
            clearImg = clearImg.cuda()
            hazyImg = hazyImg.cuda()
            dehazedImg = N(hazyImg)
            r'''    Optionally output testing set after each epoch
            import torchvision
            torchvision.utils.save_image(torch.cat((hazyImg, dehazedImg, clearImg), 0), settings.testingDirectory+str(count) + "SSIM__" + str(piq.ssim(dehazedImg, clearImg, data_range = 1).item()) + ".jpg")
            '''
            SSIMTotal = SSIMTotal + piq.ssim(dehazedImg, clearImg, data_range = 1).item()
            MSETotal = MSETotal + criterion(dehazedImg, clearImg).item()
        # Save final model and output to log file
        # torch.save(N.state_dict(), settings.modelDirectory + "completeModel.pth")
            # Optionally indent from here
        lineSSIM = "\nSSIM: " + str(SSIMTotal/count)
        print(lineSSIM)
        log.write(lineSSIM)
        lineMSE = "\nMSE: " + str(MSETotal/count)
        log.write(lineMSE)
        print(lineMSE)
        linePSNR = "\nPSNR: " + str(10*math.log10(1/(MSETotal/count)))
        print(linePSNR)
        log.write(linePSNR)
        print("---------")
        log.write("\n---------")
            # to here to obtain more detailed results
        log.close()


def compilePhotos():
    # Create lists
    listTraining = []
    listTesting = []
    masterList = {}

    # Retrieve training and testing data
    clearList = os.listdir(settings.clearTestDirectory)
    hazyList = os.listdir(settings.hazyTestDirectory)
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
            if i < 0:
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
        clearImg = (torch.from_numpy((numpy.asarray(PIL.Image.open(settings.clearTestDirectory + "\\" + clear).resize((480, 640), PIL.Image.ANTIALIAS))/255.0))).float().permute(2, 0, 1)
        hazyImg = (torch.from_numpy((numpy.asarray(PIL.Image.open(settings.hazyTestDirectory + "\\" + hazy).resize((480, 640),PIL.Image.ANTIALIAS)) / 255.0))).float().permute(2, 0, 1)
        return clearImg, hazyImg


if __name__ == "__main__":
    networkLayout.setup()
    test()
