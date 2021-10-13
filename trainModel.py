import networkLayout
import numpy
import os
import PIL.Image
import random
import settings
import torch
import torch.nn
import torch.optim
import torch.utils.data
import torchgeometry.losses
import torchvision


def train():
    # Create network object and set to training mode
    N = networkLayout.network().cuda()
    N.train()
    # Get dataset objects and convert to dataloaders
    datasetTraining, datasetTesting = compilePhotos()
    loaderTraining = torch.utils.data.DataLoader(datasetTraining, batch_size = settings.batchSize, shuffle = True, num_workers = settings.workerNum, pin_memory = True)
    loaderTesting = torch.utils.data.DataLoader(datasetTesting, batch_size = settings.batchSize, shuffle = True, num_workers = settings.workerNum, pin_memory = True)
    # Define model evaluation criterion and optimizer
    criterion = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(N.parameters(), lr = settings.learningRate, weight_decay = settings.weightDecay)
    # Loop through epochs
    count = 0
    for epoch in range(settings.epochNum):
        # Loop through images, training
        print("Epoch: " + str(epoch))
        for (clearImg, hazyImg) in loaderTraining:
            count = count + 1
            clearImg = clearImg.cuda()
            hazyImg = hazyImg.cuda()
            dehazedImg = N(hazyImg)
            loss = criterion(dehazedImg, clearImg)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(N.parameters(), settings.clipValue)
            optimizer.step()
            if ((count % settings.displayPeriod) == 0):
                print("Iteration: ", count, " Loss: ", loss.item())
            if (count % settings.savePeriod) == 0:
                torch.save(N.state_dict(), settings.modelDirectory + "incompleteModel" + str(round((count/settings.savePeriod))) + ".pth")
    # Loop through images, testing
    count = 0
    for (clearImg, hazyImg) in loaderTesting:
        count = count + 1
        clearImg = clearImg.cuda()
        hazyImg = hazyImg.cuda()
        dehazedImg = N(hazyImg)
        torchvision.utils.save_image(torch.cat((hazyImg, dehazedImg, clearImg), 0), settings.testingDirectory+str(count)+".jpg")
    torch.save(N.state_dict(), settings.modelDirectory + "completeModel.pth")


def compilePhotos():
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
        ID = hazyImg.split("_")[0] + "_" + hazyImg.split("_")[1] + ".jpg"
        masterList[ID].append(hazyImg)

    # Sort data into lists
    i = 0
    for key in masterList.keys():
        for img in masterList[key]:
            i = i + 1
            if i < (len(hazyList)*settings.trainingRatio):
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
    networkLayout.setup()
    train()
