import os
import torch
import torch.nn
import settings

class network(torch.nn.Module):
	def __init__(self):
		super(network, self).__init__()
		# Declare activation function and layers
		self.activationFunction = torch.nn.ReLU(inplace=True)
		self.layer1 = torch.nn.Conv2d(3,3,1,1,0,bias=True)
		self.layer2 = torch.nn.Conv2d(3,3,3,1,1,bias=True)
		self.layer3 = torch.nn.Conv2d(6,3,5,1,2, bias=True)

	def forward(self, input):
		# Process input according to architectural design
		stage1 = self.activationFunction(self.layer1(input))
		stage2 = self.activationFunction(self.layer2(stage1))
		stage3 = self.activationFunction(self.layer3(torch.cat((stage1, stage2), 1)))
		output = self.activationFunction(input*stage3)
		return output


def setup():
	# Generate directories
	if not (os.path.exists(settings.inputDirectory)):
		os.mkdir(settings.inputDirectory)  # input
	if not (os.path.exists(settings.outputDirectory)):
		os.mkdir(settings.outputDirectory)  # output
	if not (os.path.exists(settings.modelDirectory)):
		os.mkdir(settings.modelDirectory)  # model
	if not (os.path.exists(settings.testingDirectory)):
		os.mkdir(settings.testingDirectory)  # testing
	if not (os.path.exists(settings.clearDirectory)):
		os.mkdir(settings.clearDirectory)  # clear
	if not (os.path.exists(settings.hazyDirectory)):
		os.mkdir(settings.hazyDirectory)  # hazy