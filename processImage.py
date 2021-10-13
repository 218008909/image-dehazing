# add external dependencies
import numpy
import os
import PIL.Image
import torch
import torch.optim
import torchvision

# add custom libraries
import settings
import networkLayout


def process(img):
	# Load image and fomrat
	hazyPhoto = (torch.from_numpy(numpy.asarray(PIL.Image.open(settings.inputDirectory + "\\" + img))/255.0)).float().permute(2,0,1).cuda().unsqueeze(0)
	# Initialise instance of network
	N = networkLayout.network().cuda()
	# Set network to testing mode
	N.eval()
	# Find models
	models = os.listdir(settings.modelDirectory)
	if settings.progressDisplay:
		# If detailed output mode is enabled, process image with each model
		for model in models:
			N.load_state_dict(torch.load(settings.modelDirectory + model))
			clearPhoto = N(hazyPhoto)
			torchvision.utils.save_image(torch.cat((hazyPhoto, clearPhoto), 0), settings.outputDirectory + img + model + img)
			print(model + img + "saved!")
	else:
		# Otherwise, process image with optimal model
		N.load_state_dict(torch.load(settings.modelDirectory + models[0]))
		clearPhoto = N(hazyPhoto)
		torchvision.utils.save_image(torch.cat((hazyPhoto, clearPhoto), 0), settings.outputDirectory + "Comparison" + img)
		torchvision.utils.save_image(clearPhoto, settings.outputDirectory + models[0] + img)
		print(img + "saved!")


if __name__ == '__main__':
	networkLayout.setup()
	# Retrieve image list
	inputList = os.listdir(settings.inputDirectory)
	for input in inputList:
		# Dehaze each image
		process(input)