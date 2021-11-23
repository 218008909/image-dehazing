# Image Dehazing
# PyTorch Implementation

## Summary
This project makes use of a CNN to reduce the effect of haze in provided images. While there is some aesthetic benefit, the primary purpose is to increase image clarity, allowing other image processing tasks to excel. The functions defined here can be easily-integrated in other workflows, or the program can be run as-is.

## Prerequisites
* Installation of Python 3 (tested on 3.9)
* Run `pip3 install -r requirements.txt` to fetch other dependencies

## Instructions
* View `settings.py` to set up directories 
     for dehazing, place images in input directory
     for training, set up clear and hazy directories for chosen dataset
* Run `gui.py` to train or dehaze (experimental)
     alternatively, run `trainModel.py` or run `processImage.py`
* When employing in other projects, see function definitions for parameters 
     train function accepts several arguments, and defaults to `settings.py`
     dehaze function accepts image path
     both use directories from `settings.py`
     
## Models
* Models are trained separately for indoor and outdoor photographs
* Alternate models are provided for each set
* If progressDisplay is 0 in `settings.py`, only the first model (alphabetically) is used

## Metrics
| Dataset      | SSIM   | PSNR (db) |
|--------------|-------:|---------:|
| SOTS Indoor  | 0.8070 |  18.2020 |
| SOTS Outdoor | 0.9154 |  22.1463 |

## Samples
### Outdoor
![Comparisontest2](https://user-images.githubusercontent.com/75892147/142450344-c467d586-5280-4d38-be0e-7f42d2083952.jpg)
![Comparisontlgu3zdavtxw978ntnhd](https://user-images.githubusercontent.com/75892147/142450363-7d86c2d3-1ae7-4dc7-b73e-e4516b97c5ff.jpg)
![Comparisonhazy-landscape,-fog-157070](https://user-images.githubusercontent.com/75892147/142450400-5a3fbb5d-7626-43e3-9d2a-7b88f24005f5.jpg)
### Indoor
![_AAITS_Comparison20211106_184757](https://user-images.githubusercontent.com/75892147/142450278-43089710-865e-41f7-aaca-07fa56eaace8.jpg)
![_AAITS_Comparison20211106_184809](https://user-images.githubusercontent.com/75892147/142450385-bc2c1845-2a8c-43a6-bc08-870386505018.jpg)

## Other Scripts
* `testModel.py` evaluates SSIM and PSNR over validation dataset
* `trainVaryingParameters.py` trains several models using given parameter sets (could be more thorough if for loops were nested)
* `networkLayout.py` defines the CNN architecture

## Legacy Versions
* SSIM calculation has been removed from the default version of this project
* Instructions for how to use the previous version are in the `SSIM` directory
* The reason for this removal has to do with editing source code for external libraries

## Notes
This project was submitted as Final Computer Engineering Design Project for the University of KwaZulu-Natal in 2021.
