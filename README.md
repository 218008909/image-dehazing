# image-dehazing
## pytorch implementation of an image dehazing solution

### Instructions
* view `settings.py` to see directories and select mode
* optionally run `trainModel.py` to train, if you have a hazy/clear dataset  
 
     (or make use of provided models)
* run `processImage.py` to dehaze

### Metrics

| Dataset      | SSIM   | PSNR (db) |
|--------------|-------:|---------:|
| SOTS Indoor  | 0.8053 |  16.5947 |
| SOTS Outdoor | 0.9166 |  21.6862 |

### Samples
![Comparisontest2](https://user-images.githubusercontent.com/75892147/138856953-a5f80332-b98b-4c7a-9042-fa57866b0526.jpg)
![Comparisontlgu3zdavtxw978ntnhd](https://user-images.githubusercontent.com/75892147/138856355-1e73444a-20d2-4fa3-8df7-6f68de7a221e.jpg)
![Comparisonhazy-landscape,-fog-157070](https://user-images.githubusercontent.com/75892147/138856600-a5902998-8356-48e3-af7d-0a0760303639.jpg)

### Notes
This project was submitted as Final Computer Engineering Design Project for the University of KwaZulu-Natal in 2021.
