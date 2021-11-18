# add custom libraries
import networkLayout
from trainModel import train

if __name__ == "__main__":
    networkLayout.setup()
    # Trains models with varying arguments
    for bS in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        exp = "bS_" + str(bS) + "_"
        train(bS=bS, exp=exp)
    for tR in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        exp = "tR_" + str(tR) + "_"
        train(tR=tR, exp=exp)
    for cV in [0, 0.001, 0.01, 0.1, 0.5, 1]:
        exp = "cV_" + str(cV) + "_"
        train(cV=cV, exp=exp)
    for eN in [10, 11, 12, 13, 14, 15]:
        exp = "eN_" + str(eN) + "_"
        train(eN=eN, exp=exp)
    for lR in [0.00001, 0.0001, 0.001, 0.01]:
        exp = "lR_" + str(lR) + "_"
        train(lR=lR, exp=exp)
    for wD in [0.00001, 0.0001, 0.001, 0.01]:
        exp = "wD_" + str(wD) + "_"
        train(wD=wD, exp=exp)
