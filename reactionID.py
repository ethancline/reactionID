import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
from tqdm import tqdm 
import os
import csv
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from torch.ao.quantization import fuse_modules
import matplotlib.pyplot as plt
import reactionModel
import reactionPlots
if __name__ == "__main__":
    print("Lets identify some reactions...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on {device}")
    model = None
    def load_txt_data(file_name):
        decay = []
        data = []
        loc = []
        print(f"Loading text data from {file_name}")
        with open(file_name, 'r') as f:
            csvReader = csv.reader(f)
            header = next(csvReader)
            column = {}
            for h in header:
                column[h] = []
            for row in csvReader:
                for h, v in zip(header, row):
                    if h == "MuonDecay":
                        decay.append(int(v))
                    else:
                        column[h].append(v)

            decay = np.array(decay, dtype=np.float32)
            data = np.array([column[h] for h in header if (h != "MuonDecay" and h!="MuonDecay_X" and h!="MuonDecay_Y" and h!="MuonDecay_Z")], dtype=np.float32).T
            loc = np.array([column[h] for h in header if (h == "MuonDecay_X" or h == "MuonDecay_Y" or h == "MuonDecay_Z")], dtype=np.float32).T
        return data, decay, loc



    trainData, trainDecay, trainLoc = load_txt_data("sim_training_decay.txt")
    validData, validDecay, validLoc = load_txt_data("sim_validation_decay.txt")

    print(trainData.shape, trainDecay.shape, validData.shape, validDecay.shape, validLoc.shape)
    # h = plt.hist(trainDecay.flatten(), bins=2,histtype='step')
    # plt.show()

    if device.type == "mps":
        print(f"Using MPS")
    model = reactionModel.reactionLearner().to(device)
    model, epochs, losses, valid_losses, lr = reactionModel.train_model(model, data=torch.tensor(trainData), truth=torch.tensor(trainDecay), num_epochs=150, device=device, learning_rate=5e-3, savePath="model/output.pth", validData=torch.tensor(validData), validDecay=torch.tensor(validDecay))


    if model is None:
        model = reactionModel.reactionLearner().to(device)
        model, lr = reactionModel.load_model(model=model, checkpoint_path="model/output.pth")  
        reactionPlots.plot_and_test_model_BCE(model=model, losses=None, valid_losses=None, num_epochs=None, device=device, straws=validData, truth=validDecay, lr=lr,trainLoc=trainLoc,validLoc=validLoc)
    else:
        # plot_and_test_model_BCE(model=model, losses=losses, valid_losses=valid_losses, num_epochs=epochs, device=device, straws=straws, truth=truth_phi, lr=lr)
        reactionPlots.plot_and_test_model_BCE(model=model, losses=losses, valid_losses=valid_losses, num_epochs=epochs, device=device, straws=validData, truth=validDecay, lr=lr,trainLoc=trainLoc,validLoc=validLoc)