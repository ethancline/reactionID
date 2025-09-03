import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time
from tqdm import tqdm 
import os
import csv
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from torch.ao.quantization import fuse_modules

import reactionModel

print("Lets identify some reactions...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

def load_txt_data(file_name):
    decay = []
    data = []
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
        data = np.array([column[h] for h in header if h != "MuonDecay"], dtype=np.float32).T
    return data, decay


trainData, trainDecay = load_txt_data("mc17606_mu_positive_training.txt")

validData, validDecay = load_txt_data("mc17606_mu_positive_validation.txt")

print(trainData.shape, trainDecay.shape, validData.shape, validDecay.shape)

if device.type == "cuda":
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
model = reactionModel.reactionLearner().to(device)
model, epochs, losses, valid_losses, lr = reactionModel.train_model(model, data=torch.tensor(trainData), truth=torch.tensor(trainDecay), num_epochs=50, device=device, learning_rate=5e-3, savePath="model/output.pth", validData=torch.tensor(validData), validDecay=torch.tensor(validDecay))


# if model is None:
#     model = reactionModel.reactionLearner().to(device)
#     model = reactionModel.load_model(model=model, checkpoint_path="/home/ksalamone59/src/Machine_Learning/Left_Right/line_model.pth")  
#     plot_and_test_model_BCE(model=model, losses=None, valid_losses=None, num_epochs=None, device=device, straws=validData, truth=validDecay, reaction=reaction)
# else:
#     # plot_and_test_model_BCE(model=model, losses=losses, valid_losses=valid_losses, num_epochs=epochs, device=device, straws=straws, truth=truth_phi, lr=lr)
#     plot_and_test_model_BCE(model=model, losses=losses, valid_losses=valid_losses, num_epochs=epochs, device=device, straws=validData, truth=validDecay, reaction=reaction)