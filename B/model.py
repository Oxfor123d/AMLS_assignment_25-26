import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # TODO: 
        pass

    def forward(self, x):
        # TODO: 
        return x

def run_model_B(train_images, train_labels, val_images, val_labels, test_images, test_labels):
    print("Initialising Model B (Convolutional Neural Network)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
       
    
    return {"accuracy": 0.0}