import torch 
import matplotlib.pyplot as plt 
import supervision
from typing import Dict

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
    
def plot_losses(history : Dict):
    """ Plot Loses for Discrimintor and Generator"""
    plt.plot()
    