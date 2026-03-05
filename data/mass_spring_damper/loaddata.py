import os
import numpy as np
from matplotlib import pyplot as plt 
from torch import tensor

def scale(metrics, inp, out):
    inp = (inp - metrics[0]) / (metrics[1] - metrics[0])
    out = (out - metrics[2]) / (metrics[3] - metrics[2])
    
    return inp, out
    
def unscale(metrics, out):

    out = out * (metrics[3] - metrics[2]) + metrics[2]
    
    return out

def loaddata():
    BASE_DIR = os.path.dirname(__file__)
    times =   tensor(np.load(os.path.join(BASE_DIR, "times.npy")))
    inps  =   np.load(os.path.join(BASE_DIR, "inps.npy"))
    outs  =   np.load(os.path.join(BASE_DIR, "outs.npy"))

    return times, inps, outs