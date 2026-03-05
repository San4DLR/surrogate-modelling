import numpy as np
from matplotlib import pyplot as plt
import os 

def loadTrainingData():
    datafull = []
    path = 'training_data/'
    all_files = os.listdir(path)
    for i in range(len(all_files)):
        datafull += [np.loadtxt(open(path+all_files[i]), delimiter=',', skiprows=1)]
    
    time = np.array(datafull)[0,:,0]
    datainput = np.array(datafull, dtype='float32')[:,:,1:5]
    dataoutput = np.array(datafull, dtype='float32')[:,:,5:]
    
    #can be deleted
    #datainput = np.dstack((datainput[:,:,:2], datainput[:,:,4:12]))
    
    inputlabel = [np.loadtxt(open(path+all_files[i]), dtype=np.str_ ,delimiter=',',max_rows=1)][0][1:5]

    outputlabel = [np.loadtxt(open(path+all_files[i]), dtype=np.str_ ,delimiter=',',max_rows=1)][0][5:]
    
    return time, datainput, dataoutput, inputlabel, outputlabel

# time, datainput, dataoutput, inputlabel, outputlabel = loadTrainingData()

def plotResults(idx, title, time, inps, outs, preds, inplabel, outlabel, plot_til, labpos=(1.15,1), fsize=(10,9)):
    fig, axs = plt.subplots(4, 1, sharex=True, sharey=False, figsize=fsize)
    axs[0].plot(time[:plot_til],  outs[idx,:plot_til,0], c='tab:blue', label='true')
    #if preds != None:
    axs[0].plot(time[:plot_til], preds[idx,:plot_til,0], c='tab:orange', label='prediction')
    axs[0].set_xlabel('t in s'); axs[0].set_title(f'{outlabel[0]}')
    axs[0].grid(which='major')
    
    if outs.shape[-1] > 1:
        axs[1].plot(time[:plot_til],  outs[idx,:plot_til,1], c='tab:blue', label='true')
        #if preds != None:
        axs[1].plot(time[:plot_til], preds[idx,:plot_til,1], c='tab:orange', label='prediction')
        axs[1].set_xlabel('t in s'); axs[1].set_title(f'{outlabel[1]}')
        axs[1].grid(which='major')
    
    axs[2].plot(time[:plot_til],  inps[idx,:plot_til,1], c='red', label=inplabel[1])
    axs[2].plot(time[:plot_til],  inps[idx,:plot_til,2], c='green', label=inplabel[2])
    axs[2].set_xlabel('t in s'); axs[2].set_title('engine and brake torque in Nm')
    axs[2].grid(which='major')
    
    axs[3].plot(time[:plot_til],  inps[idx,:plot_til,0], c='black', label=inplabel[0])
    axs[3].plot(time[:plot_til],  inps[idx,:plot_til,3], c='cyan', label=inplabel[3])
    axs[3].set_xlabel('t in s'); axs[3].set_title('gradient and gear')
    axs[3].grid(which='major')

    fig.tight_layout()
    for ax in axs.flat:
        ax.set(xlabel='t in s')
    fig.legend(bbox_to_anchor=labpos)
    fig.suptitle(title, fontsize=15, y=1.1)

def scale(metrics, inp, out):
    inp = (inp - metrics[0]) / (metrics[1] - metrics[0])
    out = (out - metrics[2]) / (metrics[3] - metrics[2])
    
    return inp, out
    
def unscale(metrics, out):

    out = out * (metrics[3] - metrics[2]) + metrics[2]
    
    return out
    
    
    
    
    
    
    
    

# plotResults(0, 'test', time, datainput, dataoutput, dataoutput * 1.1, inputlabel, outputlabel, labpos=(1.15,1.))
