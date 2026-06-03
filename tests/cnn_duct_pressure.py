import torch 
from torch import tensor
import torch.nn as nn
import numpy
import matplotlib.pyplot as plt
from pathlib import Path

from nnmodels import FeedForward, MLP2CNN
from data.duct_pressure_distribution.dataloader import getInpsAndOuts

### Settings and Parameters
ROOT_DIR = Path(__file__).parent.parent
data_path = ROOT_DIR / 'data/duct_pressure_distribution/N128'
split = 10
device = 'cuda'
### Load, sort, normalize data
inps, outs = getInpsAndOuts(data_path)

p_mean = outs.mean()
p_std  = outs.std()

n_outs = (outs-p_mean) / p_std

inps_train, outs_train = tensor(inps[:split], dtype=torch.float32), tensor(n_outs[:split], dtype=torch.float32)
inps_test, outs_test = tensor(inps[split:], dtype=torch.float32), tensor(n_outs[split:], dtype=torch.float32)

outs_train = outs_train.reshape((10,1,128,128))
outs_test  = outs_test.reshape((2,1,128,128))
### Model implementation

model = MLP2CNN(
    input_dim      = 19,           # your geometry vector size
    mlp_hidden_dim = 1024,          # hidden dim in MLP
    mlp_num_layers = 3,            # number of hidden layers
    mlp_activation = nn.ReLU(),    # or nn.Tanh(), nn.GELU()
    cnn_channels   = [1, 16, 8, 1],# channel progression, must end with 1
    cnn_kernel_size= 3,
    output_size    = 128,          # spatial resolution
).to(device)

inps_train = inps_train.to(device)
inps_test = inps_test.to(device)
outs_train = outs_train.to(device)
outs_test = outs_test.to(device)
### 
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 2
trainloss = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    optimizer.zero_grad()
    pred = model(inps_train)           # (batch, 1, 128, 128)
    loss = criterion(pred, outs_train)
    loss.backward()
    optimizer.step()
    trainloss += [loss.item()]

    print(f"Epoch {epoch:03d} | Loss: {loss:.6f}")

pred = pred * p_std + p_mean
pred = pred.cpu()

"""
no = 2
fig, axes = plt.subplots(3, figsize=(5, 10))

axes[0].imshow(pred[no,0].detach().numpy())
axes[1].imshow(outs_train[no,0])
axes[2].semilogy(trainloss); axes[2].grid()

plt.tight_layout()
plt.savefig("C:\\Users\\pisk_sa\\Code\\surrogate-modelling\\tests\\results\\cnn_duct\\001.svg")
"""
plt.semilogy(trainloss); plt.grid()
plt.savefig(ROOT_DIR / "tests/results/cnn_duct/dummyloss.svg")

testpred = model(inps_test) * p_std + p_mean
testpred = testpred.cpu()

outs_train = outs_train.cpu()
outs_test = outs_test.cpu()

fig, axes = plt.subplots(4,6, figsize=(15, 15))
c = 0
for i in range(4):
    for j in range(0,6,2):
        if c==10:
            pass
        else:
            axes[i,j].imshow(outs_train[c,0])
            axes[i,j].set_title(f'Training set {c}, True')
            axes[i,j+1].imshow(pred[c,0].detach().numpy())
            axes[i,j+1].set_title(f'Training set {c}, Prediction')
            c += 1

axes[3,2].imshow(outs_test[0,0])
axes[3,2].set_title(f"Test set {0}, True")
axes[3,3].imshow(testpred[0,0].detach().numpy())
axes[3,3].set_title(f"Test set {0}, Prediction")
axes[3,4].imshow(outs_test[1,0])
axes[3,4].set_title(f"Test set 1, True")
axes[3,5].imshow(testpred[1,0].detach().numpy())
axes[3,5].set_title(f"Test set 1, Prediction")
fig.suptitle("True values and predictions of Pressure in y-z-Plane", fontsize=24)
fig.tight_layout()
plt.savefig(ROOT_DIR / "tests/results/cnn_duct/dummyall.svg")

