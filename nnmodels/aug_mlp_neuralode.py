import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch import tensor
from torchdiffeq import odeint
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

class TimeSeriesDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return self.inputs.shape[0]  # 100 samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]   

class Net(nn.Module):

    def __init__(self, input_dim, output_dim, layers, hidden_activation, out_activation):
        super(Net, self).__init__()
        self.first_layer    = nn.Linear(input_dim, layers[0])
        self.hidden_layers  = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.output_layer   = nn.Linear(layers[-1], output_dim)
        
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation

    def forward(self, input):
        
        x = self.hidden_activation(self.first_layer(input))
        for hidden_layer in self.hidden_layers:
            x = self.hidden_activation(hidden_layer(x))
        
        if self.out_activation == None:
            output = self.output_layer(x)
        else: 
            output = self.out_activation(self.output_layer(x))
        
        return output

class AugNeuralODE(nn.Module):
    def __init__(self, aug_dim, layers, hidden_activation, out_activation, times, training_inputs, training_outputs, test_inputs, test_outputs):
        super(AugNeuralODE, self).__init__()
        self.input_dim  = training_inputs.shape[-1]
        self.output_dim = training_outputs.shape[-1]
        self.aug_dim    = aug_dim
        self.fullinpdim = self.input_dim + self.output_dim + self.aug_dim
        self.layers     = layers
        self.times      = times
        self.ffnn       = Net(self.fullinpdim, self.output_dim + self.aug_dim, layers, hidden_activation, out_activation)
        self.lr         = 0.001
        self.criterion  = nn.MSELoss()
        self.batch_size = training_inputs.shape[0]
        self.training_loss   = []
        self.validation_loss = []
        self.train_init = training_outputs[:,0,:]
        self.training_time = 0
        self.training_inputs_org = training_inputs
        self.training_outputs_org = training_outputs
        self.test_inputs    = test_inputs
        self.test_outputs   = test_outputs
        self.no_params  = sum(p.numel() for p in self.ffnn.parameters() if p.requires_grad)
        print(f"MLP NeuralODE with {len(layers)} layers, {layers} neurons -> {self.no_params} trainable parameters")

    def sortData(self, batch_len, end):
        no_of_time_batches = end//batch_len
        self.training_inputs = self.training_inputs_org[:, :end].reshape((self.training_inputs_org.shape[0]*no_of_time_batches, batch_len, self.training_inputs_org.shape[-1]))
        self.training_outputs = self.training_outputs_org[:, :end].reshape((self.training_outputs_org.shape[0]*no_of_time_batches, batch_len, self.training_outputs_org.shape[-1]))
        self.val_inputs = self.test_inputs[:, :end].reshape((self.test_inputs.shape[0]*no_of_time_batches, batch_len, self.test_inputs.shape[-1]))
        self.val_outputs = self.test_outputs[:, :end].reshape((self.test_outputs.shape[0]*no_of_time_batches, batch_len, self.test_outputs.shape[-1]))
        print(f"Training batches: {self.training_inputs.shape[0]}, Test batches: {self.val_inputs.shape[0]}, time steps: {self.training_inputs.shape[1]}")
        return self.training_inputs.shape[0]

    def train(self, cut, step, epochs, set_optimizer = 'adam', lr=0.001, solver='euler', batch_size=32, plot_loss=True):
        start = time.time()
        times = self.times[:cut:step]
        training_inputs = self.training_inputs[:,:cut:step]; training_outputs = self.training_outputs[:,:cut:step]
        # training_outputs = training_outputs[:,::step]; training_inputs = training_inputs[:,::step]
        dataset    = TimeSeriesDataset(training_inputs, training_outputs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.solver = solver
        print(f" Training started with {self.training_inputs.shape[0]} training batches, and {self.training_inputs.shape[1]} time steps.")
        for epoch in range(epochs):
            self.ffnn.train()
            startepoch = time.time()

            for batch_inputs, batch_outputs in dataloader:    
                def deriv_approx(t, x):
                    batch_size = x.shape[0]
                    idx = int(round((t- times[0]).item() / (times[1] - times[0]).item()))
                    netinput = torch.zeros([batch_size, self.fullinpdim])
                    netinput[:,:self.input_dim]  = batch_inputs[:,idx,:self.input_dim]
                    netinput[:,self.input_dim:] = x[:,:]
                    return self.ffnn(netinput)
                
                if set_optimizer == 'lbfgs':
                    self.optimizer = optim.LBFGS(self.ffnn.parameters(), lr=lr, max_iter=20)
                    def closure():
                        self.optimizer.zero_grad()   # zero the gradient buffers
                        init = torch.zeros((batch_inputs.shape[0], self.output_dim+self.aug_dim), dtype=torch.float64)
                        init[:,:self.output_dim] = batch_outputs[:,0,:]
                        model_out = odeint(deriv_approx, init, times, method=solver).permute(1,0,2)[...,:self.output_dim]
                        loss = self.criterion(model_out, batch_outputs)
                        loss.backward()
                        return loss
                    loss = self.optimizer.step(closure)

                if set_optimizer == 'adam':
                    self.optimizer = optim.Adam(self.ffnn.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-7, weight_decay=0.0, amsgrad=False)
                    self.optimizer.zero_grad()
                    init = torch.zeros((batch_inputs.shape[0], self.output_dim+self.aug_dim), dtype=torch.float64)
                    init[:,:self.output_dim] = batch_outputs[:,0,:]
                    model_out = odeint(deriv_approx, init, times, method=solver).permute(1,0,2)[...,:self.output_dim]
                    loss = self.criterion(model_out, batch_outputs)
                    loss.backward()
                    self.optimizer.step()

            endepoch = time.time()
            self.training_loss += [loss.detach().numpy()]
            
            self.ffnn.eval()
            with torch.no_grad():
                self.testpredict = self.predict(times, self.val_inputs[:,:cut:step], self.val_outputs[:,0,:], numpyarr=False)
                valloss = self.criterion(self.testpredict, self.val_outputs[:,:cut:step])
                self.validation_loss += [valloss.detach().numpy()]

            print(f'Epoch: {epoch + 1}/{epochs}, loss: {np.round(self.training_loss[-1], 6)}, val loss: {np.round(self.validation_loss[-1], 6)}, {np.round(endepoch-startepoch, decimals=3)}s/epoch')

        end = time.time()
        self.training_time += np.round(end-start, decimals=1)
        print(f'Final loss: {np.round(loss.detach().numpy(), 6)}, training duration for {epochs} epochs: {self.training_time} seconds')
        
        if plot_loss == True:
            self.plot_loss()

    def setLearningRate(self, lr):
        old = self.optimizer.param_groups[0]['lr']
        self.optimizer = optim.Adam(self.ffnn.parameters(), lr=lr)
        print(f'Learning rate changed from old_lr = {old} to new_lr = {lr}.')

    def predict(self, times, inps, init, solver='euler', print_time = False, numpyarr=True):
        
        def deriv_approx(t, x):
            batch_size = x.shape[0]
            idx = int(round((t- times[0]).item() / (times[1] - times[0]).item()))
            netinput = torch.empty([batch_size, self.fullinpdim])
            netinput[:,:self.input_dim]  = inps[:,idx,:self.input_dim]
            netinput[:,self.input_dim:] = x[:,:]
            return self.ffnn(netinput)
    
        startpred = time.time()

        init_model = torch.zeros((inps.shape[0], self.output_dim+self.aug_dim), dtype=torch.float64)
        init_model[:,:self.output_dim] = init
        pred = odeint(deriv_approx, init_model, times, method=solver).permute(1,0,2)[...,:self.output_dim].float()
        endpred = time.time()

        if print_time:
            print(f'Prediction time for {times.shape[0]} time steps: {np.round(endpred-startpred, 1)} seconds')
        
        if numpyarr==True:
            return pred.detach().numpy()
        return pred

    def save(self, path):
        torch.save(self.ffnn.state_dict(), path)

    def load(self, path):
        self.ffnn.load_state_dict(torch.load(path, weights_only=True))

    def plot_loss(self):
        plt.figure(100)
        plt.semilogy(self.training_loss, c='blue'); 
        plt.semilogy(self.validation_loss, c='red'); plt.title(f'NeuralODE: {str(self.layers)}, solv.: {self.solver}, train_time: {self.training_time} s')
        plt.xlabel('epochs'); plt.ylabel('MSE'); plt.legend(['training loss', 'validation loss'])
        plt.grid(which='both')
        plt.show()

if __name__ == '__main__':

    from load_data import loadTrainingData, plotResults, scale, unscale
    from loadDTdata import loaddata

    times, inps, outs, inplab, outlab = loaddata()
    outs = outs[:,:,1:2]

    train = [0,2,4,5,6,8,9,11]
    test  = [3,7,10]

    metrics = [inps[train].min((0,1)), inps[train].max((0,1)), 
            outs[train].min((0,1)), outs[train].max((0,1))]

    n_inps, n_outs = scale(metrics, inps, outs)

    net = AugNeuralODE(2, [32,32], F.tanh, None, tensor(times), tensor(inps), tensor(outs), tensor(inps), tensor(outs))

    cut = 500
    batch_size = net.sortData(batch_len = cut, end = 20000)
    print(batch_size)
    
    net.train(cut,1,2)
    
    print(net.ffnn)