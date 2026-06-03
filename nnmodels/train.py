import torch
import torch.nn as nn
import time
import torch.optim as optim

def train(
        model,
        inputs,
        outputs,
        epochs,
        criterion,
        optimizer,
        learning_rate):
    
    start = time.time()

    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()
        model_out = model.call(inputs)
        loss = criterion(model_out, outputs)
        loss.backward()
        optimizer.step()