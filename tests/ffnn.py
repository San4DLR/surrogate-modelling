from nnmodels import FeedForward
from torch import tensor
import numpy as np

device = "cpu"

model = FeedForward(input_dim=4, 
                    output_dim=2, 
                    hidden_layers=[16,16],
                    name="ffnn_test",).to(device)

a = tensor(np.ones((100000,4), dtype=np.float32)).to(device)

print(model(a, grad=False))