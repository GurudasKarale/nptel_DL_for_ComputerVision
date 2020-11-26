import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline


## Fixing the seed for Reproducibility
np.random.seed(1)
torch.manual_seed(1)

## Define 1D input time series, which spans from t= 1 to t=6.
input_series = np.random.randn(6,1)


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        ### YOUR CODE STARTS HERE
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = n_layers
        self.rnnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)


    def forward(self, x, hidden):

        out, hn = self.rnnn(x,hidden)
        out = self.fc(out[-1, :, :])
        return out,hn



        ### YOUR CODE ENDS HERE

# decide on hyperparameters
input_size=1    ## 1D input
output_size=1   ## 1D output
hidden_dim=32  ## Hidden state feature dimension of RNN
n_layers=1     ## No. of stacked layers in RNN

# instantiate an RNN
rnn = RNN(input_size, output_size, hidden_dim, n_layers)

# MSE loss and Adam optimizer with a learning rate of 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

# train the RNN
def train(rnn, n_steps, print_every):

    # initialize the hidden state
    hidden = None

    for batch_i, step in enumerate(range(n_steps)):
        # defining the training data
        x = input_series[:-1]
        y = input_series[1:]

        # convert data into Tensors
        x_tensor = torch.Tensor(x).unsqueeze(0) # unsqueeze gives a 1, batch_size dimension
        y_tensor = torch.Tensor(y)

        # outputs from the rnn
        prediction, hidden = rnn(x_tensor, hidden)

        ## Representing Memory ##
        # make a new variable for hidden and detach the hidden state from its history
        # this way, we don't backpropagate through the entire history
        hidden = hidden.data

        # calculate the loss
        loss = criterion(prediction, y_tensor)
        # zero gradients
        optimizer.zero_grad()
        # perform backprop and update weights
        loss.backward()
        optimizer.step()

        # display loss and predictions
        if batch_i%print_every == 0:
            print (batch_i)
            print('Loss: ', loss.item())
            print ('Predicted Value: ', prediction.data.numpy().flatten())
            print ('True Value: ', y_tensor.data.numpy().flatten())


    return rnn,prediction[-1]

# train the rnn and monitor results
trained_rnn,final_prediction = train(rnn, n_steps = 75, print_every= 11)
print ('Final predicted value of input time series at t=6: ',final_prediction.item())
