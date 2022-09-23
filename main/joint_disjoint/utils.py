from torch.nn import MSELoss
from torch.optim import Adam
from sklearn.metrics import mean_squared_error
import math
import numpy as np
from main.active_learning.utils import convert2float


def train(model, dataloader, epochs, lr, device):
    '''

    For the joint vs. disjoint experiments, we want to return error at each step.
    We also want to divide error terms into valence and arousal part
    '''


    size = len(dataloader.dataset)
    print(f"Dataset Size: {size} instances")

    cum_rmse = 0  # cumulative RMSE

    # initiate optimizer and loss, set model to training mode
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()
    model.train()                       # sets model into training mode

    for e in range(epochs):
        for X, _, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = convert2float(y)  # convert to float tensor
            out = model(X)  # forward pass
            pred = convert2float(out.logits.flatten())  # extract predictions, convert to Float

            loss = loss_fn(pred, y)  # compute loss
            print('Current loss', float(loss.data))
            optimizer.zero_grad()  # clean out previous gradients
            loss.backward()  # backpropagate loss
            optimizer.step()  # update weights

            # cumulate RMSE
            rmse = math.sqrt(loss.item())
            cum_rmse += rmse

    avg_rmse = (cum_rmse / size) / epochs
    print('Average RMSE: ', avg_rmse)
    return avg_rmse
