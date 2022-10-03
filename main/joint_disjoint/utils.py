from torch.nn import MSELoss
from torch.optim import Adam
import torch
from sklearn.metrics import mean_squared_error
import math
import numpy as np
from main.active_learning.utils import convert2float
import os

def train_joint(model, dataloader, epochs, lr, device):
    '''
    For the joint vs. disjoint experiments, we want to return error at each step.
    We also want to divide error terms into valence and arousal part


    Returns rmse curves, segregated into valence and arousal parts for each epoch and batch
    '''

    size = len(dataloader.dataset)
    print(f"Dataset Size: {size} instances")

    valence_rmse_curve = {}
    arousal_rmse_curve = {}

    # initiate optimizer and loss, set model to training mode
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()
    model.train()                       # sets model into training mode

    for e in range(epochs):
        # initialize results for each epoch
        valence_rmse_curve[f'epoch_{e+1}'] = []
        arousal_rmse_curve[f'epoch_{e+1}'] = []

        for X, _, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = convert2float(y)  # convert to float tensor
            out = model(X)  # forward pass
            pred = convert2float(out.logits)  # extract predictions, convert to Float

            # seperate into valence and arousal loss
            valence_pred = pred.detach().numpy()[:, 0]
            arousal_pred = pred.detach().numpy()[:, 1]

            valence_y = y.detach().numpy()[:, 0]
            arousal_y = y.detach().numpy()[:, 1]

            # cumulate RMSE
            valence_rmse = math.sqrt(mean_squared_error(valence_y, valence_pred))
            arousal_rmse = math.sqrt(mean_squared_error(arousal_y, arousal_pred))

            # append to results
            valence_rmse_curve[f'epoch_{e+1}'].append(valence_rmse)
            arousal_rmse_curve[f'epoch_{e+1}'].append(arousal_rmse)


            # compute (joint) loss and take optimzier step
            loss = loss_fn(pred, y)  # compute loss
            print('Current loss', float(loss.data))
            optimizer.zero_grad()  # clean out previous gradients
            loss.backward()  # backpropagate loss
            optimizer.step()  # update weights


        # write results to files
        v_file_path = os.getcwd() + "\\files\\results\\joint_disjoint_logs\\v_joint_train.txt"
        a_file_path = os.getcwd() + "\\files\\results\\joint_disjoint_logs\\a_joint_train.txt"

        with open(v_file_path, "a+") as v_file:
            v_file.write(f"epoch_{e + 1}" + "|" + f"{valence_rmse_curve[f'epoch_{e+1}']}" + "\n")

        with open(a_file_path, "a+") as a_file:
            a_file.write(f"epoch_{e + 1}" + "|" + f"{arousal_rmse_curve[f'epoch_{e+1}']}" + "\n")

    return valence_rmse_curve, arousal_rmse_curve


def train_disjoint(model, dataloader, epochs, lr, device, dim):
    '''
    Trains a disjoint BERT model on either the valence or arousal dimensions. Returns a dictionary and also writes to file.

    :param model: a BERT model instance with one output layer
    :param dataloader: a torch dataloader instance of either a valence or arousal dataset
    :param epochs int: number of epochs to be trained
    :param lr float: the learning rate to be applied during learning
    :param device str: the device type, which can be cpu or gpu
    :param dim str: the dimension to be trained for, which is either valence or arousal

    :return rmse_curve dict: a dict of all rmse error terms (batch by batch) mapping epoch -> error curve
    '''

    size = len(dataloader.dataset)
    print(f"Dataset Size: {size} instances")

    rmse_curve = {}

    # initiate optimizer and loss, set model to training mode
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()
    model.train()                       # sets model into training mode

    for e in range(epochs):
        # initialize results for each epoch
        rmse_curve[f'epoch_{e+1}'] = []

        for X, _, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = convert2float(y)  # convert to float tensor
            out = model(X)  # forward pass
            pred = convert2float(out.logits.flatten())  # extract predictions, convert to Float

            # compute (disjoint) loss
            loss = loss_fn(pred, y)  # compute loss
            rmse = math.sqrt(float(loss.data))

            # append to results
            rmse_curve[f'epoch_{e + 1}'].append(rmse)

            # take optimizer steps
            optimizer.zero_grad()  # clean out previous gradients
            loss.backward()  # backpropagate loss
            optimizer.step()  # update weights


        # write to file
        if dim == 'valence':
            file_path = os.getcwd() + "\\files\\results\\joint_disjoint_logs\\v_disjoint_train.txt"
            with open(file_path, 'a+') as file:
                file.write(f"epoch_{e + 1}" + "|" + f"{rmse_curve[f'epoch_{e+1}']}" + "\n")

        if dim == 'arousal':
            file_path = os.getcwd() + "\\files\\results\\joint_disjoint_logs\\a_disjoint_train.txt"
            with open(file_path, 'a+') as file:
                file.write(f"epoch_{e + 1}" + "|" + f"{rmse_curve[f'epoch_{e + 1}']}" + "\n")

    return rmse_curve



def test_joint(model, dataloader, device):
    size = len(dataloader.dataset)
    valence_rmse_curve = []
    arousal_rmse_curve = []
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for X, _, y in dataloader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            pred = out.logits

            # seperate valence and arousal errors
            valence_pred, arousal_pred = pred.detach().numpy()[:, 0], pred.detach().numpy()[:, 1]
            valence_y, arousal_y = y.detach().numpy()[:, 0], y.detach().numpy()[:, 1]

            # calculate error metric
            valence_rmse = math.sqrt(mean_squared_error(valence_y, valence_pred))
            arousal_rmse = math.sqrt(mean_squared_error(arousal_y, arousal_pred))

            # append to results
            valence_rmse_curve.append(valence_rmse)
            arousal_rmse_curve.append(arousal_rmse)

    # write results to file
    v_file_path = os.getcwd() + "\\files\\results\\joint_disjoint_logs\\v_joint_test.txt"
    a_file_path = os.getcwd() + "\\files\\results\\joint_disjoint_logs\\a_joint_test.txt"

    with open(v_file_path, "a+") as v_file:
        v_file.write(f"{valence_rmse_curve}")

    with open(a_file_path, "a+") as a_file:
        a_file.write(f"{arousal_rmse_curve}")

    return valence_rmse_curve, arousal_rmse_curve


def test_disjoint(model, dataloader, device, dim):
    '''
    Trains a disjoint BERT model on either the valence or arousal dimensions. Returns a dictionary and also writes to file.

    :param model: a BERT model instance with one output layer
    :param dataloader: a torch test dataloader instance of either a valence or arousal dataset
    :param device str: the device type, which can be cpu or gpu
    :param dim str: the dimension to be trained for, which is either valence or arousal

    :return rmse_curve list: a list representing the RMSE error curve (per batch)
    '''

    size = len(dataloader.dataset)
    rmse_curve = []
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for X, _, y in dataloader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            pred = out.logits.flatten()

            # calculate error metric
            rmse = math.sqrt(mean_squared_error(y.detach().numpy(), pred.detach().numpy()))

            # append to results
            rmse_curve.append(rmse)

    # write to file
    if dim == 'valence':
        file_path = os.getcwd() + "\\files\\results\\joint_disjoint_logs\\v_disjoint_test.txt"
        with open(file_path, 'a+') as file:
            file.write(f"{rmse_curve}")

    if dim == 'arousal':
        file_path = os.getcwd() + "\\files\\results\\joint_disjoint_logs\\a_disjoint_test.txt"
        with open(file_path, 'a+') as file:
            file.write(f"{rmse_curve}")

    return rmse_curve
