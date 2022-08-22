
import numpy as np


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


# Monte Carlo Dropout function
def monte_carlo_dropout(dataloader, num_simulations, model, device):
    model_predictions = []  # stores the list of predictions for i=1, ... to I models
    for i in range(num_simulations):
        result = []  # stores the j=1, ..., J instances
        model.eval()  # set model to evaluation mode to freeze weights
        enable_dropout(model)
        for j, (X, _, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            out = model(X)
            pred = out.logits.flatten().item()
            result.append(pred)  # store the result
        # print(f"Result for model {i}: {result}")
        model_predictions.append(result)

    pred_arr = np.array(model_predictions)
    pred_var = np.var(pred_arr, axis=0)
    return np.argsort(np.array(pred_var))[::-1]
