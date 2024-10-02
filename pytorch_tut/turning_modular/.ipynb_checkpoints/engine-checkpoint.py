'''functionality for training the model and evaluating it'''

import torch
from tqdm import tqdm

# training step
def train_step(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader,
               loss_fn : torch.nn.Module, optim : torch.optim.Optimizer, 
               device : torch.device):
    '''Trains a model for one epoch by preparing model for training and train the model 
    by calcuating the loss and computing the gradients of loss function

    Args:
        model      : model to be trained
        data_loader: dataloader instance of data
        loss_fn    : lost function to be minimised
        optimizer  : optmizer to minimize the loss function
        device     : cpu (or) cuda
    
    returns:
        Tuple of loss and accuracy
        train_loss    : Average training loss
        train_accuracy: Average training accuracy'''

    # training and test loss initialise
    train_loss, train_acc = 0, 0

    # model into training mode
    model.train()

    for X, y in train_dataloader:

        # data on target device
        X, y = X.to(device), y.to(device)
        
        # 1. forward pass
        y_pred = model(X)

        # 2. calculate loss and accuracy per batch
        loss = loss_fn(y_pred, y)
        train_loss += loss
        
        train_acc += accuracy(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3.optimiser zero grad
        optim.zero_grad()

        # 4.loss backward
        loss.backward()

        # 5. step
        optim.step()

    # average training loss
    train_loss = train_loss.item() / len(train_dataloader)
    train_acc /= len(train_dataloader)

    return train_loss, train_acc

# testing step
def test_step(model : torch.nn.Module,
              test_dataloader : torch.utils.data.DataLoader,
              loss_fn : torch.nn.Module,
              device : torch.device):

    '''Tests model performance for single epoch

    Args:
        model      : model to be trained
        data_loader: dataloader instance of data
        loss_fn    : lost function to be minimised
        device     : cpu (or) cuda
    
    returns:
        Tuple of loss and accuracy
        train_loss    : Average training loss
        train_accuracy: Accuracy of training'''

    # training and test loss initialise
    test_loss, test_acc = 0, 0

    # model in evaluation mode
    model.eval()

    with torch.inference_mode():
        for X, y in test_dataloader:

            # data on target device
            X, y = X.to(device), y.to(device)

            # 1. forward pass
            test_pred = model(X)

            # calculate loss
            test_loss += loss_fn(test_pred, y)

            test_acc += accuracy(y_true=y, y_pred=test_pred.argmax(dim=1)) # to get the labels and get the accuracy
    
        # avg test loss
        test_loss = test_loss.item() / len(test_dataloader)
    
        # average accuracy
        test_acc /= len(test_dataloader)

    return test_loss, test_acc

# training and testing
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn : torch.float32,
          optimizer: torch.optim.Optimizer,
          epochs:int, device:torch.device):
    '''Trains and tests model for given number of epochs
    
    Args:
        model: pytorch model to be trained and tested
        train_data_loader: training dataloader instance of data
        test_data_loader: testing dataloader instance of data
        loss_fn    : lost function to be minimised
        optimizer  : optmizer to minimize the loss function
        epochs     : number of times a data is to be used for training
        device     : cpu (or) cuda

    returns:
        dictionary of training and testing loss and accuracy'''

    
    results = {"train_loss": [], "train_acc": [],
               "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model, train_dataloader=train_dataloader,
                                         loss_fn=loss_fn, optim=optimizer,
                                         device=device)
        test_loss, test_acc = test_step(model=model, test_dataloader=test_dataloader,
                                      loss_fn=loss_fn, device=device)
        
        print(f"Model: {model.__class__.__name__} | train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}%")

        print(f"Model: {model.__class__.__name__} | Test loss: {test_loss:.4f}, test accuracy: {test_acc}% \n")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results

def accuracy(y_true, y_pred):
    '''Compute model accuracy
    Args:
        y_true : Actual label
        y_pred : predicted label'''

    # computes element wise equality
    correct = torch.eq(y_true, y_pred).sum().item()
    
    return (correct/len(y_pred))*100
