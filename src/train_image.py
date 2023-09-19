import gc

import torch
from torch import nn
from torch import cuda
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from tqdm import tqdm

from model_visual import *
from settings import *

def train_image_model(args, train_loader, valid_loader, device):
    model = ResNet50(args.image_model.hid_dim, args.opts.n_classes, args.image_model.dropout).to(device)
    model = nn.DataParallel(model)
    
    criterion = nn.BCELoss()
    
    if args.image_model.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), args.image_model.lr)
        
    patience = args.image_model.patience
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience, verbose=True)
        
    # Early stopping parameters
    es_patience = args.image_model.es_patience
    min_valid_loss = 10000000
    min_eta = 0
    n_stop = 0
        
    best_valid_acc = 0
    for epoch in range(args.image_model.epochs):
        model.train()
        
        # Training step
        train_accs = []
        train_losses = []
        for x, _, y in tqdm(train_loader):
            y = y.to(device)
            features, logits = model(x.to(device))

            preds = torch.sigmoid(logits)
            loss = criterion(preds, y)
            train_losses.append(loss.item())
            
            acc = torch.sum(y==torch.round(preds)) / (preds.size()[0]*preds.size()[1])
            train_accs.append(acc.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        mean_train_acc = sum(train_accs) / len(train_accs)
        mean_train_loss = sum(train_losses) / len(train_losses)
        
        print(f'Epoch {epoch} -> Train accuracy: {mean_train_acc}')
        gc.collect()
        cuda.empty_cache()
              
        # Validation step
        valid_accs = []
        valid_losses = []
        
        model.eval()
        with torch.no_grad():
            for x, _, y in tqdm(valid_loader):
                y = y.to(device)
                features, logits = model(x.to(device))

                preds = torch.sigmoid(logits)
                loss = criterion(preds, y)
                valid_losses.append(loss.item())

                acc = torch.sum(y==torch.round(preds)) / (preds.size()[0]*preds.size()[1])
                valid_accs.append(acc.item())
            
        mean_valid_loss = sum(valid_losses) / len(valid_losses)
        mean_valid_acc = sum(valid_accs) / len(valid_accs)
        scheduler.step(mean_valid_loss)
        print(f'Epoch {epoch} -> Validation accuracy: {mean_valid_acc}')
        
        if mean_valid_acc > best_valid_acc:
            best_valid_acc = mean_valid_acc
            
            save_obj = OrderedDict([
                    ('model', model.module.state_dict()),
                    ('mean_train_acc', mean_train_acc),
                    ('mean_train_loss', mean_train_loss),
                    ('mean_valid_acc', mean_valid_acc),
                    ('mean_valid_loss', mean_valid_loss)
                ])
            torch.save(save_obj, TRAINED_MODELS_DIR + f'/image_model.pt')
            print('Saving the model...')
        
        gc.collect()
        cuda.empty_cache()
        
        # Early stopping
        if patience != -1:
            if mean_valid_loss < min_valid_loss:
                min_valid_loss = mean_valid_loss
                min_eta = 0
            elif min_eta >= patience:
                if n_stop >= es_patience:
                    print(f'EARLY STOPPING: Min {min_valid_loss} reached {min_eta} epochs ago')
                    break
                else:
                    n_stop +=1
            else:
                min_eta += 1
    return model

def test_image_model(args, model, test_loader, device):          
    # Validation step
    test_accs = []

    model.eval()
    with torch.no_grad():
        for x, _, y in tqdm(test_loader):
            y = y.to(device)
            
            # Get model's predictions
            features, logits = model(x.to(device))
            preds = torch.sigmoid(logits)

            # Compute accuracy
            acc = torch.sum(y==torch.round(preds)) / (preds.size()[0]*preds.size()[1])
            test_accs.append(acc.item())

    mean_test_acc = sum(test_accs) / len(test_accs)
    
    return mean_test_acc


def get_statistics(args):
    train_losses, train_accs, valid_losses, valid_accs = [], [], [], []
    
    for epoch in range(args.image_model.epochs):
        stats_file = torch.load(TRAINED_MODELS_DIR + f'/image_model_epoch{epoch}.pt')
        train_losses.append(stats_file['mean_train_loss'])
        train_accs.append(stats_file['mean_train_acc'])
        valid_losses.append(stats_file['mean_valid_loss'])
        valid_accs.append(stats_file['mean_valid_acc'])

    return train_losses, train_accs, valid_losses, valid_accs

def plot_training_and_validation_loss(train_loss_values, val_loss_values):
    """
    Plot the training and validation loss over epochs.

    Parameters:
    - train_loss_values: A list of training loss values representing the loss at each epoch.
    - val_loss_values: A list of validation loss values representing the loss at each epoch.
    """

    # Create a range of epochs for the x-axis (assuming one loss value per epoch).
    epochs = range(1, len(train_loss_values) + 1)

    # Create the plot
    plt.figure(figsize=(10, 5))  # Adjust figure size as needed
    plt.plot(epochs, train_loss_values, 'b', label='Training Loss')
    plt.plot(epochs, val_loss_values, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(TRAINED_MODELS_DIR+'/training_losses.png')

    # Show the plot
    plt.show()
    
    
def plot_training_and_validation_accuracy(train_accuracy_values, val_accuracy_values):
    """
    Plot the training and validation accuracy over epochs.

    Parameters:
    - train_accuracy_values: A list of training accuracy values representing accuracy at each epoch.
    - val_accuracy_values: A list of validation accuracy values representing accuracy at each epoch.
    """

    # Create a range of epochs for the x-axis (assuming one value per epoch).
    epochs = range(1, len(train_accuracy_values) + 1)

    # Create the plot
    plt.figure(figsize=(10, 5))  # Adjust figure size as needed
    plt.plot(epochs, train_accuracy_values, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracy_values, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(TRAINED_MODELS_DIR+'/training_accs.png')

    # Show the plot
    plt.show()