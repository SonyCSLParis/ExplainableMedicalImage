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
    model = ResNet50(args.image_model.hid_dim, args.image_model.n_classes, args.image_model.dropout).to(device)
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
        for x, y in tqdm(train_loader):
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
            for x, y in tqdm(valid_loader):
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
        
def test_image_model(args, test_loader, device):
    # Instantiate the model
    model = ResNet50(args.image_model.hid_dim, args.image_model.n_classes, args.image_model.dropout).to(device)
    
    # Load pre-trained weights
    file = torch.load(TRAINED_MODELS_DIR + '/image_model.pt', map_location=device)
    model.load_state_dict(file['model'])
    model = nn.DataParallel(model)
              
    # Validation step
    test_accs = []

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            y = y.to(device)
            
            # Get model's predictions
            features, logits = model(x.to(device))
            preds = torch.sigmoid(logits)

            # Compute accuracy
            acc = torch.sum(y==torch.round(preds)) / (preds.size()[0]*preds.size()[1])
            test_accs.append(acc.item())

    mean_test_acc = sum(test_accs) / len(test_accs)
    
    return mean_test_acc