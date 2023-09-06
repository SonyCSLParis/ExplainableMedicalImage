import gc
import torch
import torchtext
from torch import nn
from torch import cuda
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict

from settings import *

class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units, n_classes):
        super(TextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, lstm_units, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(lstm_units * 2, n_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(embedded)
        output = self.linear(lstm_out[:,-1,:])
        
        return lstm_out[:,-1,:], output

def train_text_model(args, model, train_loader, valid_loader, vocab_size, device):
    model = nn.DataParallel(model)
    
    criterion = nn.BCELoss()
    
    if args.text_model.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), args.text_model.lr)
        
    patience = args.text_model.patience
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience, verbose=True)
        
    # Early stopping parameters
    es_patience = args.text_model.es_patience
    min_valid_loss = 10000000
    min_eta = 0
    n_stop = 0
        
    best_valid_acc = 0
    for epoch in range(args.text_model.epochs):
        model.train()
        
        # Training step
        train_accs = []
        train_losses = []
        for _, report, y in tqdm(train_loader):
            y = y.to(device)
            _, logits = model(report.to(device).long())

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
            for _, report, y in tqdm(valid_loader):
                y = y.to(device)
                _, logits = model(report.to(device).long())

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
            torch.save(save_obj, TRAINED_MODELS_DIR + f'/text_model.pt')
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
        
def test_text_model(args, model, test_loader, device):   
    # Load pre-trained weights
    model = nn.DataParallel(model)
              
    # Validation step
    test_accs = []

    model.eval()
    with torch.no_grad():
        for _, x, y in tqdm(test_loader):
            y = y.to(device)
            
            # Get model's predictions
            _, logits = model(x.to(device))
            preds = torch.sigmoid(logits)

            # Compute accuracy
            acc = torch.sum(y==torch.round(preds)) / (preds.size()[0]*preds.size()[1])
            test_accs.append(acc.item())

    mean_test_acc = sum(test_accs) / len(test_accs)
    
    return mean_test_acc
    