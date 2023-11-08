import gc  # Garbage collection for memory management
import torch  # PyTorch library
import torchtext  # Text processing library for PyTorch
from torch import nn  # Neural network module in PyTorch
from torch import cuda  # CUDA for GPU support
from torch import optim  # Optimization algorithms in PyTorch
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Learning rate scheduler
import torch.nn.functional as F  # Functional interface to various operations in PyTorch
from tqdm import tqdm  # Progress bar for loops
from collections import OrderedDict  # Ordered dictionary for preserving the order of elements
import matplotlib.pyplot as plt # Matplot for plotting 


# Defining a class for the text generation model
class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units, n_classes):
        super(TextGenerator, self).__init__()

        # Embedding layer to convert integer-encoded words to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM layer for sequential processing of the embedded words
        self.lstm = nn.LSTM(embedding_dim, lstm_units, batch_first=True, bidirectional=True)

        # Fully connected linear layer for final classification
        self.linear = nn.Linear(lstm_units * 2, n_classes)

    def forward(self, x):
        # Embedding layer forward pass
        embedded = self.embedding(x)

        # Flatten LSTM parameters to speed up computation
        self.lstm.flatten_parameters()

        # LSTM layer forward pass
        lstm_out, _ = self.lstm(embedded)

        # Final classification using linear layer
        output = self.linear(lstm_out[:, -1, :])

        return lstm_out[:, -1, :], output


# Function for training the text generation model
def train_text_model(args, model, train_loader, valid_loader, vocab_size, device):
    model = nn.DataParallel(model) # parallise computation on multiple devide
    criterion = nn.BCELoss() # binary cross entropy loss --> binary classification tasks

    if args.text_model.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), args.text_model.lr)

    # Patience parameter for reducing learning rate on plateau --> how many epochs to wait before reducing the learning raye
    patience = args.text_model.patience
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience, verbose=True)

    # Early stopping parameters
    es_patience = args.text_model.es_patience
    min_valid_loss = 10000000
    min_eta = 0
    n_stop = 0

    best_valid_acc = 0
    # Add this line to store training losses
    training_losses = []

    for epoch in range(args.text_model.epochs):
        model.train()

        # Training step
        train_accs = []
        train_losses = []
        for _, report, y in tqdm(train_loader):
            y = y.to(device)
            _, logits = model(report.to(device).long())

            # predictions with sigmoid
            preds = torch.sigmoid(logits)
            # binary cross entropy computed between pred and ground truth
            loss = criterion(preds, y)
            train_losses.append(loss.item())

            # accuracy
            acc = torch.sum(y == torch.round(preds)) / (preds.size()[0] * preds.size()[1])
            train_accs.append(acc.item())

            # standard backpropagation step clears the gradients, computes gradients with respect to the loss, and updates the model's parameters

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # compute mean accuracy and mean training loss
        mean_train_acc = sum(train_accs) / len(train_accs)
        mean_train_loss = sum(train_losses) / len(train_losses)

        print(f'Epoch {epoch} -> Train accuracy: {mean_train_acc}')
        training_losses.append(mean_train_loss)
        
        # create plot 
        plt.figure()
        plt.plot(training_losses, label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('src/training_loss_plot.png')

        # Validation step --> similar to training but used for validation
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

                acc = torch.sum(y == torch.round(preds)) / (preds.size()[0] * preds.size()[1])
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
                    n_stop += 1
            else:
                min_eta += 1


# Function for testing the text generation model
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
            acc = torch.sum(y == torch.round(preds)) / (preds.size()[0] * preds.size()[1])
            test_accs.append(acc.item())

    mean_test_acc = sum(test_accs) / len(test_accs)

    return mean_test_acc
