import torch
from torch.utils.data import DataLoader, Dataset
from model_visual import *
from text_model import *
from torchtext.data.metrics import bleu_score

import gc
import PIL
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image


class CombinedModel(nn.Module):
    def __init__(self, args, word_to_idx, idx_to_word, device):
        super(CombinedModel, self).__init__()
        self.image_model = ResNet50(args.image_model.hid_dim, args.opts.n_classes, args.image_model.dropout)
        self.image_model.load_state_dict(
            torch.load(TRAINED_MODELS_DIR + '/image_model_512.pt', map_location=device)['model'])
        for param in self.image_model.parameters():
            param.requires_grad = True

        self.cam_extractor = GradCAM(self.image_model, self.image_model.model_wo_fc[7])

        self.patch_size = args.image_model.patch_size
        self.feat_size = args.image_model.hid_dim

        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        vocab_size = len(word_to_idx)

        self.padding = torchvision.transforms.Pad(256, fill=0, padding_mode='constant')
        '''
        self.text_model = TextGenerator(vocab_size, args.text_model.embedding_dim, args.text_model.lstm_units, args.opts.n_classes)
        self.text_model.load_state_dict(torch.load(TRAINED_MODELS_DIR + '/text_model.pt', map_location=device)['model'])
        self.text_model.linear = nn.Identity()

        self.linear = nn.Linear(args.text_model.lstm_units*2 + args.image_model.hid_dim*2, vocab_size)
        '''
        self.embedding = nn.Embedding(vocab_size, args.text_model.embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(args.text_model.embedding_dim, args.text_model.lstm_units, batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(args.text_model.lstm_units * 2, vocab_size)

        self.device = device

    def extract_patch_features(self, images):
        # Get batch predictions
        self.image_model.eval()
        features_image, logits = self.image_model(images.to(self.device))
        preds = torch.round(torch.sigmoid(logits))

        batch_features = torch.zeros((images.shape[0], self.feat_size))

        # For each image in the batch, compute the sum of the features of relevant regions for correctly predicted classes
        for i, image in enumerate(images):
            pred = preds[i]

            # Get explanations just for labels predicted as 1
            features = torch.zeros((1, self.feat_size))
            for j, pred_j in enumerate(pred):
                pred_j = pred_j.item()

                if pred_j == 1.:
                    # Get the model's predictions
                    _, logits = self.image_model(image.unsqueeze(0).to(self.device))

                    # Get masks
                    cam = self.cam_extractor(j, logits.to(self.device))[0]
                    resize = torchvision.transforms.Resize((image.shape[1], image.shape[2]),
                                                           interpolation=PIL.Image.BICUBIC)
                    mask = resize(cam).squeeze(0)

                    y, x = np.unravel_index(np.argmax(mask.cpu()), mask.shape)

                    # Extract most activated patch_size x patch_size patch in the original image
                    patch = image.cpu().numpy().transpose(1, 2, 0)
                    patch = patch[y:y + self.patch_size, x:x + self.patch_size]
                    patch = patch / np.max(patch)

                    patch = self.padding(torch.tensor(patch.transpose(2, 0, 1))).unsqueeze(0)

                    # Extract features from j_th patch
                    features_j, _ = self.image_model(patch.to(self.device))

                    # Sum features extracted from single patches
                    features = features + features_j.cpu()

                gc.collect()

            batch_features[i] = features
            self.image_model.train()
        return torch.cat((batch_features, features_image.cpu()), dim=1).to(self.device)  # shape [batch_size, 2048]

    def forward(self, images, reports):
        image_feats = self.extract_patch_features(images).unsqueeze(1)
        text_feats = self.embedding(reports)

        combined_feats = torch.cat((image_feats, text_feats), dim=1)
        lstm_out, _ = self.lstm(combined_feats)
        out = self.linear(lstm_out)

        return out

    def generate_report(self, image, max_length=50):
        result_caption = []

        image_feats = self.extract_patch_features(image)

        with torch.no_grad():
            states = None

            start_idx = self.word_to_idx['START']
            end_idx = self.word_to_idx['END']

            current_word = torch.tensor([start_idx]).unsqueeze(0).to(self.device)

            for _ in range(max_length):
                text_feats = self.embedding(current_word)
                hiddens, states = self.lstm(torch.cat((image_feats.unsqueeze(1), text_feats), dim=1))
                output = self.linear(hiddens.squeeze(0))
                output = F.softmax(output[-1, :], dim=0)
                next_word = torch.argmax(output, dim=0)
                result_caption.append(next_word.item())

                if next_word.item() == end_idx:
                    break

                current_word = torch.cat((current_word, torch.tensor([next_word]).unsqueeze(0).to(self.device)), dim=1)

        out = [self.idx_to_word[idx] for idx in result_caption]
        return out


def train_combined_model(args, model, train_loader, valid_loader, vocab_size, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.combined.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Monitoring loss
    min_loss = 1000000
    eta = 0

    print(f'Size of the vocabulary: {vocab_size}')

    for epoch in range(args.combined.epochs):
        print(f'Epoch {epoch}')
        for imgs, reports, _ in tqdm(train_loader):
            imgs = imgs.to(device)
            reports = reports.to(device)

            outputs = model(imgs, reports[:, :-1])
            outputs = outputs.permute(0, 2, 1)

            loss = criterion(outputs, reports)

            if loss < min_loss:
                eta = 0
                min_loss = loss

                # Save the weights
                save_obj = OrderedDict([
                    ('model', model.state_dict())
                ])
                torch.save(save_obj, TRAINED_MODELS_DIR + f'/combined_model.pt')
                print('Saving the model...')
            else:
                eta += 1

            print(f'Loss {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            gc.collect()

    return model


def test_combined_model(args, model, test_loader, word_to_idx, idx_to_word, device):
    model.eval()
    load = iter(test_loader)
    num_reports = 1

    tot_score = 0

    for i in range(num_reports):
        print('Report ', i)

        image, report, _ = next(load)
        image = image.to(device)

        generated_report = model.generate_report(image, max_length=100)
        report = [idx_to_word[x] for x in report.squeeze().tolist()]

        if len(generated_report) < len(report):
            tot_score += bleu_score(generated_report, report[:len(generated_report)])

        else:
            tot_score += bleu_score(generated_report[:len(report)], report)

        print('OUTPUT')
        print(" ".join(generated_report))
        print('REPORT')
        print(" ".join(report))

    print("AVG BLEU SCORE:", tot_score / num_reports)