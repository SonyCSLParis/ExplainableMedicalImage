#imports 
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
from settings import *
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score

#combined model class : architecture of the combined model to combine images and text sources
class CombinedModel(nn.Module):

    # architecture and model components
    def __init__(self, args, word_to_idx, idx_to_word, device):
        super(CombinedModel, self).__init__()

        # load pretrained image model
        self.image_model = ResNet50(args.image_model.hid_dim, args.opts.n_classes, args.image_model.dropout)
        self.image_model.load_state_dict(
            torch.load(TRAINED_MODELS_DIR + '/image_model_512.pt', map_location=device)['model'])
        for param in self.image_model.parameters():
            param.requires_grad = True

        # Define the extractor to compute GradCAM explanations
        self.cam_extractor = GradCAM(self.image_model, self.image_model.model_wo_fc[7])

        # set patch size, feature size,
        self.patch_size = args.image_model.patch_size
        self.feat_size = args.image_model.hid_dim

        # set word-to-index, index-to-word
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word

        # define vocabularies
        vocab_size = len(word_to_idx)

        # define padding
        self.padding = torchvision.transforms.Pad(256, fill=0, padding_mode='constant')

        # define the embedding layer to encode textual inputs
        self.embedding = nn.Embedding(vocab_size, args.text_model.embedding_dim, padding_idx=0)

        # LSTM for report generation
        self.lstm = nn.LSTM(args.text_model.embedding_dim, args.text_model.lstm_units, batch_first=True,
                            bidirectional=True)
        # linear layerfor output
        self.linear = nn.Linear(args.text_model.lstm_units * 2, vocab_size)

        self.device = device

    # eextract and encode relevant patches in the images
    def extract_patch_features(self, images):
        # get batch predictions and processes them to extract relevant features
        self.image_model.eval()
        # intermediate features and unbounded output
        features_image, logits = self.image_model(images.to(self.device))
        # applies sigmoid function to get probabilities and round the values to the nearest int
        preds = torch.round(torch.sigmoid(logits))

        #assign tensor to batch features, create a tensor with zeros and assign a size (num of images in batch and feature size)
        batch_features = torch.zeros((images.shape[0], self.feat_size))

        # for each image in the batch, compute the sum of the features of relevant regions for correctly predicted classes
        for i, image in enumerate(images):
            pred = preds[i]

            # get explanations just for labels predicted as 1 (i.e., when the disease is present)
            features = torch.zeros((1, self.feat_size))
            for j, pred_j in enumerate(pred):
                pred_j = pred_j.item()

                if pred_j == 1.:
                    # Get the model's predictions
                    _, logits = self.image_model(image.unsqueeze(0).to(self.device))

                    # Get masks
                    cam = self.cam_extractor(j, logits.to(self.device))[0]
                    resize = torchvision.transforms.Resize((image.shape[1], image.shape[2]), interpolation=PIL.Image.BICUBIC)
                    mask = resize(cam).squeeze(0)

                    y, x = np.unravel_index(np.argmax(mask.cpu()), mask.shape)

                    # Extract most activated patch_size x patch_size patch in the original image
                    patch = image.cpu().numpy().transpose(1, 2, 0)
                    patch = patch[y:y+self.patch_size, x:x+self.patch_size]
                    patch = patch / np.max(patch)

                    patch = self.padding(torch.tensor(patch.transpose(2,0,1))).unsqueeze(0)

                    # Extract features from j_th patch
                    features_j, _ = self.image_model(patch.to(self.device))

                    # Sum features extracted from single patches
                    features = features + features_j.cpu()

                gc.collect()

            batch_features[i] = features
            self.image_model.train()
        return torch.cat((batch_features, features_image.cpu()), dim=1).to(self.device)  # shape [batch_size, 512]

    def forward(self, images, reports):
        # Get image embeddings = whole image's representation + relevant patches' representation
        image_feats = self.extract_patch_features(images).unsqueeze(1)

        # Encode textual input
        text_feats = self.embedding(reports)

        # Concatenate image and text features
        combined_feats = torch.cat((image_feats, text_feats), dim=1)
        lstm_out, _ = self.lstm(combined_feats)

        # Get probability distribution over the words in the vocabulary
        out = self.linear(lstm_out)

        return out

    def generate_report(self, image, max_length=50, temperature=0.2):
        result_caption = []
        generated_tokens = set()
        image_feats = self.extract_patch_features(image)

        with torch.no_grad():
            states = None
            start_idx = self.word_to_idx['START']
            end_idx = self.word_to_idx['END']

            current_word = torch.tensor([start_idx]).unsqueeze(0).to(self.device)

            # Loop until a token other than 'END' is generated
            while True:
                text_feats = self.embedding(current_word)
                hiddens, states = self.lstm(torch.cat((image_feats.unsqueeze(1), text_feats), dim=1))
                output = self.linear(hiddens.squeeze(0))
                output = F.softmax(output[-1, :] / temperature, dim=0)
                next_word = torch.multinomial(output, 1)

                if next_word.item() != end_idx:  # If the generated token is not 'END', break the loop
                    result_caption.append(next_word.item())
                    generated_tokens.add(next_word.item())  # Add the generated token to the set
                    current_word = torch.cat((current_word, next_word.unsqueeze(0).to(self.device)), dim=1)
                    break

            for _ in range(1, max_length):  # Continue generating the sequence as before
                text_feats = self.embedding(current_word)
                hiddens, states = self.lstm(torch.cat((image_feats.unsqueeze(1), text_feats), dim=1))
                output = self.linear(hiddens.squeeze(0))
                output = F.softmax(output[-1, :] / temperature, dim=0)

                # Filter the output probabilities to avoid already generated tokens
                filtered_output = output.clone()
                for token in generated_tokens:
                    filtered_output[token] = 0.0
                filtered_output /= filtered_output.sum()

                next_word = torch.multinomial(filtered_output, 1)
                result_caption.append(next_word.item())

                if next_word.item() == end_idx:
                    break

                generated_tokens.add(next_word.item())  # Add the generated token to the set
                current_word = torch.cat((current_word, next_word.unsqueeze(0).to(self.device)), dim=1)

        out = [self.idx_to_word[idx] for idx in result_caption]
        return out


# train mdoel
def train_combined_model(args, model, train_loader, valid_loader, vocab_size, device):
    # Define loss and optimizer
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

            # Get generated sequence
            outputs = model(imgs, reports[:, :-1])
            outputs = outputs.permute(0, 2, 1)

            loss = criterion(outputs, reports)

            # if minimum loss is encountered, save the model
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

            # Generate and print reports during training
            for img, report in zip(imgs, reports):
                img = img.unsqueeze(0)  # Add batch dimension
                generated_report = CombinedModel.generate_report(model, img, max_length=100, temperature=0.2)
                generated_report_text = " ".join(generated_report)
                reference_report = [model.idx_to_word[x] for x in report.tolist()]
                reference_report_text = " ".join(reference_report)

                print("Generated Report:", generated_report_text)
                print("Reference Report:", reference_report_text)

            gc.collect()

    return model


def test_combined_model(args, model, test_loader, word_to_idx, idx_to_word, device):
    model.eval()
    load = iter(test_loader)
    num_reports = len(load)

    bleu_scores = []
    meteor_scores = []
    rouge_scores =[]

    for i in range(num_reports):
        print('Report ', i)

        image, report, _ = next(load)
        image = image.to(device)

        # generate a report using the model based on the provided image.
        generated_report = model.generate_report(image, max_length=100)
        print("the generated model is:" + str(generated_report))

        # convert report to a list of words
        report = [idx_to_word[x] for x in report.squeeze().tolist()]

        # compute BLEU
        bleu = corpus_bleu([[report]], [generated_report])
        bleu_scores.append(bleu)
        print('BLEU Score:', bleu)

        # compute METEOR
        meteor = meteor_score([report], generated_report)
        meteor_scores.append(meteor)
        print('METEOR Score:', meteor)

        # compute ROUGE
        rouge = Rouge()
        rouge_score = rouge.get_scores(" ".join(report), " ".join(generated_report))
        rouge_scores.append(rouge_score)
        print('ROUGE Score:', rouge_score)

        print('OUTPUT')
        print(" ".join(generated_report))
        print('REPORT')
        print(" ".join(report))

    avg_bleu = sum(bleu_scores) / num_reports
    avg_meteor = sum(meteor_scores) / num_reports
    avg_rouge = sum(rouge_scores) / num_reports

    print("AVG BLEU SCORE:", avg_bleu)
    print("AVG METEOR SCORE:", avg_meteor)
    print("AVG ROUGE SCORE:", avg_rouge)

