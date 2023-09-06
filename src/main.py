from omegaconf import OmegaConf

from preprocess import *
from settings import *
from data import *
from combined import *
from model_visual import *
from train_image import *
from text_model import *
from torch.utils.data import DataLoader

args = OmegaConf.load(ROOT_DIR+'/configs/config.yaml')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print(f'Running on device {torch.cuda.current_device()}')

if __name__ == '__main__':
    
    ## Defining datasets and dataloaders
    train_dataset = MIMIC_CXR(REPORT_PREPROCESS_DIR+'/train_output.jsonl', IMG_DIR_TRAIN, train_flag=True)
    word_idx = train_dataset.word_idx
    vocab_size = len(word_idx)
    
    valid_dataset = MIMIC_CXR(REPORT_PREPROCESS_DIR+'/valid_output.jsonl', IMG_DIR_VALID, word_idx=word_idx)
    test_dataset = MIMIC_CXR(REPORT_PREPROCESS_DIR+'/test_output.jsonl', IMG_DIR_TEST, word_idx=word_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.opts.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.opts.batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn)
    '''
    ## Image classification from textual information
    text_model = TextGenerator(vocab_size, args.text_model.embedding_dim, args.text_model.lstm_units, args.opts.n_classes).to(device)
    
    # Training the model from scratch
    if not args.text_model.pretrained:
        train_text_model(args, text_model, train_loader, valid_loader, vocab_size, device)
    # Loading a pre-trained model
    else:
        text_model.load_state_dict(torch.load(TRAINED_MODELS_DIR + '/text_model.pt', map_location=device)['model'])
    
    #test_acc = test_text_model(args, text_model, test_loader, device)
    #print(f'Using trained text model with test accuracy {test_acc}')

    ## Image Model without text
    if not args.image_model.pretrained:
        image_model = train_image_model(args, train_loader, valid_loader, device)

    else:
        image_model = ResNet50(args.image_model.hid_dim, args.opts.n_classes, args.image_model.dropout).to(device)
        file = torch.load(TRAINED_MODELS_DIR + '/image_model.pt', map_location=device)
        image_model.load_state_dict(file['model'])

    #test_acc = test_image_model(args, image_model, test_loader, device)
    #print(f'Using trained image model with test accuracy {test_acc}')
    '''
    ## Combined
    device = torch.device("cpu")
    combined_model = CombinedModel(args, word_idx, vocab_size, device).to(device)
    
    if not args.combined.pretrained:
        combined_model = train_combined_model(args, combined_model, train_loader, valid_loader, vocab_size, device)
    else:
        combined_model.load_state_dict(torch.load(TRAINED_MODELS_DIR + '/combined_model.pt')['model'])
    
    test_combined_model(args, combined_model, test_loader, word_idx, device)