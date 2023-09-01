from omegaconf import OmegaConf

from preprocess import *
from settings import *
from data import *
from model_visual import *
from train_image import *
from torch_bi_lstm import *
from torch.utils.data import DataLoader

args = OmegaConf.load(ROOT_DIR+'/configs/config.yaml')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print(f'Running on device {torch.cuda.current_device()}')

if __name__ == '__main__':

    ## Text
    model_save_path = TRAINED_MODELS_DIR + '/text_model_torch.pt'
    
    if not args.text_model.pretrained:
        trained_report_generator = train_report_generator(train_dataset=REPORT_PREPROCESS_DIR+ '/train_output.jsonl', embedding_dim=200, lstm_units=256,
                                                           epochs=1, batch_size=args.opts.batch_size)
        #trained_report_generator.model.save(model_save_path)
        torch.save(trained_report_generator.model.state_dict(), model_save_path) ### CHANGED
        
    else:
        trained_report_generator = ReportGenerator(REPORT_PREPROCESS_DIR+ '/train_output.jsonl')
        _,_ = trained_report_generator.load_dataset()
        trained_report_generator.build_model(embedding_dim=200, lstm_units=256)
        trained_report_generator.model.load_state_dict(torch.load(model_save_path))
        
    # test trained
    generate_reports_and_save(trained_report_generator, REPORT_PREPROCESS_DIR+ '/test_output.jsonl', TRAINED_MODELS_DIR + '/generated_reports.json')

    ## Image Model without text

    train_dataset = MIMIC_CXR(REPORT_PREPROCESS_DIR+'/train_output.jsonl', IMG_DIR_TRAIN, train_flag=True)
    valid_dataset = MIMIC_CXR(REPORT_PREPROCESS_DIR+'/valid_output.jsonl', IMG_DIR_VALID)
    test_dataset = MIMIC_CXR(REPORT_PREPROCESS_DIR+'/test_output.jsonl', IMG_DIR_TEST)

    train_loader = DataLoader(train_dataset, batch_size=args.opts.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.opts.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    if not args.image_model.pretrained:
        train_image_model(args, train_loader, valid_loader, device)
        test_image_model(args, test_loader, device)

    image_model = ResNet50(args.image_model.hid_dim, args.image_model.n_classes, args.image_model.dropout).to(device)
    file = torch.load(TRAINED_MODELS_DIR + '/image_model.pt', map_location=device)
    image_model.load_state_dict(file['model'])

    test_acc = test_image_model(args, test_loader, device)
    print(f'Using trained image model with test accuracy {test_acc}')

    ## Combined


