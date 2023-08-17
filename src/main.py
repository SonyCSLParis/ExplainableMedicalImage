from omegaconf import OmegaConf

from preprocess import *
from settings import *
from data import *
from model_visual import *
from train_image import *
from bi_lstm import *

args = OmegaConf.load(ROOT_DIR+'/configs/config.yaml')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print(f'Running on device {torch.cuda.current_device()}')

if __name__ == '__main__':

    ## Text

    trained_report_generator = train_report_generator(train_dataset=REPORT_PREPROCESS_DIR+ '/train_output.jsonl', embedding_dim=200, lstm_units=256,
                                                       epochs=1, batch_size=64)
    model_save_path = TRAINED_MODELS_DIR + '/text_model.pt'
    trained_report_generator.model.save(model_save_path)

    test_texts, test_labels = load_test_dataset(REPORT_PREPROCESS_DIR+ '/test_output.jsonl')

    # test trained
    output_jsonl_file = OUTPUT_DIR_TEXT + '/generated_reports.jsonl'
    generate_reports_and_save(trained_report_generator, test_texts, output_jsonl_file)

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

    image_model = ResNet50(args.image_model.hid_dim, args.data.n_classes, args.image_model.dropout).to(device)
    file = torch.load(TRAINED_MODELS_DIR + '/image_model.pt', map_location=device)
    image_model.load_state_dict(file['model'])

    test_acc = test_image_model(args, test_loader, device)

    print(f'Using trained image model with test accuracy {test_acc}')

    ## Image Model without text

    loaded_text_model = load_text_model(model_save_path)
    test_text_outputs = loaded_text_model.generate_outputs(
        test_texts)  # Adjust according to your text generation method
    test_image_outputs = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image']  # Assuming the data structure has 'image' field
            images = images.to(device)
            image_outputs = image_model(images)
            test_image_outputs.append(image_outputs)

    test_image_outputs = torch.cat(test_image_outputs, dim=0)

    # Now you can use test_text_outputs and test_image_outputs for further analysis or comparison
    
    