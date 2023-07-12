from omegaconf import OmegaConf

from preprocess import *
from settings import *
from data import *

args = OmegaConf.load(ROOT_DIR+'/configs/config.yaml')

if __name__ == '__main__':
    if args.opts.preprocess:
        preprocess_jsonl( f'{REPORTS_DIR}/Train.jsonl', f'{REPORT_PREPROCESS_DIR}/train_output.jsonl', IMG_DIR_TRAIN)
        preprocess_jsonl( f'{REPORTS_DIR}/Train.jsonl', f'{REPORT_PREPROCESS_DIR}/train_output.jsonl', IMG_DIR_VALID)
        preprocess_jsonl( f'{REPORTS_DIR}/Train.jsonl', f'{REPORT_PREPROCESS_DIR}/train_output.jsonl', IMG_DIR_TEST)
        
    dataset = MIMIC_CXR(REPORT_PREPROCESS_DIR+'/train_output.jsonl', IMG_DIR_TRAIN)

    # bi-lstm training and generation
    report_generator = ReportGenerator(REPORT_PREPROCESS_DIR + "train_output.jsonl")

    X, y = report_generator.load_dataset()

    embedding_dim = 100
    lstm_units = 64
    report_generator.build_model(embedding_dim, lstm_units)

    epochs = 1
    batch_size = 32
    report_generator.train_model(X, y, epochs, batch_size)

    text = "A tip of a left Port-A-Cath lies in the low superior vena cava."
    report = report_generator.generate_report(text)
    print(report)
