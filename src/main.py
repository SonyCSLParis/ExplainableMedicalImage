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