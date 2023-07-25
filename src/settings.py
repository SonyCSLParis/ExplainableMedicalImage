import os

import spacy

nlp = spacy.load("en_core_web_lg")

ROOT_DIR = os.path.dirname(os.getcwd())
SRC_DIR = os.path.join(ROOT_DIR + '/src')
DATA_DIR = os.path.join(ROOT_DIR + '/data')
MIMIC_DIR = os.path.join(DATA_DIR + '/mimic')

IMG_DIR = os.path.join(MIMIC_DIR + '/images')
IMG_DIR_TRAIN = os.path.join(IMG_DIR + '/Train')
IMG_DIR_TEST = os.path.join(IMG_DIR + '/Test')
IMG_DIR_VALID = os.path.join(IMG_DIR + '/Valid')
REPORTS_DIR = os.path.join(MIMIC_DIR + '/reports')
REPORT_PREPROCESS_DIR = os.path.join(REPORTS_DIR + '/preprocessed')
TRAINED_MODELS_DIR = os.path.join(ROOT_DIR + '/trained_models')

list_dir = [DATA_DIR,REPORT_PREPROCESS_DIR,TRAINED_MODELS_DIR]

for x in list_dir:
    if not os.path.exists(x):
        os.makedirs(x)