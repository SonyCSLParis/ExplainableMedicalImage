# Importing the necessary modules
import os  # Module for interacting with the operating system
import spacy  # Natural language processing library

# Loading the spaCy model for English language
nlp = spacy.load("en_core_web_lg")

# Getting the root directory of the project (parent directory of the current working directory)
ROOT_DIR = os.path.dirname(os.getcwd())

# Printing the root directory
print(ROOT_DIR)

# Creating a path to the source code directory
SRC_DIR = os.path.join(ROOT_DIR + '/ExplainableMedicalImage/src')
print(SRC_DIR)

# Creating paths for various data directories
DATA_DIR = os.path.join(ROOT_DIR + '/ExplainableMedicalImage/data')
MIMIC_DIR = os.path.join(DATA_DIR + '/mimic')

# Creating paths for various output directories
OUTPUT_DIR = os.path.join(ROOT_DIR + '/ExplainableMedicalImage/output')
OUTPUT_DIR_TEXT = os.path.join(OUTPUT_DIR + '/output_text')

# Creating paths for image directories
IMG_DIR = os.path.join(MIMIC_DIR + '/images')
IMG_DIR_TRAIN = os.path.join(IMG_DIR + '/Train')
IMG_DIR_TEST = os.path.join(IMG_DIR + '/Test')
IMG_DIR_VALID = os.path.join(IMG_DIR + '/Valid')

# Creating paths for report directories
REPORTS_DIR = os.path.join(MIMIC_DIR + '/reports')
REPORT_PREPROCESS_DIR = os.path.join(REPORTS_DIR + '/preprocessed')

# Creating a path for trained models directory
TRAINED_MODELS_DIR = os.path.join(ROOT_DIR + '/ExplainableMedicalImage/trained_models')

# Creating a list of directories that need to be created if they don't exist
list_dir = [DATA_DIR, OUTPUT_DIR, OUTPUT_DIR_TEXT, REPORT_PREPROCESS_DIR, TRAINED_MODELS_DIR]

# Looping through the list of directories
for x in list_dir:
    if not os.path.exists(x):
        # Creating the directory if it doesn't exist
        os.makedirs(x)
