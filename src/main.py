from preprocess_text import *
from settings import *

if __name__ == '__main__':
    preprocess_jsonl( f'{REPORTS_DIR}/Train.jsonl', f'{REPORT_PREPROCESS_DIR}/train_output.jsonl')
