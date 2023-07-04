import jsonlines
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex

from settings import *


def preprocess_jsonl(input_file, output_file, img_dir):
    print('Preprocessing...')
    
    infixes = (nlp.Defaults.infixes + [r"(?<=[0-9])[+\-\*^](?=[0-9-])"])
    infix_regex = compile_infix_regex(infixes)
    tokenizer = Tokenizer(nlp.vocab, infix_finditer=infix_regex.finditer)
    nlp.tokenizer = tokenizer

    with jsonlines.open(input_file) as reader, jsonlines.open(output_file, mode='w') as writer:
        for line in reader:
            
            # Preprocess report
            text = line['text']
            doc = nlp(text)

            preprocessed_text = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_stop]

            line['text'] = ' '.join(preprocessed_text)
            
            # Preprocess image path
            img_path = line['img']
            img_list = img_path.split('/')
            line['img'] = img_dir + '/' + img_list[-1]

            writer.write(line)
