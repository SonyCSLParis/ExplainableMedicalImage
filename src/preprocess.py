# Import necessary libraries and modules
import jsonlines
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex

from settings import *


# Define a function to preprocess data from a JSONL file
def preprocess_jsonl(input_file, output_file, img_dir):
    print('Preprocessing...')

    # Define custom infixes for tokenization (infixes are patterns for splitting tokens)
    infixes = (nlp.Defaults.infixes + [r"(?<=[0-9])[+\-\*^](?=[0-9-])"])
    infix_regex = compile_infix_regex(infixes)
    tokenizer = Tokenizer(nlp.vocab, infix_finditer=infix_regex.finditer)

    # Set the custom tokenizer
    nlp.tokenizer = tokenizer

    # Open the input JSONL file for reading and the output JSONL file for writing
    with jsonlines.open(input_file) as reader, jsonlines.open(output_file, mode='w') as writer:
        for line in reader:
            # Preprocess report text
            text = line['text']
            doc = nlp(text)

            # Lemmatize, convert to lowercase, and remove punctuation and stopwords
            preprocessed_text = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_stop]

            # Update the line with preprocessed text
            line['text'] = ' '.join(preprocessed_text)

            # Preprocess image path
            img_path = line['img']
            img_list = img_path.split('/')
            line['img'] = img_dir + '/' + img_list[-1]

            # Write the updated line to the output file
            writer.write(line)
