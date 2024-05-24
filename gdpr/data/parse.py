r"""
parse a .train format files 
(if we make a simple split there is a trouble with DOCSTART)
"""

import os
import pandas as pd

def read_file(path):
    with open(path, 'r') as file:
        text = file.read()

    return text


def parse_sentences_from_text(text):
    r"""
    input – a text,
    output – a list of docs in string format without tags
    """

    text_split_special_token = text.split('-DOCSTART- -X- O O')
    doc_split = [doc.split("\n") for doc in text_split_special_token]
    doc_token_level_split = [[row.split() for row in doc] for doc in doc_split]

    sentence_doc_level = [""]*len(doc_token_level_split)

    i = 0
    for doc in doc_token_level_split:
        for row in doc:
            if(row):
                sentence_doc_level[i] += row[0]
                sentence_doc_level[i] += " "
        i += 1

    return sentence_doc_level

def parse_dataframe(path="../data/", file_name="eng.train"):

    with open(os.path.join(path, file_name), 'r') as file:
        df = pd.read_csv(file, delimiter=' ')

    df.columns = ['token', 'tag1', 'tag2', 'tag3']
    return df

def parse_token_tag_from_text(text):
    r"""
    input – a text,
    output – a list of docs in string format without tags
    """

    text_split_special_token = text.split('-DOCSTART- -X- O O')
    doc_split = [doc.split("\n") for doc in text_split_special_token]
    doc_token_level_split = [[row.split() for row in doc] for doc in doc_split]

    sentence_doc_level = [""]*len(doc_token_level_split)

    i = 0
    for doc in doc_token_level_split:
        for row in doc:
            if(row):
                sentence_doc_level[i] += row[0]
                sentence_doc_level[i] += " "
        i += 1

    return sentence_doc_level

