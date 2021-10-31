import pandas as pd
import qasrl_utils as utils
import argparse
import re


correct_chars = ['[', ']', ' ', ',']

def transform_apos_in_strings(qa_string):
    new_qas_string = qa_string
    index_counter = 0

    for i, char in enumerate(qa_string):
        if i == 0 or i == len(qa_string) - 1:
            continue
    
        if '\'' in char:
            if qa_string[i - 1] in correct_chars or qa_string[i + 1] in correct_chars:
                 new_index = index_counter + i
                 index_counter = index_counter + 1
                 new_qas_string = new_qas_string[:new_index] + "\\" + new_qas_string[new_index:]

    new_qas_string = new_qas_string.replace("\\\\", "\\")
    return new_qas_string

def replace_and_remove(sent):
    new_qas_string = sent
    index_counter = 0

    for i, char in enumerate(sent):
        if i == 0 or i == len(sent) - 1:
            continue

        if '\'' in char:
            if sent[i - 1] == "\\": continue
            else:
                new_index = index_counter + i
                index_counter = index_counter + 1
                new_qas_string = new_qas_string[:new_index] + "\\" + new_qas_string[new_index:]

    new_qas_string = new_qas_string.replace("\\\\", "\\")
    return new_qas_string



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='file containing sentence pairs also in the qasrl file')
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df['qas_1'] = df['qas_1'].apply(lambda x: transform_apos_in_strings(x))
    df['qas_2'] = df['qas_2'].apply(lambda x: transform_apos_in_strings(x))
    df['text_1'] = df['text_1'].apply(lambda x: replace_and_remove(x))
    df['text_2'] = df['text_2'].apply(lambda x: replace_and_remove(x))
    df['text_1_html'] = df['text_1_html'].apply(lambda x: replace_and_remove(x))
    df['text_2_html'] = df['text_2_html'].apply(lambda x: replace_and_remove(x))
    df['prev_text_1'] = df['prev_text_1'].apply(lambda x: replace_and_remove(x))
    df['prev_text_2'] = df['prev_text_2'].apply(lambda x: replace_and_remove(x))

    df.to_csv(args.data, index=False)

