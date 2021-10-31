import pandas as pd
import argparse
from tqdm import tqdm
import itertools

import preprocess_train as utils


'''
This file is for processing data/sentences for prediction.
So for predicting we are using [P] tokens around the predicate in the question and sentence,
plus [A] spans around the answer spans to the question, in the sentence.
And no negative sampling.
'''
def eval_paramts(data):
    data['qas_1'] = data['qas_1'].apply(lambda x: eval(x))
    data['qas_2'] = data['qas_2'].apply(lambda x: eval(x))
    return data

def load_files(file_path):
    print("Loading files")
    data = pd.read_csv(file_path)
    data = eval_paramts(data)
    return data

def create_instances(data, data_split):
    '''

        :param data:
        :param data_split:
        :return:
        '''
    global DATA_SPLIT
    DATA_SPLIT = data_split
    print("Generating data instances for split:  ", data_split)
    data_dict = []
    for i, row in tqdm(data.iterrows(), total=data.shape[0]):
        all_qa_pairs = list(itertools.product(row['qas_1'], row['qas_2']))
        for pair_qa in all_qa_pairs:

            qa1 = pair_qa[0].copy()
            qa2 = pair_qa[1].copy()

            new_inst = utils.create_instance(row, qa1, qa2, True, 0)

            if not new_inst:
                continue
            new_inst = utils.create_inputs(new_inst, False)
            new_inst['abs_sent_id_1'] = row['abs_sent_id_1']
            new_inst['abs_sent_id_2'] = row['abs_sent_id_2']
            data_dict.append(new_inst)

    print("Created split: ", data_split)
    print("Total: ", len(data_dict))
    print()
    return data_dict


def main(args):
    data = load_files(args.input_file)
    processed_data = create_instances(data, args.data_split)

    pd.DataFrame(processed_data).to_csv(args.input_file.replace(".csv","_processed.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='directory containing train,dev, and test files')
    parser.add_argument('--data_split', type=str, required=False, default="train")
    args = parser.parse_args()
    main(args)


