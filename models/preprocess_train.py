import pandas as pd
import argparse
import spacy
import itertools
import data_utils as utils
from random import sample
import nltk
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')
BEG_SPEC_TOKEN = '[A]'
END_SPEC_TOKEN = '[/A]'
BEG_PRED_TOKEN = '[P]'
END_PRED_TOKEN = '[/P]'
K_SAMPLE_RATIO = 10
DATA_SPLIT = None
def tokenize_sentence(sentence):
    tokens = [tok.text for tok in nlp(sentence)]
    return tokens

def find_correct_range(tokens, answer):
    #new_answer = tokenize_sentence(pattern)
    pattern = tokenize_sentence(answer)
    for i,tok in enumerate(tokens):
        if tok == pattern[0]:
            if tokens[i:i+len(pattern)] == pattern:
                return (i,i+len(pattern))," ".join(tokens[i:i+len(pattern)])
            else:
                pat1 = " ".join(tokens[i:i + len(pattern)])
                if pat1.strip(".") == " ".join(pattern).strip("."):
                    return (i, i + len(pattern)), " ".join(tokens[i:i + len(pattern)])

    pattern = answer.split()
    for i,tok in enumerate(tokens):
        if tok == pattern[0]:
            if tokens[i:i+len(pattern)] == pattern:
                return (i,i+len(pattern))," ".join(tokens[i:i+len(pattern)])
            else:
                pat1 = " ".join(tokens[i:i + len(pattern)])
                if pat1.strip(".") == " ".join(pattern).strip("."):
                    return (i, i + len(pattern)), " ".join(tokens[i:i + len(pattern)])

    return None,None

id2tokenize = {}
def get_correct_arg_spans(sent_id, sentence, answer_range,answer):
    #TODO: add verb idx special tokens
    pattern=None
    prev_found=False
    if sent_id in id2tokenize:
        prev_found=True
        tokens = id2tokenize[sent_id]
    else:
        '''
        sentence = sentence.replace("\n", "")
        if sentence[-1] == "." and sentence[-2] != " ":
            tokens = tokenize_sentence(sentence)
        else: tokens = sentence.split()
        '''
        if DATA_SPLIT == "train":
            tokens = tokenize_sentence(sentence.replace("\n",""))
        elif sentence[-1] == "." and sentence[-2] != " ":
            tokens = tokenize_sentence(sentence)
        else: tokens = sentence.split()

    beg,end = answer_range.split(":")
    beg = int(beg)
    end = int(end)
    if DATA_SPLIT == "train":
        end += 1 #if on gold again should be commented out or moved under IF
    if " ".join(tokens[beg:end]) != answer:
        if prev_found:
            new_range,pattern = find_correct_range(tokens, answer)
            if not new_range:
                return None, None, None,None
            beg,end = new_range
        else:
            tokens = tokenize_sentence(sentence)
            if " ".join(tokens[beg:end]) != answer:
                new_range, pattern = find_correct_range(tokens, answer)
                if not new_range:
                    return None, None, None,None
                beg, end = new_range

    if not prev_found:
        id2tokenize[sent_id] = tokens.copy()
    if pattern:
        return beg, end, tokens.copy(), pattern
    return beg,end,tokens.copy(),answer



def add_special_tokens(sent_id,sentence, answer_range,answer, verb_idx=-1):
    #TODO: add verb idx special tokens
    beg,end,tokens,pattern = get_correct_arg_spans(sent_id, sentence, answer_range,answer)
    if not beg and not end:
        return None

    beg_offset = 0
    end_offset = 1
    if verb_idx != -1:
        #check which idx comes first
        if verb_idx < beg:
            tokens.insert(verb_idx, BEG_PRED_TOKEN)
            tokens.insert(verb_idx+2, END_PRED_TOKEN)
            beg_offset += 2
            end_offset += 2
            tokens.insert(beg + beg_offset, BEG_SPEC_TOKEN)
            tokens.insert(end + end_offset, END_SPEC_TOKEN)
        elif verb_idx > beg and verb_idx < end:
            tokens.insert(beg + 0, BEG_SPEC_TOKEN)
            tokens.insert(verb_idx + 1, BEG_PRED_TOKEN)
            tokens.insert(verb_idx + 3, END_PRED_TOKEN)
            tokens.insert(end + 3, END_SPEC_TOKEN)
        else:
            tokens.insert(beg + 0, BEG_SPEC_TOKEN)
            tokens.insert(end + 1, END_SPEC_TOKEN)
            beg_offset += 2
            end_offset += 3
            tokens.insert(verb_idx + beg_offset, BEG_PRED_TOKEN)
            tokens.insert(verb_idx + end_offset, END_PRED_TOKEN)

    else:
        tokens.insert(beg+beg_offset, BEG_SPEC_TOKEN)
        tokens.insert(end+end_offset, END_SPEC_TOKEN)#adding 1 because tokens shifted by 1
    return " ".join(tokens)

def create_context(text, prev_text):
    if ("------" in prev_text) or (prev_text == "NA"):
        return text
    if prev_text.endswith("."):
        return " ".join([prev_text.strip(), text.strip()])
    else: return ". ".join([prev_text.strip(), text.strip()])


def create_instance(row, qa1, qa2, pred_token, label):
    #adding [A]&[/A] for answer spans and , and concatenating contexts
    if pred_token:
        pred_idx = utils.get_pred_idx(qa1['question'], qa1['verb'])
        q_tokens = nltk.word_tokenize(qa1['question'])
        q_tokens.insert(pred_idx, BEG_PRED_TOKEN)
        q_tokens.insert(pred_idx+2, END_PRED_TOKEN)
        question_1 = " ".join(q_tokens)

        pred_idx = utils.get_pred_idx(qa2['question'], qa2['verb'])
        q_tokens = nltk.word_tokenize(qa2['question'])
        q_tokens.insert(pred_idx, BEG_PRED_TOKEN)
        q_tokens.insert(pred_idx + 2, END_PRED_TOKEN)
        question_2 = " ".join(q_tokens)

        new_input_text_1 = add_special_tokens(row['abs_sent_id_1'],row['tokens_1'], qa1['answer_range'],qa1['answer'],qa1['verb_idx'])
        new_input_text_2 = add_special_tokens(row['abs_sent_id_2'],row['tokens_2'], qa2['answer_range'],qa2['answer'],qa2['verb_idx'])
    else:
        new_input_text_1 = add_special_tokens(row['abs_sent_id_1'],row['tokens_1'], qa1['answer_range'],qa1['answer'])
        new_input_text_2 = add_special_tokens(row['abs_sent_id_2'],row['tokens_2'], qa2['answer_range'],qa2['answer'])

        question_1 = qa1['question']
        question_2 = qa2['question']

    if not new_input_text_1 or not new_input_text_2:
        return {}
    if 'prev_text_1' in row:
        new_input_text_1 = create_context(new_input_text_1, row['prev_text_1'])
        new_input_text_2 = create_context(new_input_text_2, row['prev_text_2'])

    new_inst = {"key":row['key'], "text_1":new_input_text_1, "text_2":new_input_text_2,
                "qa_uuid_1":qa1['qa_uuid'], "qa_uuid_2":qa2['qa_uuid'],
                "question_1": question_1, "question_2": question_2, "label":label}
    return new_inst

def create_inputs(inst, sep_token=False):
    if sep_token:
        sep = " </s> "
    else: sep = " [Q] "
    input_1 = inst['question_1'] + sep + inst['text_1']
    input_2 = inst['question_2'] + sep + inst['text_2']
    return {"key":inst['key'], "input_1":input_1, "input_2":input_2, "label": inst['label'],
            "qa_uuid_1":inst['qa_uuid_1'], "qa_uuid_2":inst['qa_uuid_2']}


def eval_paramts(data):
    data['qas_1'] = data['qas_1'].apply(lambda x: eval(x))
    data['qas_2'] = data['qas_2'].apply(lambda x: eval(x))
    data['alignments'] = data['alignments'].apply(lambda x: eval(x))
    return data


def generate_data_instances(data, data_split, neg_sample=False, pred_token=False, for_lemma=False):
    '''

    :param data:
    :param data_split:
    :param neg_sample: whether or not to sample negative instances based on which positives we have
    :param pred_token: TODO-decorate predicate with tokens
    :return:
    '''
    global DATA_SPLIT
    DATA_SPLIT = data_split
    print("Generating data instances for split:  ", data_split)
    data_dict = []
    negc = 0
    many2many = 0
    for i, row in tqdm(data.iterrows(), total=data.shape[0]):
        if row['key'] == 'train6118~!~0.txt~!~61279~!~train6118~!~3.txt~!~72110':
            print("yes")
        neg = []
        pos = 0
        pos_keys = set()
        qas = {qa['qa_uuid']: qa for qa in row['qas_1']}
        qas.update({qa['qa_uuid']: qa for qa in row['qas_2']})
        # collecting positive instances
        verb_keys = set()
        answer_keys = set()
        for align in row['alignments']:  # remove from dict once we added the + instance
            if len(align['sent1']) > 1 or len(align['sent2']) > 1:
                many2many += 1
                continue
            qaid_short1 = "~!~".join(align['sent1'][0]['qa_uuid'].split("~!~")[:-1])
            qaid_short2 = "~!~".join(align['sent2'][0]['qa_uuid'].split("~!~")[:-1])
            key1 = "~!~".join([qaid_short1,align['sent1'][0]['question'],align['sent1'][0]['answer']])
            key2 = "~!~".join([qaid_short2,align['sent2'][0]['question'],align['sent2'][0]['answer']])
            sorted_keys = sorted([key1, key2])
            pos_keys.add("|||".join(sorted_keys))
            qa1 = align['sent1'][0]
            qa2 = align['sent2'][0]

            # adding [A]&[/A] for answer spans, and concatenating contexts
            if for_lemma:
                beg1, end1, tokens1,answer1 = get_correct_arg_spans(row['abs_sent_id_1'], row['tokens_1'], qa1['answer_range'], qa1['answer'])
                beg2, end2, tokens2,answer2 = get_correct_arg_spans(row['abs_sent_id_2'], row['tokens_2'], qa2['answer_range'], qa2['answer'])
                if (not beg1 and not end1) or (not beg2 and not end2):
                    pos_keys.remove(sorted_keys)
                    continue
                new_inst = {"key":row['key'], "abs_sent_id_1":row['abs_sent_id_1'], "abs_sent_id_2":row['abs_sent_id_2'], "text_1": " ".join(tokens1), "text_2":" ".join(tokens2),
                            "qa_uuid_1":key1, "qa_uuid_2":key2,'answer_1':answer1,'answer_2':answer2,
                            "answer_range_1": str(beg1)+":"+str(end1),"answer_range_2": str(beg2)+":"+str(end2),
                            "verb_idx_1": qa1['verb_idx'],"verb_idx_2":qa2['verb_idx'],
                            "question_1": qa1['question'],"question_2":qa2['question'],"verb_1": qa1['verb'],"verb_2":qa2['verb'], "label":1}
            else:
                new_inst = create_instance(row, qa1, qa2, pred_token, 1)
                if not new_inst:
                    pos_keys.remove(sorted_keys)
                    continue
                new_inst = create_inputs(new_inst, args.sep_token)
            data_dict.append(new_inst)

            verb_keys.add(qa1['verb'])
            verb_keys.add(qa2['verb'])
            answer_keys.add(qa1['answer'])
            answer_keys.add(qa2['answer'])
            pos += 1

        # print("pos_examples: ", pos)
        if len(pos_keys) == 0:
            continue
        all_qa_pairs = list(itertools.product(row['qas_1'], row['qas_2']))
        for pair_qa in all_qa_pairs:

            qaid_short1 = "~!~".join(pair_qa[0]['qa_uuid'].split("~!~")[:-1])
            qaid_short2 = "~!~".join(pair_qa[1]['qa_uuid'].split("~!~")[:-1])
            key1 = "~!~".join([qaid_short1,pair_qa[0]['question'], pair_qa[0]['answer']])
            key2 = "~!~".join([qaid_short2, pair_qa[1]['question'], pair_qa[1]['answer']])
            sorted_keys = "|||".join(sorted([key1, key2]))
            if sorted_keys in pos_keys:
                continue

            if neg_sample:  # do smart filtering - if verb is not from the positive ones, and answer is not in the positive ones, then we don't add it
                verbs_notin_pos = False
                answer_notin_pos = False
                if (pair_qa[0]['verb'] not in verb_keys) and (pair_qa[1]['verb'] not in verb_keys):
                    verbs_notin_pos = True
                if (pair_qa[0]['answer'] not in answer_keys) and (pair_qa[1]['answer'] not in answer_keys):
                    answer_notin_pos = True
                if verbs_notin_pos and answer_notin_pos:
                    continue
                # then we add this sample

            qa1 = pair_qa[0].copy()
            qa2 = pair_qa[1].copy()

            if for_lemma:
                beg1, end1, tokens1, answer1 = get_correct_arg_spans(row['abs_sent_id_1'], row['tokens_1'],
                                                                     qa1['answer_range'], qa1['answer'])
                beg2, end2, tokens2, answer2 = get_correct_arg_spans(row['abs_sent_id_2'], row['tokens_2'],
                                                                     qa2['answer_range'], qa2['answer'])

                if (not beg1 and not end1) or (not beg2 and not end2):
                    continue
                new_inst = {"key": row['key'], "abs_sent_id_1": row['abs_sent_id_1'],
                            "abs_sent_id_2": row['abs_sent_id_2'], "text_1": " ".join(tokens1),
                            "text_2": " ".join(tokens2),
                            "qa_uuid_1": key1, "qa_uuid_2": key2, 'answer_1': answer1, 'answer_2': answer2,
                            "answer_range_1": str(beg1) + ":" + str(end1),
                            "answer_range_2": str(beg2) + ":" + str(end2),
                            "verb_idx_1": qa1['verb_idx'], "verb_idx_2": qa2['verb_idx'],
                            "question_1": qa1['question'], "question_2": qa2['question'], "verb_1": qa1['verb'],
                            "verb_2": qa2['verb'], "label": 0}

            else:
                new_inst = create_instance(row, qa1, qa2, pred_token, 0)
                if not new_inst:
                    continue
                new_inst = create_inputs(new_inst, args.sep_token)
            neg.append(new_inst)

        if (len(neg) != (len(all_qa_pairs) - pos)) and not neg_sample:
            print("WARNING, negatives more than should")
            print(row['key'])
        if len(neg) / (pos + 1) > K_SAMPLE_RATIO and ( not for_lemma and neg_sample):  # ratio of larger than kx
            if pos != 0:
                sampled_neg = sample(neg, K_SAMPLE_RATIO * (pos))
            else:
                sampled_neg = sample(neg, K_SAMPLE_RATIO)
            data_dict.extend(sampled_neg)
            negc += len(sampled_neg)
        else:
            data_dict.extend(neg)
            negc += len(neg)
            # print("neg_examples: ", len(neg))
    print("Created split: ", data_split)
    print("Total: ", len(data_dict))
    print("Positive: ", len(data_dict)-negc)
    print("Negative: ", negc)
    print("Many to many skipped: ", many2many)
    print()
    return {data_split: data_dict}

def load_files(dir_path):
    print("Loading files")
    #train = pd.read_csv(dir_path+"/gold_qa_alignment_train_fixed.csv")
    train = pd.read_csv(dir_path + "/gold_qa_alignment_train_fixed.csv")
    train = eval_paramts(train)
    val = pd.read_csv(dir_path+"/gold_qa_alignment_val_fixed.csv")
    val = eval_paramts(val)
    test = pd.read_csv(dir_path +"/gold_qa_alignment_test_fixed.csv")
    test = eval_paramts(test)
    return train,val,test


def main(args):
    '''

    a. load from dir three files with names "train/dev/test.csv"
    b. create candidates for each row
    c. have options for smart selection
    '''
    if args.for_lemma:
        _, val, test = load_files(args.input_dir)
        data = {}
        data.update(generate_data_instances(val, "val", False, args.pred_token, True))
        print("saving val")
        val_df = pd.DataFrame(data['val'])
        val_df.to_csv("../data/train/fixed_input/val_for_lemma.csv", index=False)
        data.update(generate_data_instances(test, "test", False, args.pred_token, True))
        print("saving test")
        test_df = pd.DataFrame(data['test'])
        test_df.to_csv("../data/train/fixed_input/test_for_lemma.csv", index=False)
    else:
        train, val, test = load_files(args.input_dir)
        data = {}
        output_name = args.out_name
        data.update(generate_data_instances(train, "train", args.neg_sample, args.pred_token))
        print("Saving train")
        pd.DataFrame(data['train']).to_csv(args.out_dir+"/train_" + output_name, index=False)
        data.update(generate_data_instances(val, "val", args.neg_sample, args.pred_token))
        print("saving val")
        pd.DataFrame(data['val']).to_csv(args.out_dir+"/val_" + output_name, index=False)
        data.update(generate_data_instances(test, "test", args.neg_sample, args.pred_token))
        print("saving test")
        pd.DataFrame(data['test']).to_csv(args.out_dir+"/test_" + output_name, index=False)


if __name__ == "__main__":
    '''
    for predicting on extended fusion data
    --neg_sample, --pred_token, and no need for sep.
    
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=False)
    parser.add_argument('--input_dir', type=str, required=True, help='directory containing train,dev, and test files')
    parser.add_argument('--neg_sample', action="store_true",default=False, help='whether to sample "smart" from neg examples')
    parser.add_argument('--pred_token', action="store_true", default=False,
                        help='whether to add special token around the predicate')
    parser.add_argument('--for_lemma', action="store_true", default=False,
                        help='whether to parse data for the lemma baseline')
    parser.add_argument('--sep_token', action="store_true", default=False,
                        help='whether to add sep for roberta')
    parser.add_argument('--out_dir', type=str, required=False, default="../data/predict/")
    parser.add_argument('--out_name', type=str, required=False,default="processed_wpred.csv")

    args = parser.parse_args()
    main(args)