import pandas as pd
import argparse
from tqdm import tqdm
import itertools
import os
from collections import defaultdict,OrderedDict

def get_sorted(x):
    key = [x['abs_sent_id_1'],x['abs_sent_id_2']]
    key = sorted(key)
    return "~!~".join(key)

CLUSTER_SIZE=True

def add_predicate_spans_only(alignum, quuids_1, quuids_2, qasrl_data, sent_1_dict, sent_2_dict,align2index,args):
    sent1_df = qasrl_data[qasrl_data.qa_uuid.isin(quuids_1)]
    sent2_df = qasrl_data[qasrl_data.qa_uuid.isin(quuids_2)]

    sent_1_key = list(sent_1_dict.keys())[0]
    sent_1_align2tok = list(sent_1_dict.values())[0]
    sent_2_key = list(sent_2_dict.keys())[0]
    sent_2_align2tok = list(sent_2_dict.values())[0]

    if sent_1_align2tok['tokens'] == "":
        tokens_1 = sent1_df['tokens'].iloc[0]
        tokens_1 = " ".join(tokens_1)
        sent_1_align2tok['tokens'] = tokens_1
    if sent_2_align2tok['tokens'] == "":
        tokens_2 = sent2_df['tokens'].iloc[0]
        tokens_2 = " ".join(tokens_2)
        sent_2_align2tok['tokens'] = tokens_2

    verbs_1 = sent1_df['verb'].to_list()
    verbs_2 = sent2_df['verb'].to_list()

    alignment_appeared = set()
    # print("aligns: ",[answers_1,verbs_1,answers_2,verbs_2])
    for pair in zip(*[verbs_1, verbs_2]):
        verb1 = pair[0]
        verb2 = pair[1]
        if str(verb1 + verb2) in alignment_appeared:  # if this exact alignment appeared already, just with a different role, well skip
            #print("Not adding: ", str(verb1 + verb2))
            continue
        alignment_appeared.add(str(verb1 + verb2))

        # text1
        sent_1_align2tok['v2align'][verb1].append(str(alignum))
        # text2
        sent_2_align2tok['v2align'][verb2].append(str(alignum))
        alignum += 1

    return alignum, {sent_1_key: sent_1_align2tok}, {sent_2_key: sent_2_align2tok}, align2index


def add_argument_spans_only(alignum, quuids_1, quuids_2, qasrl_data, sent_1_dict, sent_2_dict,align2index,args):
    sent1_df = qasrl_data[qasrl_data.qa_uuid.isin(quuids_1)]
    sent2_df = qasrl_data[qasrl_data.qa_uuid.isin(quuids_2)]

    sent_1_key = list(sent_1_dict.keys())[0]
    sent_1_align2tok = list(sent_1_dict.values())[0]
    sent_2_key = list(sent_2_dict.keys())[0]
    sent_2_align2tok = list(sent_2_dict.values())[0]

    if sent_1_align2tok['tokens'] == "":
        tokens_1 = sent1_df['tokens'].iloc[0]
        tokens_1 = " ".join(tokens_1)
        sent_1_align2tok['tokens'] = tokens_1
    if sent_2_align2tok['tokens'] == "":
        tokens_2 = sent2_df['tokens'].iloc[0]
        tokens_2 = " ".join(tokens_2)
        sent_2_align2tok['tokens'] = tokens_2

    answers_1 = sent1_df['answer'].to_list()
    answers_2 = sent2_df['answer'].to_list()

    alignment_appeared = set()
    # print("aligns: ",[answers_1,verbs_1,answers_2,verbs_2])
    for pair in zip(*[answers_1, answers_2]):
        answer1 = pair[0]
        answer2 = pair[1]
        if str(answer1 + answer2) in alignment_appeared:  # if this exact alignment appeared already, just with a different role, well skip
            #print("Not adding: ", str(answer1 + answer2))
            continue
        alignment_appeared.add(str(answer1 + answer2))

        # text1
        sent_1_align2tok['a2align'][answer1].append(str(alignum))
        # text2
        sent_2_align2tok['a2align'][answer2].append(str(alignum))
        alignum += 1

    return alignum, {sent_1_key: sent_1_align2tok}, {sent_2_key: sent_2_align2tok}, align2index




def count_alignment_spans(alignum, quuids_1, quuids_2, qasrl_data, sent_1_dict, sent_2_dict, align2index,args):
    '''
    The way we are currently wrapping answers and predicates, we are missing role information, but also the question text itself.
    quuids_1 - list of qa_uuids from all alignments (with sentence 2) from sentence 1
    quuids_2 - list of qa_uuids from all alignments (with sentence 1) from sentence 2
    qasrl_data - all generated questions
    '''
    rows = []
    for qa_id1 in quuids_1:
        row = qasrl_data[qasrl_data.qa_uuid == qa_id1].iloc[0]
        rows.append(row)
    sent1_df = pd.DataFrame(rows)
    rows = []
    for qa_id2 in quuids_2:
        row = qasrl_data[qasrl_data.qa_uuid == qa_id2].iloc[0]
        rows.append(row)
    sent2_df = pd.DataFrame(rows)

    sent_1_key = list(sent_1_dict.keys())[0]
    sent_1_align2tok = list(sent_1_dict.values())[0]
    sent_2_key = list(sent_2_dict.keys())[0]
    sent_2_align2tok = list(sent_2_dict.values())[0]

    if sent_1_align2tok['tokens'] == "":
        tokens_1 = sent1_df['tokens'].iloc[0]
        tokens_1 = " ".join(tokens_1)
        sent_1_align2tok['tokens'] = tokens_1
    if sent_2_align2tok['tokens'] == "":
        tokens_2 = sent2_df['tokens'].iloc[0]
        tokens_2 = " ".join(tokens_2)
        sent_2_align2tok['tokens'] = tokens_2

    answers_1 = sent1_df['answer'].to_list()
    answers_2 = sent2_df['answer'].to_list()
    verbs_1 = sent1_df['verb'].to_list()
    verbs_2 = sent2_df['verb'].to_list()

    alignment_appeared = set()
    # print("aligns: ",[answers_1,verbs_1,answers_2,verbs_2])
    for pair in zip(*[quuids_1, answers_1, verbs_1, quuids_2, answers_2, verbs_2]):
        qa1 = pair[0]
        answer1 = pair[1]
        verb1 = pair[2]
        qa2 = pair[3]
        answer2 = pair[4]
        verb2 = pair[5]

        if str(answer1 + verb1 + answer2 + verb2) in alignment_appeared:  # if this exact alignment appeared already, just with a different role, well skip
            print("Not adding: ", str(answer1 + verb1 + answer2 + verb2))
            continue
        alignment_appeared.add(str(answer1 + verb1 + answer2 + verb2))

        if qa1 in align2index and qa2 not in align2index:  # this align gets the index of sent1 that already has an index for the same QA
            prev_alignnum = align2index[qa1]
            if prev_alignnum not in sent_2_align2tok['a2align'][answer2]:
                sent_2_align2tok['a2align'][answer2].extend(prev_alignnum)
                sent_2_align2tok['v2align'][verb2].extend(prev_alignnum)
            align2index[qa2].extend(prev_alignnum)
        elif qa2 in align2index and qa1 not in align2index:  # this align gets the index of sent1 that already has an index for the same QA
            prev_alignnum = align2index[qa2]
            if prev_alignnum not in sent_1_align2tok['a2align'][answer1]:
                sent_1_align2tok['a2align'][answer1].extend(prev_alignnum)
                sent_1_align2tok['v2align'][verb1].extend(prev_alignnum)
            align2index[qa1].extend(prev_alignnum)
        elif qa1 in align2index and qa2 in align2index: #then both are already aligned to a third.
            continue
        else:
            # text1
            if alignum not in sent_1_align2tok['a2align'][answer1]:
                sent_1_align2tok['a2align'][answer1].append(str(alignum))
                sent_1_align2tok['v2align'][verb1].append(str(alignum))
            # text2
            if alignum not in sent_2_align2tok['a2align'][answer2]:
                sent_2_align2tok['a2align'][answer2].append(str(alignum))
                sent_2_align2tok['v2align'][verb2].append(str(alignum))
            align2index[qa1].append(str(alignum))
            align2index[qa2].append(str(alignum))
            alignum += 1

    return alignum, {sent_1_key:sent_1_align2tok}, {sent_2_key:sent_2_align2tok},align2index

def attach_alignment_spans_wo_ids(sent_span_2_indices):
    tokens = sent_span_2_indices['tokens']
    split_tokens = tokens.split(" ")
    for k,spans in sent_span_2_indices.items(): #iterating over answers and verbs
        if k == 'a2align':
            for answer,indices in spans.items():
                answer_l = answer.split(" ")
                if answer_l[0] == split_tokens[0]:
                    tokens = tokens.replace(answer + " ", "[A] " + answer + " " + "[\A] ",1)
                elif answer_l[-1] == split_tokens[-1]:
                    tokens = tokens.replace(" " + answer, " [A]" + " " + answer + " [\A]",1)
                else:
                    tokens = tokens.replace(" "+answer+" ", " [A]" + " "+answer+" " + "[\A] ",1)
        elif k == 'v2align':
            for verb,indices in spans.items():
                tokens = tokens.replace(" "+verb+" ", " [P]" + " "+verb+" " + "[\P] ")

    return tokens

def attach_alignment_spans(sent_span_2_indices):
    tokens= sent_span_2_indices['tokens']
    split_tokens = tokens.split(" ")
    for k,spans in sent_span_2_indices.items(): #iterating over answers and verbs
        if k == 'a2align':
            for answer,indices in spans.items():
                answer_l = answer.split(" ")
                token_ids= "".join(set(indices))
                if answer_l[0] == split_tokens[0]:
                    tokens = tokens.replace(answer + " ","[A" + token_ids + "] " +answer + " " + "[\A" + token_ids + "] ", 1)
                elif answer_l[-1] == split_tokens[-1]:
                    tokens = tokens.replace(" "+answer, " [A" + token_ids + "]" +" "+ answer + " [\A" + token_ids + "]", 1)
                else:
                    tokens = tokens.replace(" "+answer+" ", " [A" + token_ids + "]" + " "+answer+" " + "[\A" + token_ids + "] ",1)
        elif k == 'v2align':
            for verb,indices in spans.items():
                token_ids= "".join(set(indices))
                tokens = tokens.replace(" "+verb+" ", " [P" + token_ids + "]" + " "+verb+" " + "[\P" + token_ids + "] ")

    return tokens


def combine_aligned_text_in_clusters(fusion_input, pred_data, qasrl_data, args):
    align2index = defaultdict(list)

    all_qa_pairs = list(itertools.combinations(zip(fusion_input['abs_sent_id'].to_list(), fusion_input['text'].to_list()),2))

    # these are the keys and pairs of texts that appear in this fusion cluster
    uniq_sent_2_span_tok = OrderedDict()
    for item in fusion_input['abs_sent_id'].to_list():
        uniq_sent_2_span_tok[item] = {'a2align': defaultdict(list), 'v2align': defaultdict(list), 'tokens': ""}

    current_align_number = 1
    pairs_wo_aligns = 0
    for pair_qa in all_qa_pairs:
        key = get_sorted({"abs_sent_id_1":pair_qa[0][0], "abs_sent_id_2":pair_qa[1][0]})
        df = pred_data[(pred_data.key == key) & (pred_data.pred == 1)]  # all predicted true alignments
        if len(df) == 0:  # there were no found alignments for this pair
            pairs_wo_aligns += 1
            continue

        #we select non alignments if doing ablation study
        if args.ablation:
            pos_labels = len(df)
            df = pred_data[(pred_data.key == key) & (pred_data.pred == 0)] #same # as pos labels
            if len(df) == 0:
                continue
            if len(df) >= pos_labels:
                df = df.sample(pos_labels)


        #df['qauuid_key'] = df.apply(lambda x: get_sorted({"abs_sent_id_1":x['qa_uuid_1'], "abs_sent_id_2":x['qa_uuid_2']}),axis=1)
        #df.drop_duplicates("qauuid_key", inplace=True)
        #df.drop_duplicates("qa_uuid_1", inplace=True)
        #df.drop_duplicates("qa_uuid_2", inplace=True)
        #if pair_qa[0][0] not in uniq_sent_2_span_tok:
        #    uniq_sent_2_span_tok[pair_qa[0][0]] = {'a2align': defaultdict(list), 'v2align': defaultdict(list), 'tokens': ""}
        #if pair_qa[1][0] not in uniq_sent_2_span_tok:
        #    uniq_sent_2_span_tok[pair_qa[1][0]] = {'a2align': defaultdict(list), 'v2align': defaultdict(list), 'tokens': ""}

        qa_uuids_1 = df['qa_uuid_1'].to_list()
        qa_uuids_2 = df['qa_uuid_2'].to_list()
        sent_1_dict = {}
        sent_2_dict = {}
        if "~!~".join(qa_uuids_1[0].split("~!~")[:4]) in pair_qa[0][0]:
            sent_1_dict[pair_qa[0][0]] = uniq_sent_2_span_tok[pair_qa[0][0]]
            sent_2_dict[pair_qa[1][0]] = uniq_sent_2_span_tok[pair_qa[1][0]]
        else:
            sent_1_dict[pair_qa[1][0]] = uniq_sent_2_span_tok[pair_qa[1][0]]
            sent_2_dict[pair_qa[0][0]] = uniq_sent_2_span_tok[pair_qa[0][0]]

        if args.only_pred:
            current_align_number, sent_1_align2tok, sent_2_align2tok, align2index = add_predicate_spans_only(
                current_align_number, qa_uuids_1, qa_uuids_2,
                qasrl_data, sent_1_dict,
                sent_2_dict,
                align2index, args)
        elif args.only_arg:
            current_align_number, sent_1_align2tok, sent_2_align2tok, align2index = add_argument_spans_only(
                current_align_number, qa_uuids_1, qa_uuids_2,
                qasrl_data, sent_1_dict,
                sent_2_dict,
                align2index, args)
        else:
            current_align_number, sent_1_align2tok, sent_2_align2tok,align2index = count_alignment_spans(current_align_number, qa_uuids_1, qa_uuids_2,
                                                                            qasrl_data, sent_1_dict,
                                                                            sent_2_dict,
                                                                            align2index,args)
        if pair_qa[0][0] in sent_1_align2tok:
            uniq_sent_2_span_tok[pair_qa[0][0]] = sent_1_align2tok[pair_qa[0][0]]
            uniq_sent_2_span_tok[pair_qa[1][0]] = sent_2_align2tok[pair_qa[1][0]]
        else:
            uniq_sent_2_span_tok[pair_qa[1][0]] = sent_1_align2tok[pair_qa[1][0]]
            uniq_sent_2_span_tok[pair_qa[0][0]] = sent_2_align2tok[pair_qa[0][0]]

    fusion_tokens = []
    fusion_tokens_wo_nums = []
    for sent_id,obj in uniq_sent_2_span_tok.items():
        #toks = attach_alignment_spans_wo_ids(obj)
        if len(obj['tokens']) == 0:
            continue
        toks = attach_alignment_spans(obj)
        toks_wo_nums = attach_alignment_spans_wo_ids(obj)
        fusion_tokens.append(toks)
        fusion_tokens_wo_nums.append(toks_wo_nums)

    for i,row in fusion_input.iterrows():
        if len(uniq_sent_2_span_tok[row['abs_sent_id']]['tokens']) == 0:
            fusion_tokens.append(row['text'])
            fusion_tokens_wo_nums.append(row['text'])


    return fusion_tokens,fusion_tokens_wo_nums



def create_training_format(fus_data, pred_data, qasrl_data, args):
    import random
    '''

    :param df: Data to format into training
    :param cluster_size: indicates to create clusters up to size 4
    :return:
    '''
    # df = df.sample(frac=1).reset_index(drop=True) #shuffling our data before we split it
    pairs_wno_aligns = 0
    total_pairs = 0
    total_size_2 = 0
    avg_num_aligns = []
    g = fus_data.groupby('abs_scu_id')
    data = defaultdict(list)
    data_thadani = defaultdict(list)
    data_wo_num = defaultdict(list)
    orig_key = {}
    avg_len_of_inputs = []

    for i, df in tqdm(g, desc="Creating fusions with alignments..."):  # iterating over fusion instances
  # sorting by word overlap between source and target sentence
        #data['target'].append(df['scu_label'].iloc[0].strip())  # appending the output for this training instance
        data['target'].append(df['scu_label'].iloc[0].strip())
        data_thadani['target'].append(df['scu_label'].iloc[0].strip())
        data_wo_num['target'].append(df['scu_label'].iloc[0].strip())

        #if CLUSTER_SIZE:
        # we shuffle what we extract, since we don't want our input to be biased/sensitive towards
        # sentences that appear in the beginning, since they will have the most related words to the target.
        if len(df) > 4:
            df.sort_values('w_overlap', inplace=True,
                           ascending=False)
        source = df[:4]

        '''
        if args.thadani:
            source = source.sample(frac=1).reset_index(drop=True)
            data['source'].append([item.strip() for item in source['text'].to_list()])
            avg_len_of_inputs.append(len(df))
            continue
        '''

        source = source.sample(frac=1).reset_index(drop=True)
        data_thadani['source'].append([item.strip() for item in source['text'].to_list()])
        tokens, tokens_wo_num = combine_aligned_text_in_clusters(source, pred_data, qasrl_data,args)
        #random.shuffle(tokens)
        data['source'].append([item.strip() for item in tokens])
        data_wo_num['source'].append([item.strip() for item in tokens_wo_num])
        avg_len_of_inputs.append(len(tokens))
        '''
        else:

            if args.thadani:
                df = df.sample(frac=1).reset_index(drop=True)
                data['source'].append([item.strip() for item in df['text'].to_list()])
                avg_len_of_inputs.append(len(df))
                continue
            #for full clusters
            tokens, num_pairs, num_aligns, num_wno_aligns = combine_aligned_text_in_clusters(df, pred_data,
                                                                                             qasrl_data)
            random.shuffle(tokens)
            total_pairs += num_pairs
            pairs_wno_aligns += num_wno_aligns
            avg_num_aligns.append(num_aligns-1)
            data['source'].append([item.strip() for item in tokens])
        '''


    print("Number of pairs w/o alignments: ", pairs_wno_aligns)
    print("Out of: ", total_pairs)
    if len(data['source']) != len(data['target']): raise ValueError(
        "Malformed data, source and target should be of equal length")
    print("Out of: ", total_pairs)
    if len(data_thadani['source']) != len(data_thadani['target']): raise ValueError(
        "Malformed data, source and target should be of equal length")
    print("sum of all input sentences: ", sum(avg_len_of_inputs))
    print("avg of all input sentences: ", sum(avg_len_of_inputs)/len(avg_len_of_inputs))
    return data,data_thadani,data_wo_num

def concate_input_sentences(source, separator, special_token=True):
    new_inputs = []
    for finst in source:
        sents = []
        for i,sent in enumerate(finst):
                sents.append(sent)
        new_inputs.append(separator.join(sents))
    return new_inputs

def build_input(source):
    data = source.copy()
    data['source'] = concate_input_sentences(source['source'], " </s> ", False) #.... </s> .....

    if len(data['source']) != len(data['target']): raise ValueError(
        "Malformed source, source and target should be of equal length")

    print("Source and target lengths: ",len(data['source']), len(data['target']))

    return data

def save_data(data, output_dir, split="train"):
    source = data['source']
    target = data['target']

    data = [source, target]
    file_paths = [split+".source", split+".target"]

    for i,file in enumerate(file_paths):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_dir + "/" + file, "w") as f:
            print('Data split: ',len(data[i]))
            for i,line in enumerate(data[i]):

                f.write(" ".join(line.strip().split("\n")))
                f.write("\n")
    return

def main(args):
    pred_data = pd.read_csv(args.pred_file)
    fusion_data = pd.read_csv(args.fusion_file)
    qasrl_data = pd.read_csv(args.qasrl_file)
    qasrl_data['tokens'] = qasrl_data['tokens'].apply(lambda x: eval(x))
    for i in range(20):
        print("Creating experiment #",i,' for split: ', args.data_split)
        formatted_data,thadani_data, formatted_data_wonum = create_training_format(fusion_data, pred_data, qasrl_data, args)
        formatted_data = build_input(formatted_data)
        thadani_data = build_input(thadani_data)
        formatted_data_wonum = build_input(formatted_data_wonum)
        save_data(formatted_data, args.out_dir + "align_wnums_#"+str(i), split=args.data_split)
        save_data(thadani_data, args.out_dir + "thadani_#" + str(i), split=args.data_split)
        save_data(formatted_data_wonum, args.out_dir + "align_NO_nums_#" + str(i), split=args.data_split)
    #pd.DataFrame(key2dups.items()).to_csv(args.out_dir+"/key2dups.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, required=True, help='')
    parser.add_argument('--fusion_file', type=str, required=True, help='')
    parser.add_argument('--qasrl_file', type=str, required=True, help='')
    parser.add_argument('--out_dir', type=str, required=True, help='directory path for final train,dev,test files')
    parser.add_argument('--data_split', type=str, required=False, default="train")

    #ablation experiments
    parser.add_argument('--thadani', action='store_true', required=False, default=False)
    parser.add_argument('--no_align', action='store_true', required=False, default=False, help='No alignments, create regular fusion clusters')
    parser.add_argument('--save_order', action='store_true', required=False, default=False)
    parser.add_argument('--ablation', action='store_true', required=False, default=False)
    parser.add_argument('--only_pred',action='store_true', required=False, default=False)
    parser.add_argument('--only_arg',action='store_true', required=False, default=False)
    #parser.add_argument('--rand_pa',action='store_true', required=False)
    #parser.add_argument('--all_pa',action='store_true', required=False)
    args = parser.parse_args()
    main(args)