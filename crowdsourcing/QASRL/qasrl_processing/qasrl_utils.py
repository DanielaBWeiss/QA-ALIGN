import pandas as pd
import json
import nltk
from nltk.stem import PorterStemmer
nltk.download('averaged_perceptron_tagger')
ps = PorterStemmer()
import re

'''
functions for:
 - preparing data for qasrl parser input
 - cleaning csv files post qasrl annotations
 - preparing post qasrl annotations for qa-alignment annotations

necessary columns for qa-alignment annotations UI:
 ['qa_uuid', 'qasrl_id', 'verb', 'question', 'answer', 'sentence','answer_range', 'verb_idx']
'''

def parse_data_for_qasrl_input(file_path, filename, output_dir):
    df = pd.read_csv(file_path)
    sents = df["sentence"]
    with open(output_dir + "/" + filename +".json", "w") as f:
        for sent in sents:
            sen = {"sentence": sent}
            json.dump(sen, f)
            f.write('\n')


def replace_brackets(answer):
    if "-LRB-" in answer:
        answer = answer.replace("-LRB-", "(")
    if "-RRB-" in answer:
        answer = answer.replace("-RRB-", ")")
    if "-LSB-" in answer:
        answer = answer.replace("-LSB-", "[")
    if "-RSB-" in answer:
        answer = answer.replace("-RSB-", "]")
    return answer

def clean_qasrl_annots_post_arb_mturk(df):
    '''
    We want to remove all QAs that have "is_redundant" false and answer is empty, these rows are the "invalid" questions
    :param df: post qasrl annotation dataframe (using mturk)
    :return: same df, but without redundant annotations, and cleaned text
    '''
    df = df[(df.is_redundant == False)]
    df = df[~df.answer.isnull()]
    df["answer"] = df["answer"].apply(lambda x: replace_brackets(x))
    return df

def clean_qasrl_annots_post_gen_mturk(df):
    '''
    We want to remove all QAs that the answer is empty, these rows are the "invalid" questions
    These are annotations that did not go through arbitration.
    :param df: post generation qasrl annotation dataframe (using mturk)
    :return: same df, but without redundant annotations, and cleaned text
    '''
    df = df[~df.answer.isnull()]
    df["answer"] = df["answer"].apply(lambda x: replace_brackets(x))
    return df

def get_qasrl_annots_post_parser(parer_output_file, ids_list, id2sent_dict):
    '''

    :param parer_output_file: the file output that running the qasrl parser results in (.jsonl)
    :param ids_list: list of qasrl ids in order of which they appear in qasrl parser input (.jsonl)
    :param id2sent_dict: dictionary of qasrl_ids to original sentence
    :return: qasrl dataframe post annotation (using parser)
    '''
    qasrl = []
    with open(parer_output_file) as f:
        for line in f:
            qasrl.append(json.loads(line))

    qas = []
    for i, qa in enumerate(qasrl):
        qasrl_id = ids_list[i]
        for verb_obj in qa["verbs"]:
            verb = verb_obj["verb"]
            verb_ind = verb_obj["index"]
            for qa_pair in verb_obj["qa_pairs"]:
                row = (qasrl_id, verb, qa_pair["question"], qa_pair["spans"][0]["text"], id2sent_dict[qasrl_id],
                       str(qa_pair["spans"][0]["start"]) + ":" + str(qa_pair["spans"][0]["end"]), verb_ind)
                qas.append(row)

    fusion_qasrl = pd.DataFrame(qas)
    fusion_qasrl.reset_index(inplace=True)
    fusion_qasrl.columns = ['qa_uuid', 'qasrl_id', 'verb', 'question', 'answer', 'sentence',
                            'answer_range', 'verb_idx']



def split_argument_answers_post_mturk(df):
    '''
    This function is intended for qa-alignments annotations, where we want to split every question that contains
    multiple answers (~!~), and create its own row with qa_uuid
    :param df: dataframe with qasrl annotations
    :return: same dataframe with new QA rows, and duplicate questions
    '''
    rows_to_add = []
    to_delete = []
    print("Previously, df was size of: ", len(df))
    for i, row in df.iterrows():
        if "~!~" in row.answer:
            args = row.answer.split('~!~')
            ranges = row.answer_range.split('~!~')

            for j, arg in enumerate(args):
                row_copy = row.copy()
                row_copy["answer"] = arg
                row_copy["answer_range"] = ranges[j]
                rows_to_add.append(row_copy)

            to_delete.append(row)

    df_delete = pd.DataFrame(to_delete)
    df_add = pd.DataFrame(rows_to_add)

    print("Deleting {} rows".format(len(df_delete)))
    print("Adding {} rows".format(len(df_add)))

    df.drop(df_delete.index, inplace=True)
    df = df.append(df_add)
    #df.reset_index(inplace=True)
    #df.drop("index", axis=1, inplace=True)
    df.reset_index(inplace=True)
    cols = list(df.columns)
    cols[0] = "qa_uuid"
    df.columns = cols
    return df

def replace_quotes_in_qasrl(qa):
    #qa[2] is the question, qa[3] is the answer.
    if "\'" in qa[2]:
        new_qa = qa[2].replace("\'", "\\'")
        new_qa = new_qa.replace("\\\\", "\\")
        return (qa[0], qa[1], new_qa, qa[3])

    return qa

'''
functions for getting qa-alignment annotations
necessary columns for qa-alignment annotation collection using UI:
   [ 'text_1', 'text_2', 'abs_sent_id_1', 'abs_sent_id_2', 'qas_1', 'qas_2', 'prev_text_1','prev_text_2']
'''
def get_qas_per_qasrlid(df):
    '''

    :param df: post qasrl annotation file with the following
    :return: dict, key:qasrl_id, value: list of list items, where each item represents a QA
    "qasrl_id":[ [ "qa_uuid","verb","question","answer","answer_range"]]
    '''

    grouped = df.groupby("qasrl_id")
    qaid2qas = {}
    qaid2qas_d = {}
    for i, g in grouped:
        qa_zip = list(zip(g["qa_uuid"].tolist(), g["verb"].tolist(),g["question_html"].tolist(),
                          g["answer"].tolist()))

        qa_zip.sort(key=lambda x: x[1])
        list_qas = []
        list_qas_d = []
        for item in qa_zip:
            #new_item = replace_quotes_in_qasrl(item)
            list_qas.append(list(item))
            list_qas_d.append({"qa_uuid": item[0], "verb":item[1], "question_html": item[2],
                     "answer": item[3]})
        qaid2qas[i] = list_qas
        qaid2qas_d[i] = list_qas_d

    return qaid2qas, qaid2qas_d

def get_verbs(qal):
    verb_set = set()
    for qa in qal:
        verb_set.add(qa[1]) #verb, verb_idx
    return verb_set

def find_verbs(sent, verbs):
    for verb in verbs:
        for token in sent.split(" "):
            if verb.lower() == token.lower():
                sent = sent.replace(token, "<strong>"+token+"</strong>")

    return sent

def get_html_text(df):
    '''

    :param df: prepped dataframe with pairs of sentences ready for anntoations.
    The following columns are a must (diff datasets have extra different columns):
    [ 'text_1', 'text_2', 'qasrl_id_1', 'qasrl_id_2', 'qas_1', 'qas_2', 'prev_text_1','prev_text_2']
    :return: two columns that are needed for the UI collection, "sent1_html" & "sent2_html ,which contains the text with
    <strong></strong> tags around the verbs that exist in the text.
    '''

    sent1_html = []
    sent2_html = []
    for i, row in df.iterrows():
        verbs = get_verbs(row["qas_1"])
        sent1 = row["tokens_1"]
        nsent = find_verbs(sent1, verbs)
        sent1_html.append(nsent)

        verbs = get_verbs(row["qas_2"])
        sent2 = row["tokens_2"]
        nsent = find_verbs(sent2, verbs)
        sent2_html.append(nsent)

    return sent1_html, sent2_html

def get_html_questions(question, verb):
    particles_df = pd.read_csv("particles.csv")
    new_question = question
    not_verbs = particles_df.particles.to_list()
    particle_verbs = ["have", "had", "has"]
    #checking if a sentence has an identifiable verb, beyond the first token
    stemmed_verb = ps.stem(verb)
    token_list = nltk.word_tokenize(question)
    cnt = 0
    for i,token in enumerate(token_list):
        if i == 0: continue
        stemmed_tok = ps.stem(token)
        if stemmed_tok.lower() == stemmed_verb.lower():
            new_question = new_question.replace(token, "strong1"+token+"strong2")
            return new_question
        else:
            if stemmed_verb in particle_verbs or verb in particle_verbs:
                if token.lower() in particle_verbs:
                    new_question = new_question.replace(token, "strong1"+token+"strong2")
                    return new_question
            if token not in not_verbs:
                new_question = new_question.replace(token, "strong1"+token+"strong2")
                return new_question

    pos = nltk.pos_tag(token_list)
    #checking if a sentence has an identifiable verb, beyond the first token
    cnt = 0
    for token_pos in enumerate(pos):
        if "VB" in token_pos[1][1]:
            new_question = new_question.replace(token_pos[1][0], "strong1" + token_pos[1][0] + "strong2")
            return new_question
    return question


def get_qas_per_qasrlid_parser(df):
    '''

    :param df: post qasrl annotation file with the following
    :return: dict, key:qasrl_id, value: list of list items, where each item represents a QA
    "qasrl_id":[ [ "qa_uuid","verb","question","answer","answer_range"]]
    '''

    grouped = df.groupby("qasrl_id")

    qaid2qas_d = {}
    for i, g in grouped:
        qa_zip = list(zip(g["qa_uuid"], g["verb"],g['verb_idx'],g["question"],
                          g["answer"],g['answer_range']))

        qa_zip.sort(key=lambda x: x[1])
        list_qas_d = []
        for item in qa_zip:
            list_qas_d.append({"qa_uuid": item[0], "verb":item[1], "verb_idx":item[2], "question": item[3],
                     "answer": item[4], "answer_range":item[5]})
        qaid2qas_d[i] = list_qas_d

    return qaid2qas_d