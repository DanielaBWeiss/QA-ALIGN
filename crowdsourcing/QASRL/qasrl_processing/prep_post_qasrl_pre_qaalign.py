import pandas as pd
import qasrl_utils as utils
import argparse
import json


def replace_empty_prev_text(prev_text):
    if (pd.isna(prev_text)):
        return "------------------"
    if (not prev_text) or ("NA" in prev_text):
        return "------------------"
    else:
        return prev_text #json.dumps(prev_text)

def get_qas(qas, qasrl_id):
    if qasrl_id in qas:
        return qas[qasrl_id]
    return []

def get_tokens(qasrl,row):
    df = qasrl[qasrl.qasrl_id == row['abs_sent_id']]
    if len(df) == 0:
        return row['text']
    else:
        return " ".join(df['tokens'].iloc[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--qasrl', type=str, required=True, help='file containing qasrl annotations')
    parser.add_argument('--qasrl_sents', type=str, required=True, help="sentences used to get qasrl annotations")
    parser.add_argument('--qa_align', type=str, required=True, help='file containing sentence pairs or original sentences that exist in the qasrl file')
    parser.add_argument('--output', type=str, required=True, help='Where to save the file ready for qa-align annotations')
    parser.add_argument('--save_qasrl', action='store_true', required=False, default=False)

    args = parser.parse_args()

    sents = pd.read_csv(args.qasrl_sents)
    split_qasrl = pd.read_csv(args.qasrl)
    if args.save_qasrl and 'qa_uuid' not in split_qasrl.columns:
        split_qasrl = utils.clean_qasrl_annots_post_gen_mturk(split_qasrl)
        split_qasrl = utils.split_argument_answers_post_mturk(split_qasrl)
        #split_qasrl['question_html'] = split_qasrl.apply(lambda x: utils.get_html_questions(x['question'], x['verb']), axis=1)
        split_qasrl['qa_uuid'] = split_qasrl.apply(lambda x:  x['qasrl_id'] + "~!~" + str(x['verb_idx']) + "~!~" + x['verb']+ "~!~"+ str(x['qa_uuid']), axis=1)
        split_qasrl.to_csv(args.qasrl, index=False)

    split_qasrl['tokens'] = split_qasrl['tokens'].apply(lambda x: eval(x))
    qaid2qas_d_json = utils.get_qas_per_qasrlid_parser(split_qasrl) #dict of qasrl_id to qas in that sentence

    sent_pairs = pd.read_csv(args.qa_align)
    sent_pairs['qas'] = sent_pairs.abs_sent_id.apply(lambda x:  get_qas(qaid2qas_d_json,x))
    sent_pairs['tokens'] = sent_pairs.apply(lambda x: get_tokens(split_qasrl,x),axis=1)

    '''
    sent_pairs['qas_1'] = sent_pairs['abs_sent_id_1'].apply(lambda x: sent2qas_d[x])
    sent_pairs['qas_2'] = sent_pairs['abs_sent_id_2'].apply(lambda x: sent2qas_d[x])
    sent_pairs['tokens_1'] = sent_pairs.abs_sent_id_1.apply(lambda x: sents[sents.qasrl_id == x]['tokens'].iloc[0])
    sent_pairs['tokens_2'] = sent_pairs.abs_sent_id_2.apply(lambda x: sents[sents.qasrl_id == x]['tokens'].iloc[0])

    sent1_htmls, sent2_htmls = utils.get_html_text(sent_pairs)
    sent_pairs['qas_2'] = sent_pairs['abs_sent_id_2'].apply(lambda x: json.dumps(qaid2qas_d_json[x]))
    sent_pairs['qas_1'] = sent_pairs['abs_sent_id_1'].apply(lambda x: json.dumps(qaid2qas_d_json[x]))

    sent_pairs['text_1_html'] = sent1_htmls #[json.dumps(x) for x in sent1_htmls]
    sent_pairs['text_2_html'] = sent2_htmls #[json.dumps(x) for x in sent2_htmls]
    sent_pairs['prev_text_1'] = sent_pairs['prev_text_1'].apply(lambda x: replace_empty_prev_text(x))
    sent_pairs['prev_text_2'] = sent_pairs['prev_text_2'].apply(lambda x: replace_empty_prev_text(x))
    sent_pairs.to_csv(args.output, index=False)

    '''
    sent_pairs.to_csv(args.output, index=False)