import pandas as pd
import argparse
from collections import defaultdict
import eval_utils as utils
import qasrl_utils as qa_utils
import json

def get_conflicts(wrk1_ans, wrk2_ans):
    uniq_key_1 = defaultdict(list)
    uniq_key_2 = defaultdict(list)
    for align in wrk1_ans:  # get uniq keys for each alignment made
        uniq_key_1[utils.get_unique_align_keys(align)] = align

    for align in wrk2_ans:
        uniq_key_2[utils.get_unique_align_keys(align)] = align

    agreements = []
    diss_a = []  # find agreements, and delete in worker2 if found
    diss_b = []
    cnta = 0
    cntb = 0
    QAs_in_Agreement = {}
    for key, val in uniq_key_1.items():
        sent1_keys = utils.get_unique_sent_keys(val['sent1'])
        sent2_keys = utils.get_unique_sent_keys(val['sent2'])
        if key in uniq_key_2:
            aligned_val = val.copy()
            aligned_val['index'] = cnta
            cnta += 1
            QAs_in_Agreement[sent1_keys] = aligned_val
            QAs_in_Agreement[sent2_keys] = aligned_val
            agreements.append(aligned_val)
            del uniq_key_2[key]
        else:
            if (sent1_keys in QAs_in_Agreement) or (sent2_keys in QAs_in_Agreement):
                print("shoulnd't happen")
                continue
            aligned_val = val.copy()
            aligned_val['index'] = cnta
            cnta += 1
            diss_a.append(aligned_val)

    for key, val in uniq_key_2.items():  # add left over alignments from wrk2 to disagreements
        sent1_keys = utils.get_unique_sent_keys(val['sent1'])
        sent2_keys = utils.get_unique_sent_keys(val['sent2'])
        if (sent1_keys in QAs_in_Agreement) or (sent2_keys in QAs_in_Agreement):
            print("shoulnd't happen")
            continue
        aligned_val = val.copy()
        aligned_val['index'] = cntb
        cntb += 1
        diss_b.append(aligned_val)

    return agreements, diss_a, diss_b

def get_html_questions(answers):

    for align in answers:
        new_sent1 = []
        for qa in align['sent1']:
            answer = qa[1].split("-")[1:]
            html_qa = qa_utils.get_html_questions(qa[1].split("-")[0], qa[2])
            new_sent1.append([qa[0]," - ".join([html_qa," - ".join(answer)]),qa[2],qa[1]])
        new_sent2 = []
        for qa in align['sent2']:
            answer = qa[1].split("-")[1:]
            html_qa = qa_utils.get_html_questions(qa[1].split("-")[0], qa[2])
            new_sent2.append([qa[0], " - ".join([html_qa," - ".join(answer)]), qa[2],qa[1]])
        align['sent1'] = new_sent1
        align['sent2'] = new_sent2
    return answers

def prepare_consolidation(prod):
    consolidation = []
    agreed_on = []
    for i, df in prod.groupby(['key']):

        new_row = [i, df['abs_sent_id_1'].iloc[0], df['text_1'].iloc[0],df['text_1_html'].iloc[0], df['prev_text_1'].iloc[0],df['abs_sent_id_2'].iloc[0],
                   df['text_2'].iloc[0],df['text_2_html'].iloc[0],df['prev_text_2'].iloc[0], df['worker_id'].iloc[0], df['worker_id'].iloc[1]]
        # key, abs_sent_id_1,text_1,abs_sent_id_2, text_2, disagreements, agreements,worker_id_1,worker_id_2
        if (df['answers'].iloc[0] == {}) & (df['answers'].iloc[1] == {}):
            agreed_on.append(i)
            continue
        agree, diss_a, diss_b = get_conflicts(df['answers'].iloc[0], df['answers'].iloc[1])
        if (not diss_a) and (not diss_b):
            agreed_on.append(i)
            continue
        new_row.append(agree)
        new_row.append(diss_a)
        new_row.append(diss_b)
        consolidation.append(new_row)

    for_consol = pd.DataFrame(consolidation, columns=["key", "abs_sent_id_1","text_1","text_1_html","prev_text_1","abs_sent_id_2","text_2", "text_2_html",
                                                     "prev_text_2", "worker_id_1","worker_id_2","aligned_qas", "disagreements_a",
                                                      "disagreements_b"])
    for_consol['aligned_qas'] = for_consol['aligned_qas'].apply(lambda x: get_html_questions(x))
    for_consol['disagreements_a'] = for_consol['disagreements_a'].apply(lambda x:get_html_questions(x))
    for_consol['disagreements_b'] = for_consol['disagreements_b'].apply(lambda x: get_html_questions(x))

    agreed_prod = prod[prod.key.isin(agreed_on)].copy()

    agreed_prod.drop_duplicates("key", inplace=True)

    print("# agreed on: ",len(agreed_prod))
    print("# hits originally: ",len(prod.groupby("key")))
    print("# to consolidate: ",len(for_consol))
    return for_consol,agreed_prod


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='csv file containing qa alignment annotations')

    args = parser.parse_args()
    prod = pd.read_csv(args.input)
    prod['answers'] = prod['answers'].apply(lambda x: eval(x))

    for_consol,agreed_on = prepare_consolidation(prod)
    for_consol['aligned_qas'] = for_consol['aligned_qas'].apply(lambda x: json.dumps(x))
    for_consol['disagreements_a'] = for_consol['disagreements_a'].apply(lambda x: json.dumps(x))
    for_consol['disagreements_b'] = for_consol['disagreements_b'].apply(lambda x: json.dumps(x))
    new_input = ".."+args.input.strip(".csv")
    for_consol.to_csv(new_input+"_pre_consol.csv", index=False)
    agreed_on.to_csv(new_input+"_agreements.csv", index=False)