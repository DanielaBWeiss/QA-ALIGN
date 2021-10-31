from collections import defaultdict
import itertools

def get_unique_align_keys(align):
    '''
    For evaluation at the alignment level
    :param align: predicted or gold alignment
    :return: all edges in a bi-partite graph (each qa to qa in an alignment is an edge)
    '''
    sent_1_keys = [qa[0] for qa in align[
        'sent1']]  # ['TAC2011~!~D1126-B-AEFH~!~26~!~1~!~faces~!~1030','What does someone face? - up to seven years in prison','faces']
    sent_2_keys = [qa[0] for qa in align['sent2']]
    align_keys = sent_1_keys + sent_2_keys
    sorted_edge_keys = sorted(align_keys)
    return "|||".join(sorted_edge_keys)

def get_unique_sent_keys(sent_qas):
    '''
    For evaluation at the alignment level
    :param align: predicted or gold alignment
    :return: all edges in a bi-partite graph (each qa to qa in an alignment is an edge)
    '''
    sent_keys = [qa[0] for qa in sent_qas] # ['TAC2011~!~D1126-B-AEFH~!~26~!~1~!~faces~!~1030','What does someone face? - up to seven years in prison','faces']
    sorted_edge_keys = sorted(sent_keys)
    return "|||".join(sorted_edge_keys)

def get_unique_edges(align):
    '''
    For evaluation at the edge level
    :param align: predicted or gold alignment
    :return: all edges in a bi-partite graph (each qa to qa in an alignment is an edge)
    '''
    sent_1_keys = [qa[0] for qa in align['sent1']]
    sent_2_keys = [qa[0] for qa in align['sent2']]
    unique_edges = list(itertools.product(sent_1_keys, sent_2_keys))
    unique_edges = set("|||".join(sorted([item[0], item[1]])) for item in unique_edges)
    return unique_edges

def get_unique_edge_dict(align):
    '''

    :param align: a single alignment
    :return: dictionary with each edge as a key, and an alignment dict for that edge
    '''
    bipartite_edges = list(itertools.product(align['sent1'], align['sent2']))
    unique_edges = {}

    for pair in bipartite_edges:
        pair_d = {'key_1':'', 'q1':[], 'a1': [], 'key_2':'', 'q2':[],'a2':[]}
        pair_d['key_1'] = pair[0][0]
        qa_splits = pair[0][1].split(" - ")
        pair_d['q1'].append(qa_splits[0])
        if len(qa_splits) > 2:
            pair_d['a1'].append(" - ".join(qa_splits[1:]))
        else:
            pair_d['a1'].append(qa_splits[1])

        pair_d['key_2'] = pair[1][0]
        qa_splits = pair[1][1].split(" - ")
        pair_d['q2'].append(qa_splits[0])
        if len(qa_splits) > 2:
            pair_d['a2'].append(" - ".join(qa_splits[1:]))
        else:
            pair_d['a2'].append(qa_splits[1])

        sorted_key = "|||".join(sorted([pair_d['key_1'], pair_d['key_2']]))
        unique_edges[sorted_key] = pair_d

    return unique_edges

def get_node2align_dict(align):
    '''
    This function creates a mapping between one full side of an alignmnt (set of QA), to all the questions from the other side of an alignment.
    This allows us to give half a point for someone choosing all the right questions, except one answer or a set of answers might be different.
    :param align: n alignment
    :return: a dict with unique keys for each set of QAs per sentence, aligned to Qs from the other sentence in this alignment.

    '''

    pair_d = {'key_1':[], 'q1':[], 'a1': [], 'key_2':[], 'q2':[],'a2':[]}
    for qa in align['sent1']:
        pair_d['key_1'].append(qa[0])
        qa_splits = qa[1].split(" - ")
        pair_d['q1'].append(qa_splits[0])
        if len(qa_splits) > 2:
            pair_d['a1'].append(" - ".join(qa_splits[1:]))
        else:
            pair_d['a1'].append(qa_splits[1])

    for qa in align['sent2']:
        pair_d['key_2'].append(qa[0])
        qa_splits = qa[1].split(" - ")
        pair_d['q2'].append(qa_splits[0])
        if len(qa_splits) > 2:
            pair_d['a2'].append(" - ".join(qa_splits[1:]))
        else:
            pair_d['a2'].append(qa_splits[1])

    return pair_d

def get_gold_instance_edges(annotations):
    '''
    This function creates unique key mappings for gold annotations for evaluation purposes at the alignment level
    :param annotations: gold annotations
    :return: prep dictionaries for evaluation purposes
    '''
    edge2key = defaultdict()
    key2align = defaultdict()
    align_to_annots = {"unique_edges": {}}
    for align in annotations:
        align_edges = get_unique_edge_dict(align)
        sorted_key = get_unique_align_keys(align)
        for edge,_ in align_edges.items():
            edge2key[edge] = sorted_key
        key2align[sorted_key] = align
        align_to_annots["unique_edges"].update(align_edges)

    align_to_annots["key2align"] = key2align
    align_to_annots["edge2key"] = edge2key
    return align_to_annots

def get_gold_instance_keys(annotations):
    '''
    This function creates unique edge mappings for gold annotations for evaluation purposes at the edge level.
    Instead of evaluating per alignment (set to set), we create a bi-partite graph out of all alignments in a HIT.
    :param annotations: gold annotations
    :return: prep dictionaries for evaluation purposes
    '''
    key2align = defaultdict()
    align_to_annots = {"unique_edges": {}}
    for align in annotations:
        sorted_key = get_unique_align_keys(align)
        key2align[sorted_key] = align
        pair_d = get_node2align_dict(align)
        align_to_annots["unique_edges"].update({sorted_key:pair_d})

    align_to_annots["key2align"] = key2align
    return align_to_annots


def get_gold_key_edges(gold, edges=False):
    gold_keys = {}
    for i, row in gold.iterrows():
        if edges:
            gold_keys[row['key']] = get_gold_instance_edges(row['answers'])
        else: gold_keys[row['key']] = get_gold_instance_keys(row['answers'])
    return gold_keys

def prep_alignment(align):
    '''
    Prepping alignments in text format for eval word docx
    :param align: a single alignment
    :return:
    '''
    qas_1 = [qa[1] for qa in align['sent1']]
    qas_2 = [qa[1] for qa in align['sent2']]
    pretty_align = {}
    pretty_align['sent1'] = qas_1
    pretty_align['sent2'] = qas_2
    return pretty_align


def calc_alignment_iou(instance_pred, instance_gold_keys, edge_eval, soft_match):
    '''
    :param instance_pred: a single prediction for a pair of sentences, i.e a list of alignments (row['answers'])
    :param gold_keys: dictionary containing gold alignments (gold_keys[row.key]), 'unique_edges', and 'node2edge/key'
    :return: iou eval
    '''


    instance_feedback = {"prec_err": [], "recall_err": [], "correct_aligns": [], "feedback":'{}'}
    partial_matches = []
    total_alignment_predictions = 0
    total_correct_aligments = 0
    total_gold_predictions = 0  # since each worker might do a different number of alignments
    pairs2performance = 0

    if edge_eval:
        pred_keys = {"unique_edges": {}} if not instance_pred else get_gold_instance_edges(instance_pred)
    else:
        pred_keys = {"unique_edges": {}} if not instance_pred else get_gold_instance_keys(instance_pred)
    if (not instance_gold_keys['unique_edges']) and (not pred_keys['unique_edges']): #precision errors
        for align in pred_keys['key2align'].items():
            instance_feedback['prec_err'].append(prep_alignment(align))
        total_alignment_predictions = len(pred_keys['unique_edges'])
        return total_correct_aligments, total_alignment_predictions, total_gold_predictions, \
               instance_feedback, pairs2performance, partial_matches

    if (not pred_keys['unique_edges']) and instance_gold_keys['unique_edges']: #recall errors
        for _, align in instance_gold_keys['key2align'].items():
            instance_feedback['recall_err'].append(prep_alignment(align))

        total_gold_predictions = len(instance_gold_keys['unique_edges'])
        return total_correct_aligments, total_alignment_predictions, total_gold_predictions, \
               instance_feedback, pairs2performance, partial_matches

    total_alignment_predictions = len(pred_keys['unique_edges'])
    total_gold_predictions = len(instance_gold_keys['unique_edges'])
    pred_edges = set()
    prec_errors = []
    recall_errors = []
    prec_correct = []
    soft_match_score = 0
    matched_gold_keys = {}

    for pred_align_key,align_val in pred_keys['unique_edges'].items():  # going over every alignment predicted for a pair of sentences

        pred_edges.add(pred_align_key)

        # first check exact match
        current_alignment = None
        if edge_eval:
                align_key = pred_keys['edge2key'][pred_align_key]
                current_alignment = pred_keys['key2align'][align_key]
        else: current_alignment = pred_keys['key2align'][pred_align_key]

        if pred_align_key in instance_gold_keys['unique_edges'].keys():
            if pred_align_key in matched_gold_keys:
                if matched_gold_keys[pred_align_key] == 0.5:
                    total_correct_aligments -= 0.5
                    soft_match_score -= 0.5
            prec_correct.append(prep_alignment(current_alignment))
            total_correct_aligments += 1
            matched_gold_keys[pred_align_key] = 1
        else:
            prec_errors.append(prep_alignment(current_alignment))
            if soft_match:
                # need to check if we at least have a softmatch
                # check for a complete match of at least one side of an alignment, and soft match the other
                #align_val = {'key_1':[], 'q1':[], 'a1': [], 'key_2':[], 'q2':[],'a2':[]}
                for gold_key, gold_alignment in instance_gold_keys['unique_edges'].items():
                    if gold_key in matched_gold_keys: continue

                    #Doing Labeled Argument, if both sides of the alignment the questions match
                    if sorted(align_val['q1']) == sorted(gold_alignment['q1']):
                        #if sorted(align_val['key_1']) == sorted(gold_alignment['key_1']): #great one side matches
                        pred_qs = set(align_val['q2'])
                        gold_qs = set(gold_alignment['q2'])
                        intersection_qs = pred_qs.intersection(gold_qs)

                        if len(intersection_qs) != 0:
                            matched_gold_keys[gold_key] = 0.5
                            total_correct_aligments += 0.5
                            soft_match_score += 0.5
                            if edge_eval:
                                align_key = instance_gold_keys['edge2key'][gold_key]
                                gold_alignment = instance_gold_keys['key2align'][align_key]
                            else: gold_alignment = instance_gold_keys['key2align'][gold_key]
                            partial_matches.append((current_alignment, gold_alignment))
                            break

                    elif sorted(align_val['q2']) == sorted(gold_alignment['q2']):
                        pred_qs = set(align_val['q1'])
                        gold_qs = set(gold_alignment['q1'])
                        intersection_qs = pred_qs.intersection(gold_qs)
                        if len(intersection_qs) != 0:
                            matched_gold_keys[gold_key] = 0.5
                            total_correct_aligments += 0.5
                            soft_match_score += 0.5
                            if edge_eval:
                                align_key = instance_gold_keys['edge2key'][gold_key]
                                gold_alignment = instance_gold_keys['key2align'][align_key]
                            else:
                                gold_alignment = instance_gold_keys['key2align'][gold_key]
                            partial_matches.append((current_alignment, gold_alignment))
                            break


    edges_iou = len(prec_correct) + soft_match_score
    edges_union = pred_edges.union(instance_gold_keys['unique_edges'].keys())
    edges_iou = edges_iou / (len(edges_union) - soft_match_score)

    recalls = instance_gold_keys['unique_edges'].keys() - pred_edges
    for recall in recalls:
        if edge_eval:
            align_key = instance_gold_keys['edge2key'][recall]
            missed_alignment = instance_gold_keys['key2align'][align_key]
        else:
            missed_alignment = instance_gold_keys['key2align'][recall]
        recall_errors.append(prep_alignment(missed_alignment))

    if prec_errors or recall_errors:
        instance_feedback["correct_aligns"] = prec_correct
        instance_feedback["prec_err"] = prec_errors
        instance_feedback["recall_err"] = recall_errors
    else:
        instance_feedback["correct_aligns"] = prec_correct

    pairs2performance = edges_iou

    return total_correct_aligments, total_alignment_predictions, total_gold_predictions, \
           instance_feedback, pairs2performance, partial_matches

