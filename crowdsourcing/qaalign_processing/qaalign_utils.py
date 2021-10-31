import pandas as pd
import os
from collections import defaultdict
import argparse


def get_sorted_key(x):
    key = [x['abs_sent_id_1'],x['abs_sent_id_2']]
    key = sorted(key)
    return "~!~".join(key)

def get_unique_edge_keys(align):
    sent_1_keys = [qa[0] for qa in align[
        'sent1']]  # ['TAC2011~!~D1126-B-AEFH~!~26~!~1~!~faces~!~1030','What does someone face? - up to seven years in prison','faces']
    sent_2_keys = [qa[0] for qa in align['sent2']]
    edge_keys = sent_1_keys + sent_2_keys
    sorted_edge_keys = sorted(edge_keys)
    return "|||".join(sorted_edge_keys)

def compare_two_annotations(predictions):
    hit_results = defaultdict(dict)
    for i ,df in predictions.groupby("key"):
        if i == 'DUC2006~!~D0601~!~26~!~DUC2006~!~D0601~!~69':
            w1 = df.iloc[0]
            w2 = df.iloc[1]
            w1_pred_edges = set()
            for align in w1['answers']:  # going over a single alignment in a pair of sentences)
                pred_keys_for_alignment = get_unique_edge_keys(align)
                w1_pred_edges.add(pred_keys_for_alignment)

            w2_pred_edges = set()
            for align in w2['answers']:  # going over a single alignment in a pair of sentences)
                pred_keys_for_alignment = get_unique_edge_keys(align)
                w2_pred_edges.add(pred_keys_for_alignment)

            print("w1 : ", w1_pred_edges)
            print("w2 : ", w2_pred_edges)
            inter = w1_pred_edges.intersection(w2_pred_edges)
            print("intersection: ", inter)
            print("diss : ", w1_pred_edges.union(w2_pred_edges) - inter)

            break