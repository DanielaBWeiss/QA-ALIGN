from argparse import ArgumentParser
import os
import pandas as pd
from collections import defaultdict
from sklearn.metrics import precision_recall_curve

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def find_threshold(pred, labels):
    '''

    :param gold: numpy array of gold labels (0|1)
    :param pred: numpy array of predictions (float)
    :return:
    '''
    best_thresh = 0
    best_f1 = 0
    precision, recall, thresholds = precision_recall_curve(labels, pred)
    for trio in zip(precision, recall, thresholds[:-1]):
        prec = trio[0]
        rec = trio[1]
        thresh = trio[2]
        f1 = (2 * prec * rec) / (prec + rec)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_f1, best_thresh


def calc_f1(preds, labels, threshold=None):
    TP = 0
    total_true = sum(labels)
    total_preds = 0
    for pred, gold in zip(preds, labels):
        if threshold:
            if pred > threshold:
                if gold == 1:
                    TP += 1
                total_preds += 1
        elif pred == 1 and gold == 1:
            TP += 1

    prec = 1.0 * TP / total_preds if total_preds != 0 else 0
    recall = 1.0 * TP / total_true if total_true != 0 else 0
    if (prec + recall) == 0:
        f1 = 0
    else:
        print(prec)
        print(recall)
        f1 = (2 * prec * recall) / (prec + recall)
    return f1, prec, recall


def main(args):
    '''
    For train: 107 were many2many alignments that were skipped that we have to add to the recall denominator.
    TP / 3055 + 107 = true recall
    For dev: 49
    TP / 1156 + 49 = 59 Recall
    for test: 65
    TP / 1443 + 65



    :param args:
    :return:
    '''
    pred_file = pd.read_csv(args.file_path)
    thresh = None
    labels = pred_file['gold'].to_numpy()
    pred = pred_file['pred'].to_numpy()
    if args.threshold:
        thresh = args.threshold
    if args.find_threshold:
        best_f1, thresh = find_threshold(pred, labels)
        print("best f1: ", best_f1)
        print("best threshold: ", thresh)

    print("Using threshold: {}".format(thresh))
    f1, prec, recall = calc_f1(pred, labels, thresh)
    print("Evaluating on {} samples".format(len(pred_file)))
    print("F1: {}".format(f1))
    print("Precision: {}".format(prec))
    print("Recall: {}".format(recall))



if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--file_path', type=str, required=True, help='directory containing preds')
    parser.add_argument("--save_dir", default="./inference")
    parser.add_argument("--output", default="pred_exp.csv", required=False)
    parser.add_argument("--threshold", default=None, type=float)
    parser.add_argument("--find_threshold",required=False, default=False, action='store_true')


    main(parser.parse_args())