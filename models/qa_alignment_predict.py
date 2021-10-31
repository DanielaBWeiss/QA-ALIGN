import types
from argparse import ArgumentParser
import os
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers


from qa_alignment_dataset import QAAlignmentDataset
from qa_alignment_model import QAAlignmentModule

def load_file(file_path):
    data = pd.read_csv(file_path)
    data = data.to_dict(orient='records')
    return data

def main(args):
    data = load_file(args.file_path)
    print("Number of instances to predict on: ",len(data))
    tok_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ['[A]', '[/A]', '[P]', '[/P]', '[Q]']})
    dataset = QAAlignmentDataset(data, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    align_model = QAAlignmentModule(model)
    if 'roberta' in args.exp_name or 'coref' in args.exp_name:
        print("loading roberta loader")
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_samples_roberta)
    else:
        print("loading bert loader")
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_samples)

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_num = args.gpu
    device = torch.device(gpu_num)
    align_model.to(device)

    all_preds = []
    if 'roberta' in args.exp_name or 'coref' in args.exp_name:
        print("infering using roberta")
        for batch in tqdm(data_loader, desc="Predicting..."):
            preds = align_model.infer_roberta(batch, device)
            all_preds.extend(preds)
    else:
        print("infering using bert")
        for batch in tqdm(data_loader, desc="Predicting..."):
            preds = align_model.infer(batch,device)
            all_preds.extend(preds)
    print("Length of all predictions: ",len(all_preds))
    print()
    print("Finding max bipartite matching...")
    print()
    all_labels, preds2maximalprob = align_model.get_max_bipartite_matching(all_preds, dataset, predict=True,fusion=args.fusion)
    if args.eval:
        f1, prec, recall, total_preds, total_true = align_model.calc_bipartite_f1(all_labels, preds2maximalprob)
        print("Evaluating on predicted: ")
        print("F1: " ,f1)
        print("Prec: ", prec)
        print("Recall: ", recall)
        print("Total preds: ", total_preds)
        print("Total gold: ", total_true)
    results = []
    print("Len of all labels: ",len(all_labels))
    for pair_key, instance in tqdm(all_labels.items(), "Saving probs"):
        prob = preds2maximalprob[pair_key] if pair_key in preds2maximalprob else 0
        if args.threshold:
            prob = 1 if prob > args.threshold else 0
        #key, input_1, input_2, qa_uuidd_1, #qa_uuid_2, abs_sent-id_1, #gold, #pred
        if args.eval:
            results.append((instance[0], instance[1], instance[2], instance[3], instance[4], instance[5], prob))
        else:
            results.append((instance[0], instance[1], instance[2], instance[3], instance[4], instance[5], instance[6],prob))

    print("len of final results: ",len(results))
    if args.eval:
        #df = pd.DataFrame(results, columns=["key", "input_1", "input_2","abs_sent_id_1","abs_sent_id_2", "qa_uuid_1", "qa_uuid_2", "gold", "pred"])
        df = pd.DataFrame(results, columns=["key", "input_1", "input_2", "qa_uuid_1",
                                            "qa_uuid_2", "gold", "pred"])
    else:
        #df = pd.DataFrame(results, columns=["key", "input_1", "input_2","abs_sent_id_1","abs_sent_id_2", "qa_uuid_1", "qa_uuid_2", "pred"])
        df = pd.DataFrame(results, columns=["key", "input_1", "input_2", "abs_sent_id_1", "abs_sent_id_2", "qa_uuid_1",
                                            "qa_uuid_2", "pred"])

    if args.threshold:
        print("Total number of predicted alignments: ", len(df[df.pred > args.threshold]))
        print()
    else:
        print("Total number of predicted alignments: ",len(df[df.pred == 1]))
        print()
        print("Saving prediction")
    df.to_csv(args.save_dir + "/"+args.output, index=False)




if __name__ == "__main__":
    '''
    If only predicting on new data, pass arg threshold.
    Else if predicting without evaluating but want probability, no need to pass --eval or --threshold
    If want to evaluate on the same passed data and it has gold, pass --eval.
    '''


    parser = ArgumentParser()

    parser.add_argument('--file_path', type=str, required=True, help='directory containing train,dev, and test files')
    parser.add_argument("--save_dir", default="./qa-align-evaluations")
    parser.add_argument("--output", default="pred_exp.csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--exp_name", required=False, default='bert')
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--tokenizer",required=False)
    parser.add_argument("--threshold", required=False, type=float, default=None)
    parser.add_argument("--eval", action="store_true", required=False)
    parser.add_argument("--fusion", action="store_true", required=False, default=False)
    parser.add_argument("--batch_size", default=16, type=int)

    ap = pl.Trainer.add_argparse_args(parser)
    main(ap.parse_args())