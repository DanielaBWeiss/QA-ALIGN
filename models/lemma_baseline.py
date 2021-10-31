import re
from argparse import ArgumentParser
from collections import Counter
from typing import Tuple, Dict
from sklearn.metrics import classification_report, precision_recall_fscore_support

import pandas as pd
import spacy
from spacy.tokens import Doc

nlp = spacy.load('en_core_web_sm', disable=['ner'])


def lemmatize_answer(r, all_docs, suffix):
    span = r['answer_range' + suffix]
    start, end = [int(t) for t in span.split(":")]
    doc_id = r['abs_sent_id' + suffix]
    doc = all_docs[doc_id]
    lemmas = [doc[i].lemma_ for i in range(start, end)]
    return " ".join(lemmas)


def parse_span(span_: str):
    start, end = span_.split(":")
    start = int(start)
    end = int(end)
    return start, end


def calc_bag_of_words_iou(lemmas_1, lemmas_2):
    bag_1 = lemmas_1.split()
    bag_2 = lemmas_2.split()
    bag_1 = Counter(bag_1)
    bag_2 = Counter(bag_2)

    # This is not exactly set intersection over set union,
    # because every word is weighted by its frequency
    # whereas in a set all words are unique (deduplicated)
    inter_keys = list(bag_1.keys() & bag_2.keys())
    # this accounts for all joint words in BOTH spans
    inter_size = sum(bag_1[key] + bag_2[key] for key in inter_keys)
    # this accounts for all words used
    union_size = sum(bag_1.values()) + sum(bag_2.values())
    iou = float(inter_size) / union_size
    return iou


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def find_head(parsed_sentence: spacy.tokens.Doc, span: Tuple[int, int]):
    arg_span = parsed_sentence[span[0]:span[1]]
    # https://spacy.io/api/span#root
    # The token with the shortest path to the root of the sentence (or the root itself).
    # If multiple tokens are equally high in the tree, the first token is taken.
    arg_root = arg_span.root
    # if this is a preposition, try to find the object of the preposition this span is made of.
    arg_children = list(arg_root.children)
    if arg_root.dep_ == "prep" and len(arg_children) == 1:
        first_child = arg_children[0]
        if first_child.dep_ == "pobj":
            arg_root = first_child

    head_idx = arg_root.i
    return head_idx


def make_doc(text):
    tokens = text.split()
    doc = Doc(words=tokens, vocab=nlp.vocab)
    for name, pipe in nlp.pipeline:
        doc = pipe(doc)
    return doc


def get_all_docs(df: pd.DataFrame) -> Dict[str, spacy.tokens.Doc]:
    all_texts = {}
    for suffix in ["_1", "_2"]:
        doc_ids = df['abs_sent_id'+suffix].tolist()
        texts = df['text' + suffix].tolist()
        all_texts.update(dict(zip(doc_ids, texts)))
    docs = {doc_id: make_doc(text)
            for doc_id, text in all_texts.items()}
    return docs


def main(args):
    df = pd.read_csv(args.in_file)
    all_docs = get_all_docs(df)

    for suffix in ["_1", "_2"]:
        doc_ids = df['abs_sent_id' + suffix].tolist()
        verb_ids = df['verb_idx' + suffix].tolist()
        answer_ranges = [parse_span(span) for span in df['answer_range' + suffix]]
        docs = [all_docs[doc_id] for doc_id in doc_ids]
        pred_lemmas = [doc[verb_idx].lemma_.lower() for doc, verb_idx in zip(docs, verb_ids)]
        answer_heads = [find_head(doc, span) for doc, span in zip(docs, answer_ranges)]
        answer_lemmas = [doc[head_idx].lemma_.lower() for doc, head_idx in zip(docs, answer_heads)]
        df['pred_lemma' + suffix] = pred_lemmas
        df['answer_head_lemma' + suffix] = answer_lemmas

    df['pred_lemma_match'] = df.pred_lemma_1 == df.pred_lemma_2
    df['answer_head_lemma_match'] = df.answer_head_lemma_1 == df.answer_head_lemma_2
    df['predicted'] = (df.pred_lemma_match & df.answer_head_lemma_match).astype(int)
    # prec, recall, f1, support = precision_recall_fscore_support(df.label, df.predicted, average='binary', pos_label=1)
    print(classification_report(df.label, df.predicted))
    df.to_csv(args.out_file, index=False, encoding="utf-8")


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--in_file", help="/path/to/input_file.csv", default="../val_for_lemma.csv")
    ap.add_argument("--out_file", help="/path/to/output_file.csv", default="../val_for_lemma.out.csv")
    main(ap.parse_args())


