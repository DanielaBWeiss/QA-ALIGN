import pandas as pd
from collections import defaultdict
import argparse
import eval_utils as utils
import itertools

def calc_workers_agreement(workers, edges=False, soft_match=False):
    '''

    :param workers:
    :return:
    '''
    workers2F1 = {}
    it = iter(workers.groupby(['key', 'worker_id']).count().index)
    workers2keys = defaultdict(list)
    for x in it:
        pair_workers = [x[1], next(it)[1]]
        workers2keys["|||".join(sorted(pair_workers))].append(x[0])

    for w_pair,hit_keys in workers2keys.items():
        #let wrk1 be gold
        wrk1_id,wrk2_id = w_pair.split("|||")
        total_alignment_predictions = 0
        total_correct_aligments = 0
        total_gold_predictions = 0  # since each worker might do a different number of alignments

        wrk1 = workers[(workers.worker_id == wrk1_id) & (workers.key.isin(hit_keys))]
        wrk2 = workers[(workers.worker_id == wrk2_id) & (workers.key.isin(hit_keys))]
        for i, row in wrk2.iterrows():

            curr_gold_keys = wrk1[wrk1.key == row['key']]['gold_keys'] #current HIT's gold keys
            #if len(curr_gold_keys) == 0:continue
            curr_gold_keys = curr_gold_keys.iloc[0]
            if not curr_gold_keys['unique_edges'] and not row['alignments']:
                continue

            if not curr_gold_keys['unique_edges'] and not row['alignments']:
                continue

            correct_aligments, alignment_predictions, gold_predictions, instance_feedback, \
            instance_pairs2performance, partial_matches = utils.calc_alignment_iou(row['alignments'], curr_gold_keys, edges, soft_match)



            total_correct_aligments += correct_aligments
            total_alignment_predictions += alignment_predictions
            total_gold_predictions += gold_predictions

        if total_alignment_predictions == 0:
            prec = 0
        else:
            prec = total_correct_aligments / total_alignment_predictions

        recall = total_correct_aligments / total_gold_predictions

        if (prec + recall) == 0:
            f1 = 0
        else:
            f1 = 2 * (prec * recall) / (prec + recall)

        sorted_wrk_key = "|||".join(sorted([wrk1_id,wrk2_id]))
        if sorted_wrk_key in workers2F1: print(round(f1, 2), workers2F1[sorted_wrk_key])
        workers2F1[sorted_wrk_key] = round(f1, 2)

    return workers2F1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=str, required=True,
                        help='csv file containing qa alignment annotations by workers')
    parser.add_argument('--soft_match', action="store_true", required=False, default=False,
                        help='evaluate soft alignment matches in addition to exact match')
    parser.add_argument('--edge_level', action="store_true", required=False, default=False,
                        help='evaluate at the edge level in a bi-partite graph, rather than alignment level')

    args = parser.parse_args()
    workers = pd.read_csv(args.workers)
    workers['alignments'] = workers['alignments'].apply(lambda x: eval(x))
    if args.edge_level:
        workers['gold_keys'] = workers['alignments'].apply(lambda x: utils.get_gold_instance_edges(x)) #edge level
    else: workers['gold_keys'] = workers['alignments'].apply(lambda x: utils.get_gold_instance_keys(x)) #edge level
    workers_F1_IAA = calc_workers_agreement(workers, args.edge_level, args.soft_match)

    #print F1 agreements between all workers
    print(pd.DataFrame(workers_F1_IAA.items(), columns=['key', 'F1']))
    print("Average F1: ", sum(workers_F1_IAA.values())/len(workers_F1_IAA))