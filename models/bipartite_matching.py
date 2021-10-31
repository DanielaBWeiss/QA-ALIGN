""" 
max-bipartite matching for decoding the QA-Alignment model predictions.
"""
from typing import List, Tuple, Dict
import networkx as nx
from argparse import ArgumentParser

# custom types
#QA = Tuple[str, str]  # (question, answer); you can change that to any primitive "hashable" type
QA_ALIGNMENT = Tuple[str, str] # currently supporting one-to-one alignment only

def max_bipartite_match(alignment_probs: List[Tuple[QA_ALIGNMENT, float]],
                        probability_threshold: float = None) -> Dict[QA_ALIGNMENT, float]:
    """ 
    :arg alignment_probs: probability score for every alignment
    :arg probability_threshold (optional): qa-alignments with probability lower than this threshold
     would not be part of the final matching.
    :returns: dict of most probably complete alignment,
     i.e. QA-alignments within maximal-weight bipartite matching (along with their probability). 
    """
    if not probability_threshold:
        probability_threshold = 0.0
    # construct bipartite weighted graph
    pairs2probs = {}
    sent_1_QAs = set()
    sent_2_QAs = set()
    for alignment,p in alignment_probs:
        sent_1_QAs.add(alignment[0])
        sent_2_QAs.add(alignment[1])
        pairs2probs[alignment[0]+"|"+alignment[1]] = p

    g = nx.Graph()
    g.add_nodes_from(sent_1_QAs, bipartite=0)
    g.add_nodes_from(sent_2_QAs, bipartite=1)
    for (qa1, qa2), prob in alignment_probs: 
        weight = prob if prob > probability_threshold else -1
        g.add_edge(qa1, qa2, weight=weight)
    
    match = nx.max_weight_matching(g)
    # get all pairs in right order
    match = [(qa1, qa2) if qa1 in sent_1_QAs else (qa2, qa1) 
             for qa1, qa2 in match]

    match_with_prob = {qa_pair[0]+"|"+qa_pair[1]: pairs2probs[qa_pair[0]+"|"+qa_pair[1]] for qa_pair in match}
    return match_with_prob


if __name__ == "__main__":

    l = [(('f','h'),0.1),(('a','b'),0.5), (('a','c'), 0.4), (('d', 'c'), 0.3), (('d','b'),0.1)]
    max_bipartite_match(l)