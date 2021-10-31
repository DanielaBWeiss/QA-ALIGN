import pandas as pd
import json
import nltk
from nltk.stem import PorterStemmer
nltk.download('averaged_perceptron_tagger')
ps = PorterStemmer()



def get_pred_idx(question, verb):
    particles_df = pd.read_csv("particles.csv")
    new_question = question
    not_verbs = particles_df.particles.to_list()
    particle_verbs = ["have", "had", "has"]
    #checking if a sentence has an identifiable verb, beyond the first token
    stemmed_verb = ps.stem(verb)
    token_list = nltk.word_tokenize(question)
    for i,token in enumerate(token_list):
        if i == 0: continue
        stemmed_tok = ps.stem(token)
        if stemmed_tok.lower() == stemmed_verb.lower():
            return i
        else:
            if stemmed_verb in particle_verbs or verb in particle_verbs:
                if token.lower() in particle_verbs:
                    return i
            if token not in not_verbs:
                return i

    pos = nltk.pos_tag(token_list)
    #checking if a sentence has an identifiable verb, beyond the first token
    for token_pos in enumerate(pos):
        if "VB" in token_pos[1][1]:
            return i
    return -1
