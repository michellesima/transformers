from gensim.models.keyedvectors import KeyedVectors
from utils import *
import sys

#glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)

if __name__ == '__main__':
    agenv = agen_verbs()
    sys.exit()
    verb = sys.argv[1]
    for cat, verbset in agenv.items():
        
        verb_simi = glove_model.most_similar_to_given(verb, verbset)
        print(verb_simi)
