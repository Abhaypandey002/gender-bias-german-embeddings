from pathlib import Path
import gensim
from gensim.models import KeyedVectors

# def load_word2vec(path):
#     path = str(path)
#     # GermanWordEmbeddings' downloadable model is a native Gensim model.
#     return gensim.models.Word2Vec.load(path).wv

def load_word2vec(path):
    return KeyedVectors.load_word2vec_format(str(path), binary=True)

def load_keyed_vectors(path, binary=True):
    return KeyedVectors.load_word2vec_format(str(path), binary=binary)

def load_fasttext(path):
    # Native fastText .bin can be loaded by gensim's FastText loader.
    return gensim.models.fasttext.load_facebook_model(str(path)).wv
