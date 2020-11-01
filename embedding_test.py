#pip install gensim==3.2.0

from gensim.models.keyedvectors import KeyedVectors
gensim_model = KeyedVectors.load_word2vec_format(
   r"C:\josh\ai\Pinokio\GoogleNews-vectors-negative300.bin", binary=True, limit=100000 )
print('I\'m =', gensim_model['I\'m'])