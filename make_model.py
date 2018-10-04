import pickle
import time
from gensim.models.wrappers.dtmmodel import DtmModel
from gensim import corpora
start_time = time.time()

dtm_path = "dtm-linux64"

# Importation de la liste de texte lemmatisé
corpus = pickle.load(open('corpus_geo.pkl', 'rb'))

# Mise en format pour gensim
dictionary = corpora.Dictionary(corpus)
corpus = [dictionary.doc2bow(text) for text in corpus]

# Pour 10 topics
time_slice = [11468]*9
time_slice.append(11472)

# Pour 20 topics
#time_slice = [5734]*9
# time_slice.append(5738)

nb_topics = 10

model = DtmModel(dtm_path, corpus, time_slice, num_topics=nb_topics,
                 id2word=dictionary, initialize_lda=True)

model.save("DTMModel")

print("---- %s seconds ----" % (time.time() - start_time))
