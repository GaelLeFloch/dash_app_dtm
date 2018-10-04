import pandas as pd
import pickle
from spacy.lang.fr import French
from spacy.lang.fr import STOP_WORDS
# Ajout de mots dans la liste de stop word original
SW = ["pouvoir", "entrer", "partir", "faire", "grand", "dan"]
[STOP_WORDS.add(w) for w in SW]

data_all = pd.read_csv(
    "~/Documents/Documents/Stages/ERIC/NewsBrowserPack/CORPUS/femmes/data_geo.csv", sep="\t")

documents = data_all["description"]

parser = French()


def lemmatize(t):
    """
    Découpe une chaine de charachtère en mots puis prend leur forme canonique.
    Et ne garde pas les mots de la liste des STOPS_WORDS ainsi que les mots inférieur à 3 lettres.
    """
    t = parser(t.lower())
    t = [x for x in t if x.is_alpha]
    t = [x.text if (x.text == "plus" or x.text == "mais")
         else x.lemma_ for x in t]
    t = [x for x in t if ((x not in STOP_WORDS) & (len(x) > 2))]
    return t


# Prend environ 20 moin à tut traiter
i = 0
corpus = []
for document in documents:
    if (not i % 100):
        print(i)
    i += 1
    sublist = []
    for word in lemmatize(document):
        sublist.append(word)
    corpus.append(sublist)

# corpus est une liste de liste contenant des mots lemmatisé
pickle.dump(corpus, open('corpus_geo.pkl', 'wb'))
