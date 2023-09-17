import re
import pandas as pd
pd.options.mode.chained_assignment = None
import nltk
from nltk.tokenize import ToktokTokenizer
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import gensim.downloader as api
import numpy as np

def generalCleaning(text):
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = text.lower()
    return text

def specificalCleaning(tokens):
    #nltk.download("stopwords")
    sw = set(stopwords.words("english"))
    onomatopeias = ['oh', 'ooh', 'ah', 'uh', 'eh', 'na', 'yeah', 'la', 'mmm', 'mm', 'hm', 'whoa']
    return [token for token in tokens if token not in sw and token not in onomatopeias and not token.isdigit()]

def lemmatize(tokens):
    #nltk.download('wordnet')
    #nltk.download('omw-1.4')
    wnl = WordNetLemmatizer()
    return [wnl.lemmatize(token) for token in tokens]

df = pd.read_csv("taylor-songs.csv")
dfLyrics = df[["Lyrics"]]

# 1.- REMOVE SPECIAL CHARACTERS, REMOVE BLANK SPACES, TURN INTO LOWERCASE...
dfLyrics["Tokens"] = df.Lyrics.apply(generalCleaning)

# 2.- TOKENIZE
tokenizer = ToktokTokenizer()
dfLyrics["Tokens"] = dfLyrics.Tokens.apply(tokenizer.tokenize)

# 3.- REMOVE STOPWORDS, DIGITS AND ONOMATOPEIAS
dfLyrics["Tokens"] = dfLyrics.Tokens.apply(specificalCleaning)

# 4.- LEMMATIZE
dfLyrics["Tokens"] = dfLyrics.Tokens.apply(lemmatize)

# 5.- USE WORDEMBEDDINGS FOR TEXT REPRESENTATION
model = api.load('word2vec-google-news-300')
words = set(model.index_to_key)
dfLyrics['New_Input_vect'] = np.array([np.array([model[i] for i in ls if i in words]) for ls in dfLyrics['Tokens']])
text_vect_avg = []
for v in dfLyrics['New_Input_vect']:
    if v.size:
        text_vect_avg.append(v.mean(axis=0))
    else:
        text_vect_avg.append(np.zeros(300, dtype=float))
dfLyrics['Lyrics_Embeddings'] = text_vect_avg
dfLyrics.to_csv("taylor-songs(WE).csv")