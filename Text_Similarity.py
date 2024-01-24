import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import string
from nltk.stem import WordNetLemmatizer  # For Lemmetization of words
from nltk.corpus import stopwords  # Load list of stopwords
from nltk import word_tokenize # Convert paragraph in tokens
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df=pd.read_csv('DataNeuron_Text_Similarity.csv')

df.shape 
df.isnull().sum()

df1=df.copy()

stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = text.lower()
    text = ''.join([word for word in text if word not in string.punctuation])
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df1['text1'] = df1['text1'].apply(preprocess)
df1['text2'] = df1['text2'].apply(preprocess)


def word_tokenizer(text):
    #tokenizes and stems the text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer() 
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

for i in tqdm(df1.index):
    s1=df1['text1'][i]
    s2=df1['text2'][i]
    s1words = ' '.join(word_tokenizer(s1))
    s2words = ' '.join(word_tokenizer(s2))
    df1['text1'][i]=s1words
    df1['text2'][i]=s2words


tfidf=TfidfVectorizer()

def calculate_similarity(row):
    tfidf_matrix = tfidf.fit_transform(row[['text1', 'text2']].values.astype('U'))
    cos_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return cos_sim[0][0]
 
# Apply the calculate_similarity function to create a new 'similarity' column
df1['similarity'] = df1.apply(calculate_similarity, axis=1)

df1.head()




