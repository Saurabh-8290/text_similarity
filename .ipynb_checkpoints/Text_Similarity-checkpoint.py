#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import re
from tqdm import tqdm

from nltk.stem import WordNetLemmatizer  # For Lemmetization of words
from nltk.corpus import stopwords  # Load list of stopwords
from nltk import word_tokenize # Convert paragraph in tokens

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[4]:


df=pd.read_csv('Precily_Text_Similarity.csv')


# In[5]:


df.shape 


# In[6]:


df.isnull().sum()


# In[7]:


df1=df.copy()


# In[8]:


def preprocess_text(text_column):
    preprocessed_text = []
    for sentence in tqdm(text_column.values):
        sent = re.sub('[^A-Za-z0-9]+', ' ', sentence)
        sent = ' '.join(e for e in sent.split() if e not in stopwords.words('english'))
        preprocessed_text.append(sent.lower().strip())
    return preprocessed_text

# Example usage
df1['text1'] = preprocess_text(df1['text1'])
df1['text2'] = preprocess_text(df1['text2'])


# In[59]:


preprocessed_text1 = []

# tqdm is for printing the status bar

for sentance in tqdm(df['text1'].values):
    sent = re.sub('[^A-Za-z0-9]+', ' ', sentance)

    sent = ' '.join(e for e in sent.split() if e not in stopwords.words('english'))
    preprocessed_text1.append(sent.lower().strip())


# In[60]:


df1['text1'] = preprocessed_text1
df1.head(3)


# In[61]:


preprocessed_text2 = []

# tqdm is for printing the status bar

for sentance in tqdm(df['text2'].values):
    sent = re.sub('[^A-Za-z0-9]+', ' ', sentance)

    sent = ' '.join(e for e in sent.split() if e not in stopwords.words('english'))
    preprocessed_text2.append(sent.lower().strip())


# In[64]:


df1['text2'] = preprocessed_text2


# In[9]:


df1


# In[10]:


def word_tokenizer(text):
    #tokenizes and stems the text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer() 
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


# In[11]:


for i in tqdm(df1.index):
    s1=df1['text1'][i]
    s2=df1['text2'][i]
    s1words = ' '.join(word_tokenizer(s1))
    s2words = ' '.join(word_tokenizer(s2))
    df1['text1'][i]=s1words
    df1['text2'][i]=s2words


# In[12]:


tfidf=TfidfVectorizer()


# In[13]:


def calculate_similarity(row):
    tfidf_matrix = tfidf.fit_transform(row[['text1', 'text2']].values.astype('U'))
    cos_sim=cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return cos_sim[0][0]
df1['similarity'] = df1.apply(calculate_similarity, axis=1)


# In[16]:


df1['binary_score'] = (df1['similarity'] > 0.5).astype(int)


# In[ ]:





# In[ ]:


def preprocess_text(text):
    # Remove non-alphanumeric characters and convert to lowercase
    sent = re.sub('[^A-Za-z0-9]+', ' ', text)
    # Remove stopwords
    sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords.words('english'))
    return sent.lower().strip()

def calculate_cosine_similarity(text1, text2):
    # Preprocess the texts
    preprocessed_text1 = preprocess_text(text1)
    preprocessed_text2 = preprocess_text(text2)

    # Tokenize and lemmatize the texts
    lemmatizer = WordNetLemmatizer()
    tokens1 = [lemmatizer.lemmatize(t) for t in word_tokenize(preprocessed_text1)]
    tokens2 = [lemmatizer.lemmatize(t) for t in word_tokenize(preprocessed_text2)]

    # Convert the preprocessed texts to strings
    s1words = ' '.join(tokens1)
    s2words = ' '.join(tokens2)

    # Create a TF-IDF matrix
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([s1words, s2words])

    # Calculate cosine similarity
    cos_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

    return round(cos_sim[0][0])


# In[ ]:




