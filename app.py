#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

from nltk.stem import WordNetLemmatizer  # For Lemmetization of words
from nltk.corpus import stopwords  # Load list of stopwords
from nltk import word_tokenize # Convert paragraph in tokens

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    vectorize = TfidfVectorizer()
    tfidf_matrix = vectorize.fit_transform([s1words, s2words])

    # Calculate cosine similarity
    cos_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

    return round(cos_sim[0][0])


app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def index():
    try:
        print("Entering in Def")
        # Get the input texts from the request
        data = request.get_json()

        # Validate input data
        if 'text1' not in data or 'text2' not in data:
            raise ValueError("Both 'text1' and 'text2' must be provided.")

        text1 = data['text1']
        text2 = data['text2']

        # Calculate similarity using the function
        cos_sim = calculate_cosine_similarity(text1, text2)

        # Return the result as JSON
        result = {'similarity_score': cos_sim}
        return jsonify(result)

    except ValueError as ve:
        return jsonify({'error': str(ve)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)