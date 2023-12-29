from flask import Flask, request, jsonify # imports the necessary modules from the Flask library to create a web application, handle HTTP requests, and work with JSON data.
import pandas as pd # import the Pandas library, which are commonly used for data manipulation .
import numpy as np #import the NumPy library, which are commonly used for numerical operations.
import re  # imports the regular expression (regex) module, which is used for pattern matching and manipulation of strings.

from nltk.stem import WordNetLemmatizer  # used for lemmatization (reducing words to their base or root form).
from nltk.corpus import stopwords  # imports the stopwords module from nltk, which contains a list of common words (e.g., "the," "and") that are often excluded from text processing.
from nltk import word_tokenize # imports the word_tokenize function from nltk, which is used to split text into individual words (tokens).
import nltk
nltk.download('stopwords')
nltk.download('punkt')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#This function preprocesses a given text by removing non-alphanumeric characters, converting to lowercase, and removing stopwords.
def preprocess_text(text):
    # Remove non-alphanumeric characters and convert to lowercase
    sent = re.sub('[^A-Za-z0-9]+', ' ', text)
    # Remove stopwords
    sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords.words('english'))
    return sent.lower().strip()

#This function calculates the cosine similarity between two input texts after preprocessing them using lemmatization and TF-IDF transformation.
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

#This function defines the main endpoint of the Flask application. It receives JSON data with 'text1' and 'text2', validates the input, calculates the similarity score using the calculate_cosine_similarity function, and returns the result as JSON. If there are validation errors or other exceptions, it returns an error message.
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