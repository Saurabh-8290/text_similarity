# Import necessary libraries
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer  # For Lemmetization of words
from nltk.corpus import stopwords  # Load list of stopwords
from nltk import word_tokenize # Convert paragraph in tokens
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# Read the CSV file into a Pandas DataFrame
df = pd.read_csv('Precily_Text_Similarity.csv')
 
# Display the shape of the DataFrame (number of rows and columns)
df.shape
 
# Display the number of null values in each column of the DataFrame
df.isnull().sum()
 
# Create a copy of the DataFrame for preprocessing
df1 = df.copy()
 
# Define a function for text preprocessing
def preprocess_text(text_column):
    preprocessed_text = []
    for sentence in tqdm(text_column.values):
        # Remove non-alphanumeric characters and convert to lowercase
        sent = re.sub('[^A-Za-z0-9]+', ' ', sentence)
        # Remove stopwords
        sent = ' '.join(e for e in sent.split() if e not in stopwords.words('english'))
        preprocessed_text.append(sent.lower().strip())
    return preprocessed_text
 
# Apply text preprocessing to 'text1' and 'text2' columns of the DataFrame
df1['text1'] = preprocess_text(df1['text1'])
df1['text2'] = preprocess_text(df1['text2'])
 
# Create lists to store preprocessed text for 'text1' and 'text2'
preprocessed_text1 = []
preprocessed_text2 = []
 
# Apply text preprocessing to 'text1' and 'text2' columns using a loop
for sentence in tqdm(df['text1'].values):
    sent = re.sub('[^A-Za-z0-9]+', ' ', sentence)
    sent = ' '.join(e for e in sent.split() if e not in stopwords.words('english'))
    preprocessed_text1.append(sent.lower().strip())
 
for sentence in tqdm(df['text2'].values):
    sent = re.sub('[^A-Za-z0-9]+', ' ', sentence)
    sent = ' '.join(e for e in sent.split() if e not in stopwords.words('english'))
    preprocessed_text2.append(sent.lower().strip())
 
# Update 'text1' and 'text2' columns in the DataFrame with preprocessed text
df1['text1'] = preprocessed_text1
df1['text2'] = preprocessed_text2
 
# Define a function for word tokenization and lemmatization
def word_tokenizer(text):
    # Tokenize and lemmatize the text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens
 
# Apply word tokenization and lemmatization to 'text1' and 'text2' columns
for i in tqdm(df1.index):
    s1 = df1['text1'][i]
    s2 = df1['text2'][i]
    s1words = ' '.join(word_tokenizer(s1))
    s2words = ' '.join(word_tokenizer(s2))
    df1['text1'][i] = s1words
    df1['text2'][i] = s2words
 
# Initialize a TF-IDF Vectorizer
tfidf = TfidfVectorizer()
 
# Define a function to calculate cosine similarity
def calculate_similarity(row):
    tfidf_matrix = tfidf.fit_transform(row[['text1', 'text2']].values.astype('U'))
    cos_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return cos_sim[0][0]
 
# Apply the calculate_similarity function to create a new 'similarity' column
df1['similarity'] = df1.apply(calculate_similarity, axis=1)
 
# Create a 'binary_score' column based on the similarity threshold of 0.5
df1['binary_score'] = (df1['similarity'] > 0.5).astype(int)
 
print(df1)