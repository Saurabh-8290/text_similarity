# Text Similarity API Testing
 
### Approach of Part A:

1) Data Loading and Exploration:
    pandas: Imports the CSV data into a DataFrame for efficient manipulation and analysis.
    numpy: Potentially used for numerical operations later (not shown in this code snippet).
    df.shape: Displays the DataFrame's dimensions to understand its structure.
    df.isnull().sum(): Checks for missing values to ensure data integrity.
2) Text Preprocessing:
    re: Employs regular expressions to clean the text by removing non-alphanumeric characters.
    nltk.corpus.stopwords: Loads a list of common English stopwords (e.g., "the", "a", "and") to filter out less meaningful words.
    preprocess_text() function: Encapsulates the cleaning steps for reusability.
3) Word Tokenization and Lemmatization:
    nltk.word_tokenize: Splits text into individual words (tokens) for further analysis.
    nltk.stem.WordNetLemmatizer: Reduces words to their root forms (lemmas) to improve similarity comparisons.
    word_tokenizer() function: Organizes tokenization and lemmatization for consistency.
4) Feature Representation:
    sklearn.feature_extraction.text.TfidfVectorizer: Creates a TF-IDF (Term Frequency-Inverse Document Frequency) matrix, a numerical representation that captures word importance within each text snippet.
5) Similarity Calculation:
    sklearn.metrics.pairwise.cosine_similarity: Computes the cosine similarity between pairs of text snippets based on their TF-IDF representations. Cosine similarity measures the angle between vectors, indicating similarity in text content.
    calculate_similarity() function: Applies TF-IDF and cosine similarity to each row of the DataFrame.
6) Result Generation:
    Creates new columns in the DataFrame to store the calculated similarity scores and a binary score (0 or 1) based on a threshold of 0.5.
7) Key Libraries and Tools:
    pandas: Essential for data manipulation and analysis.
    numpy: Often used for numerical computations, though not explicitly shown in this code.
    re: Provides regular expression capabilities for text cleaning.
    nltk: Natural Language Toolkit for tasks like stopword removal, tokenization, and lemmatization.
    sklearn: Scikit-learn library offers tools for feature extraction (TF-IDF) and similarity calculations.

 
### Approach of Part B:

1) Import Necessary Libraries:
    flask: For building the web application and handling requests.
    pandas, numpy, re, nltk, sklearn: Identical to those used in the previous code for text processing and similarity calculations.
2) Define Text Preprocessing and Similarity Calculation Functions:
    preprocess_text(): Cleans text data for comparison (identical to the previous code).
    calculate_cosine_similarity(): Calculates cosine similarity between two preprocessed text snippets (identical to the previous code).
3) Create a Flask Application:
    app = Flask(name): Initializes a Flask application instance.
    Define a Route for Handling Requests:
    @app.route('/', methods=['POST', 'GET']): Decorates a function to handle requests to the root URL of the application, accepting both POST and GET methods.
4) Implement the Request Handling Function:
    index():
        Prints a log message for debugging purposes.
        Retrieves input data: Parses JSON data from the request body, containing 'text1' and 'text2' fields.
        Validates input: Raises a ValueError if 'text1' or 'text2' is missing.
        Calls the similarity calculation function: Passes the extracted text snippets to calculate_cosine_similarity() to compute the similarity score.
        Formats the result as JSON: Creates a JSON response containing the calculated similarity score.
        Handles errors: Catches ValueError and other exceptions, returning appropriate error messages in JSON format.
5) Run the Application:
    if name == 'main': Ensures the code runs only when executed directly, not when imported as a module.
    app.run(debug=True): Starts the Flask application in debug mode, enabling automatic reloading upon code changes.
 

## How to Test API
 


This repository contains instructions for testing the Text Similarity API using cURL and a Python script (`TestAPI.py`). The API measures the similarity between two text inputs.

 ## Testing the API using cURL


1. Open your terminal.
 
2. Execute the following cURL command, replacing `{Enter your text1}` and `{Enter your text2}` with your desired text inputs:
 
    ```
    curl -X POST -d "{\"text1\": \"{Enter your text1}\",\"text2\": \"{Enter your text2}\"}" -H "Content-Type: application/json" https://text-similarity-c378e9b00a22.herokuapp.com/
    ```
 
3. Review the API response in the terminal to assess the text similarity.
 
## Running the Python Script
 
1. Ensure you have the `requests` library installed. If not, install it using:
 
    ```
    pip install requests
    ```
 
2. Run the Python script `TestAPI.py`. The script prompts for `text1` and `text2`. Enter your desired text inputs, and it will display the API response.
 
    ```
    python TestAPI.py
    ```
 
## Note
 
- The API endpoint is hosted at https://text-similarity-c378e9b00a22.herokuapp.com/. Ensure a stable internet connection for successful API communication.
 
- The Python script (`TestAPI.py`) demonstrates how to interact with the API using the `requests` library.