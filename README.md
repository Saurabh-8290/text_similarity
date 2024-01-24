# Text Similarity API Testing
 
# Approach of Part A
## Data Handling
    1) Load CSV Data: We use the pandas library to efficiently load the text data from a CSV file into a DataFrame, a structured table-like format for easy manipulation and analysis.
    2) Explore Data: We inspect the DataFrame's dimensions using df.shape to understand its structure and check for missing values using df.isnull().sum() to ensure data integrity.
## Text Cleaning
    1) Remove Unwanted Characters: We employ regular expressions from the re library to discard non-alphanumeric characters, such as punctuation and symbols, that might not be relevant for text comparison.
    2) Filter Common Words: We leverage the nltk library's list of common English stopwords (e.g., "the", "a", "and") to remove them, as they often carry less meaning in text analysis.
    3) Encapsulate Steps: We create a reusable function called preprocess_text() to streamline these cleaning steps.
## Word Processing
    1) Split into Words: We tokenize the text, meaning we split it into individual words or tokens, using nltk.word_tokenize() for further analysis.
    2)Reduce to Roots: We apply lemmatization with nltk.stem.WordNetLemmatizer() to reduce words to their root forms (lemmas), improving similarity comparisons by grouping related words together.
    3) Organize Process: We organize tokenization and lemmatization within a word_tokenizer() function for consistency.
## Feature Extraction
    1) TF-IDF Representation: We create a TF-IDF (Term Frequency-Inverse Document Frequency) matrix using sklearn.feature_extraction.text.TfidfVectorizer. This numerical representation captures the importance of words within each text snippet, considering both their frequency within a document and their rarity across the entire dataset.
## Similarity Measurement
    1) Compute Cosine Similarity: We calculate the cosine similarity between pairs of text snippets based on their TF-IDF representations using sklearn.metrics.pairwise.cosine_similarity. Cosine similarity measures the angle between vectors, with higher values indicating greater similarity in text content.
    2) Apply to DataFrame: We create a calculate_similarity() function to apply TF-IDF and cosine similarity to each row of the DataFrame.
## Generate Results
    1) Store Similarity Scores: We add new columns to the DataFrame to store the calculated similarity scores and a binary score (0 or based on a threshold of 0.5, indicating whether pairs of text snippets are considered similar or not.
## Key Libraries
    1) pandas: Essential for data manipulation and analysis.
    2) numpy: Often used for numerical computations (not explicitly shown in this code).
    3) re: Provides regular expression capabilities for text cleaning.
    4) nltk: Natural Language Toolkit for tasks like stopword removal, tokenization, and lemmatization.
    5) sklearn: Scikit-learn library offers tools for feature extraction (TF-IDF) and similarity calculations.

 
# Approach of Part B
## Gather Essential Tools:
    1) flask: This library serves as the foundation for building the web application and managing incoming requests.
    2) pandas, numpy, re, nltk, sklearn: These libraries, already introduced in Part A, continue to play crucial roles in text processing and similarity calculations.
## Define Core Functions for Reusability:
    1) preprocess_text(): This function, identical to the one in Part A, diligently cleanses text data for accurate comparison.
    2) calculate_cosine_similarity(): This function, also mirroring its counterpart in Part A, meticulously computes the cosine similarity between two preprocessed text snippets.
## Construct the Flask Application:
    1) app = Flask(name): This line of code meticulously initializes a Flask application instance, laying the groundwork for the web API.
## Establish a Route:
    1) @app.route('/', methods=['POST', 'GET']): This decorator meticulously designates a function to gracefully handle requests directed to the root URL of the application. It demonstrates a welcoming embrace of both POST and GET methods, fostering flexibility in data submission.
### Implement the index() Function:
    1) Log Message (Debugging): A message is elegantly printed for debugging purposes, aiding in troubleshooting and code refinement.
    2) Retrieve Input Data: The function meticulously parses JSON data from the request body, eagerly extracting the 'text1' and 'text2' fields that hold the text snippets awaiting comparison.
    3) Validate Input: To ensure integrity, a ValueError is decisively raised if either 'text1' or 'text2' is missing, preventing potential errors and promoting data quality.
    4) Calculate Similarity: The extracted text snippets are diligently passed to the calculate_cosine_similarity() function, which meticulously computes the similarity score, quantifying the degree of resemblance between the two text segments.
    5) Format Result as JSON: The calculated similarity score is elegantly encapsulated within a JSON response, a format embraced for its readability and cross-platform compatibility, enabling seamless integration with diverse systems.
    6) Handle Errors Gracefully: The function vigilantly catches ValueError and other unforeseen exceptions, gracefully returning informative error messages in JSON format to aid in troubleshooting and promote a user-friendly experience.
## Execute Independently:
    1) if name == 'main': This conditional statement ensures that the code embarks on its execution journey only when directly invoked, not when imported as a module within another script, preserving code organization and modularity.
    2) app.run(debug=True): This line enthusiastically initiates the Flask application in debug mode, fostering a development-friendly environment with automatic reloading upon code modifications, streamlining the development process and fostering efficiency.




# How to Test API
 
This repository contains instructions for testing the Text Similarity API using cURL and a Python script (`TestAPI.py`). The API measures the similarity between two text inputs.

 ## Testing the API using cURL

1. Open your terminal.
 
2. Execute the following cURL command, replacing `{Enter your text1}` and `{Enter your text2}` with your desired text inputs:
 
    ```
    curl -X POST -d "{\"text1\": \"{Enter your text1}\",\"text2\": \"{Enter your text2}\"}" -H "Content-Type: application/json" https://textsimilarity-4eac8082be50.herokuapp.com/
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
 
- The API endpoint is hosted at https://textsimilarity-4eac8082be50.herokuapp.com/. Ensure a stable internet connection for successful API communication.
 
- The Python script (`TestAPI.py`) demonstrates how to interact with the API using the `requests` library.
