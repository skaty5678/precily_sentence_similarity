# precily_sentence_similarity_APP

This is a Flask-based APP for calculating similarity scores between two sentences using a Sentence Transformer model.

## Prerequisites

- Python 3.7 or above
- Install the required dependencies by running `pip install -r requirements.txt`

## Usage

1. Start the API server by running `python app.py`.
2. Open your web browser and navigate to `http://localhost:5000`.
3. Use the web interface to select sentences or provide custom input for text1 and text2.
4. Click the "Calculate Similarity" button to send a POST request to the API and display the similarity score.


## Files

- `app.py`: The main Flask application file.
- `index.html`: The HTML template file for the web interface.
- `Precily_Text_Similarity.csv`: The CSV file containing the text data for similarity calculation.

## Dependencies

- Flask: A micro web framework for building the API.
- pandas: A library for data manipulation and analysis.
- sentence_transformers: A library for encoding sentences into embeddings using various transformer models.


*index.html created with the help of chatgpt*
