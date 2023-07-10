from flask import Flask, jsonify, request, render_template
import pandas as pd
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

df = pd.read_csv('Precily_Text_Similarity.csv')
model = SentenceTransformer('all-MiniLM-L6-v2')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_text_options', methods=['GET'])
def get_text_options():
    text1_options = df['text1'].tolist()
    text2_options = df['text2'].tolist()
    return jsonify(text1=text1_options, text2=text2_options)


@app.route('/calculate_similarity', methods=['POST'])
def calculate_similarity():
    data = request.get_json()
    sentence1 = data['text1']
    sentence2 = data['text2']
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
    similarity_score = round(cosine_scores.item(), 2)
    return jsonify({"similarity score": similarity_score})


if __name__ == '__main__':
    app.run()



