import torch
# import pandas as pd
from sentence_transformers import SentenceTransformer, util

# df = pd.read_csv('Precily_Text_Similarity.csv')

model = SentenceTransformer('all-MiniLM-L6-v2')

sentence1 = "In the midst of chaos and uncertainty, she found solace in the beauty of nature." \
            " The vibrant colors of the flowers brought a sense of calmness to her weary soul." \
            " With every step she took in the lush green meadows, her worries faded away like distant memories." \
            " Nature became her sanctuary, a place where she could rejuvenate and find inner peace."

sentence2 = "The bustling city streets filled with people rushing by, each with their own purpose and destination." \
            " Skyscrapers reached for the sky, casting shadows on the busy sidewalks below." \
            " Amidst the concrete jungle, she thrived on the energy and excitement." \
            " The city lights illuminated her path, guiding her through the urban maze." \
            " The rhythm of the city became her heartbeat, a constant reminder of life's endless possibilities."

# Encode sentences to get their embeddings
embedding1 = model.encode(sentence1, convert_to_tensor=True)
embedding2 = model.encode(sentence2, convert_to_tensor=True)

# Compute similarity scores of two embeddings
cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)

# Convert the range from -1 to 1 to 0 to 1
normalized_score = 0.5 * (cosine_scores + 1)
final_score = round(normalized_score.item(), 2)

print("Sentence 1:", sentence1)
print("Sentence 2:", sentence2)
print("Similarity score:", final_score)
