import torch
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertForTokenClassification
from sklearn.neighbors import NearestNeighbors
from datasets import load_dataset

word2vec_model = Word2Vec.load("word2vec_recipes.model")

def get_recipe_vector(text_tokens, model):
    vectors = [model.wv[word] for word in text_tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

dataset = load_dataset("d0rj/povarenok_recipes_detail")["train"]
recipe_data = pd.DataFrame(dataset)
recipe_data["vector_description"] = recipe_data["description"].apply(
    lambda x: get_recipe_vector(x.split() if isinstance(x, str) else [], word2vec_model)
)

recipe_data["vector_ingredients"] = recipe_data["ingredients"].apply(
    lambda x: get_recipe_vector(" ".join([ing["name"] for ing in x if "name" in ing]).split() if isinstance(x, list) else [], word2vec_model)
)

recipe_data["recipe_vector"] = recipe_data.apply(
    lambda row: np.concatenate([row["vector_description"], row["vector_ingredients"]]), axis=1
)
recipe_vectors = np.vstack(recipe_data["recipe_vector"].values)

knn_model = NearestNeighbors(n_neighbors=5, metric="cosine")
knn_model.fit(recipe_vectors)

model = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3)
model.load_state_dict(torch.load("ner_model.pth", map_location=torch.device("cpu")))
tokenizer = BertTokenizer.from_pretrained("tokenizer")

def recommend_recipes(ingredients):
    tokens = ingredients.lower().split()
    vector_description = get_recipe_vector(tokens, word2vec_model)
    vector_ingredients = get_recipe_vector(tokens, word2vec_model)

    input_vector = np.concatenate([vector_description, vector_ingredients]).reshape(1, -1)

    distances, indices = knn_model.kneighbors(input_vector, n_neighbors=5)
    
    print("Рекомендованные рецепты:")
    for idx in indices[0]:
        print(recipe_data.iloc[idx]['title'])

if __name__ == "__main__":
    user_input = input("Введите список ингредиентов через запятую: ")
    recommend_recipes(user_input)
