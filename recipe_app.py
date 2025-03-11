import torch
import pandas as pd
from transformers import BertTokenizer, BertForTokenClassification
from nltk.tokenize import word_tokenize
from pymorphy3 import MorphAnalyzer

model_path = "ner_model.pth"
model = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3)
model.load_state_dict(torch.load(model_path))
model.eval()

tokenizer = BertTokenizer.from_pretrained("tokenizer")

dataset = load_dataset("d0rj/povarenok_recipes_detail")["train"]
df = pd.DataFrame(dataset)

morph = MorphAnalyzer()

def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [morph.parse(word)[0].normal_form for word in words]
    return words

def extract_ingredients(text):
    tokens = word_tokenize(text)
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs).logits
    predictions = torch.argmax(outputs, dim=-1).squeeze().tolist()
    
    ingredients = []
    for token, label_id in zip(tokens, predictions):
        if label_id == 1 or label_id == 2:
            ingredients.append(token)
    return set(ingredients)

def find_matching_recipes(user_ingredients):
    matching_recipes = []
    for _, row in df.iterrows():
        recipe_ingredients = set(preprocess_text(" ".join([ing["name"] for ing in eval(row["ingredients"]) if "name" in ing])))
        match_count = len(user_ingredients & recipe_ingredients)
        if match_count > 0:
            matching_recipes.append((row["title"], match_count))
    matching_recipes.sort(key=lambda x: x[1], reverse=True)
    return matching_recipes[:5]

if __name__ == "__main__":
    user_input = input("Введите список ингредиентов: ")
    extracted_ingredients = extract_ingredients(user_input)
    print(f"Выделенные ингредиенты: {', '.join(extracted_ingredients)}")
    
    recipes = find_matching_recipes(extracted_ingredients)
    print("\nПохожие рецепты:")
    for title, matches in recipes:
        print(f"{title} (Совпадений: {matches})")
