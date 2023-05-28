from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_names = {
    "ProsusAI/finbert": "finbert",
    "finiteautomata/bertweet-base-sentiment-analysis": "bertweet"
}

tokenizer_names = {
    "ProsusAI/finbert": "finbert",
    "finiteautomata/bertweet-base-sentiment-analysis": "bertweet"
}

for model_name in model_names.keys():
    label = model_names[model_name]
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.save_pretrained(f"./app/artifacts/models/{label}")

for tokenizer_name in tokenizer_names.keys():
    label = tokenizer_names[tokenizer_name]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained(f"./app/artifacts/tokenizers/{label}")



