from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pymongo import MongoClient
import torch
import math
from datetime import datetime

database_name = "MODAL_testdata" # Replace with database name
collection_name = "LH_HH_71_Hemmerechts" # Replace with collection name
finetuned_model_name = "bert-base-dutch-cased_finetuned" # Replace with fine tuned model folder name

# Get data
client = MongoClient("mongodb://localhost:27017/")
db = client[database_name]
collection = db[collection_name]

cursor = collection.find({'$and': [ {'generic_file_type': 'berichtbestand'}, {'word_count': {'$gte': 100} }, {'word_count': {'$lte': 120}} ] })

texts_all = []
ids_all = []
for doc in cursor:
    if doc.get("extracted_text"):
        texts_all.append(doc["extracted_text"])
        ids_all.append(doc["_id"])

if not texts_all:
    raise ValueError("Geen records gevonden met het veld 'extracted_text'.")

# Load model and tokenizer
model_dir = "./" + finetuned_model_name
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define batch function (avoids memory issues)
def predict_batch(texts):
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()

    id2label = model.config.id2label
    return [id2label[int(i)] for i in predictions]

# Start classification
batch_size = 32
total_batches = math.ceil(len(texts_all) / batch_size)

for i in range(total_batches):
    start = i * batch_size
    end = start + batch_size
    batch_texts = texts_all[start:end]
    batch_ids = ids_all[start:end]

    labels = predict_batch(batch_texts)

    for doc_id, predicted_class in zip(batch_ids, labels):
        enrichment = {
            "model_used": finetuned_model_name,
            "enrichment_date": datetime.utcnow().isoformat(),
            "class": predicted_class
        }

        # Remove existing classifications with the same model
        collection.update_one(
            {"_id": doc_id},
            {"$pull": {"enrichments": {"model_used": enrichment["model_used"]}}}
        )

        # Apply new classification data
        collection.update_one(
            {"_id": doc_id},
            {"$push": {"enrichments": enrichment}}
        )

    print(f"Batch {i+1}/{total_batches} processed and written to MongoDB.")

print("Classification finished. All results written to MongoDB.")

