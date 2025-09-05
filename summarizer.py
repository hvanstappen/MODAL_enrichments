from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from pymongo import MongoClient
import datetime
import logging
import os
import re
from dotenv import load_dotenv
load_dotenv()

# SET DB COLLECTION:
database_name = "MODAL_data"
collection_name = "collection_name"

# Connect to the MongoDB server
client = MongoClient("mongodb://localhost:27017/")
db = client[database_name] # Replace with  database name
collection = db[collection_name] #Replace with  collection name

token = os.environ["HF_TOKEN"]

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu" # uncomment this option in case of GPU memory issues

model_name = "google/gemma-3-1b-it"
# model_name = "google/gemma-2-2b-it"

# Load model in 8-bit precision
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, max_new_tokens=48, device=device)

def remove_multiple_newlines(longtext):
    try:
        cleaned_text = re.sub(r'[\n|\r]{1,}', '\n', longtext)
    except:
        cleaned_text = longtext
    cleaned_text = cleaned_text[:min(1000, len(cleaned_text))]
    return cleaned_text

def update_record_in_mongodb(record_id, enrichments):
    """Update the MongoDB record with NER enrichments."""
    try:
        # Ensure "enrichments" array exists and append new data
        collection.update_one(
            {"_id": record_id},
            {"$push": {"enrichments": enrichments}},
            upsert=True  # Ensures the document exists before appending
        )

        # client.close()
    except Exception as e:
        logging.error(f"MongoDB update error for record {record_id}: {e}")

error_count = 0
file_count = 0

# Use batch processing with skip/limit to avoid cursor time out
batch_size = 100
skip = 0

# Select records with text documents to process
query = (
    {
        "$and": [
            {"$or": [{"file_mimetype": "application/msword"},
                     {"file_mimetype": "application/vnd.wordperfect; version=5.1"},
                     {"file_mimetype": "application/vnd.wordperfect; version=5.0"},
                     {"file_mimetype": "application/rtf"},
                     {"file_mimetype": "text/html"},
                     {"file_mimetype": "application/pdf"},
                     {"file_mimetype": "application/vnd.ms-works"},
                     {"file_mimetype": "application/x-tika-msoffice"},
                     {"file_mimetype": "application/vnd.oasis.opendocument.tika.flat.document"},
                     {"file_mimetype": "application/vnd.ms-word.document.macroenabled.12"},
                     {"file_mimetype": "application/msword2"},
                     {"file_mimetype": "application/vnd.wordperfect"},
                     {"file_mimetype": "application/x-mspublisher"},
                     {"file_mimetype": "application/vnd.openxmlformats-officedocument.presentationml.presentation"},
                     {"file_mimetype": "application/vnd.openxmlformats-officedocument.presentationml.slideshow"},
                     {"file_mimetype": "application/vnd.oasis.opendocument.text"},
                     {"file_mimetype": "application/vnd.oasis.opendocument.presentation"},
                     {"file_mimetype": "message/rfc822"},
                     {"file_mimetype": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
                     {"file_mimetype": "message/x-emlx"},
                     {"file_mimetype": "application/vnd.ms-powerpoint"},
                     {"file_mimetype": "application/vnd.ms-outlook"},
                     {"file_mimetype": "text/plain"},
                     {"file_mimetype": "application/vnd.wordperfect; version=6.x"}, ]},
            { "enrichments.summary": { "$exists": 0 } },
            {"word_count": {"$gte": 30}},
            { "word_count": {"$lt": 99999999999} }
        ]
    }
)

query_length = len(list(collection.find(query))) # get length of query to use for progress bar

while True:
    torch.cuda.empty_cache()
    batch = list(collection.find(query)
                 .skip(skip)
                 .limit(batch_size))

    if not batch:
        break  # No more documents

    for record in batch:
         logging.basicConfig(level=logging.INFO)

         file_name = record["file_name"]
         # logging.info(f"Processing file: {file_name}")
         text = record["extracted_text"]
         # text_start = text[:min(1000, len(text))] #get first x characters of text
         text_start = remove_multiple_newlines(text)
         # logging.info(f"Text fragment: {text_start[:100]}...")  # Log only the start of the text


         message = [
             {
                 "role": "system",
                 "content": "Geef een antwoord in een korte zin. Geef GEEN verdere toelichting bij je antwoord"},
             {
                 "role": "user",
                 "content": "Vat deze tekst samen in het Nederlands:\n\n{}".format(
                     text_start)
             }
         ]


         prompt = pipe.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
         try:
             outputs = pipe(
                 prompt,
                 do_sample=True,
                 temperature=0.1,
                 top_k=20,
                 top_p=0.1,
                 # pad_to_multiple_of=8
             )


             print(f"\n\nFile name: {file_name}")

             if not outputs or "generated_text" not in outputs[0]:
                 logging.error(f"Unexpected model output: {outputs}")
                 continue  # Skip to the next record

             response = outputs[0]["generated_text"][len(prompt):].replace('#', '')
             print(f"\nExtracted response: {response}")
             file_count += 1
             print(f"Files processed: {file_count} of {query_length}")
             print("--------------------------------")
         except RuntimeError as e:
             logging.error(f"Error processing text from file {file_name}: {e}")
             print(f"Files with error {file_name}: {e}")
             error_count += 1
             print(f"Errors encountered: {error_count}")
             continue

         if any(response):  # Check if there's at least one non-empty entity list
                # Prepare enrichment data
                 enrichment_data = {
                     "model_used": model_name,
                     "enrichment_date": datetime.datetime.utcnow().isoformat(),
                     "summary": response
                 }

                # Update MongoDB record
                 update_record_in_mongodb(record["_id"], enrichment_data)


client.close()

print(f"Files processed: {file_count}")
print(f"Errors encountered: {error_count}")