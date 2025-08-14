'''
This script uses the Gemma model to extract named entities from text.
Quality is  better than wikineural script, but only with the larger model google/gemma-3-4b-it.
'''

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from pymongo import MongoClient
import datetime
import logging


# Connect to the MongoDB server
client = MongoClient("mongodb://localhost:27017/")  # Update URI as needed
db = client["MODAL_testdata"]  # Replace with  database name
collection = db["ADVN_VEA260_VUJO"]  # Replace with  collection name

token = "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"

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# model_name = "google/gemma-2-2b-it"
# model_name = "google/gemma-3-1b-it"
model_name = "google/gemma-3-4b-it"

# Load model in 8-bit precision
model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, max_new_tokens=200, device=device)


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
for record in collection.find({"word_count": {"$gt": 20}})[20:200]:
     logging.basicConfig(level=logging.INFO)

     file_name = record["file_name"]
     # logging.info(f"Processing file: {file_name}")
     text = record["extracted_text"]
     text_start = text[:min(1000, len(text))] #get first x characters of text
     # logging.info(f"Text fragment: {text_start[:100]}...")  # Log only the start of the text

     # message = [
     #     {
     #         "role": "system",
     #         "content": "You are a text corrector specialized in the correction of person names."},
     #     {
     #         "role": "user",
     #         "content": "Welke personen worden in deze tekst vermeld?:\n\n{}".format(
     #             text_start)
     #     }
     # ]

     message = [
         {
             "role": "system",
             "content": "You are an geographic entities extraction system. You answer concise, without explaining your answer."},
         {
             "role": "user",
             "content": "Given a text, your task is to extract all names of countries, people and languages - nothing else. The output should be in a list of key-value pairs, e.g. [country : Nigeria].\nText:\n{}".format(
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
             top_p=0.5,
             # pad_to_multiple_of=8
         )

        # text = extract_text_after_recipients(text)

         print(f"\n\nFile name: {file_name}")
         print(f"Text fragment:\n{text_start}\n")

         if not outputs or "generated_text" not in outputs[0]:
             logging.error(f"Unexpected model output: {outputs}")
             continue  # Skip to the next record

         response = outputs[0]["generated_text"][len(prompt):].replace('#', '')
        # extracted_response = extract_model_reply(response)
         print(f"\nExtracted response:\n{response}")
         file_count += 1
         print(f"Files processed: {file_count}")
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
            # update_record_in_mongodb(record["_id"], enrichment_data)
            print("TEST - record not updated")
            print(enrichment_data)

print(f"Files processed: {file_count}")
print(f"Errors encountered: {error_count}")