'''
This script uses the Gemma model to extract named entities from text.
'''

import os
import datetime
import logging
import torch
import ast # Added for safe string-to-list parsing
from transformers import pipeline
from pymongo import MongoClient

# --- CONFIGURATION ---

# SET DB COLLECTION
database_name = "MODAL_data"  # Replace with your database name
collection_name = "IN"   # Replace with your collection name

# MODEL AND DEVICE
model_name = "google/gemma-3-4b-it"

# RECOMMENDED: Use GPU if available for a massive performance boost.
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# --- INITIALIZATION ---
# Configure logging once at the start of the script.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# SECURITY: Load token from an environment variable instead of hardcoding.
# Set this in your terminal before running: export HF_TOKEN='your_hugging_face_token'
hf_token = os.getenv("HF_TOKEN")
# hf_token = "MY_TOKEN" # insecure alternative
if not hf_token:
    logging.warning("Hugging Face token not found in environment variables. Downloads may fail for gated models.")

# Connect to MongoDB
try:
    client = MongoClient("mongodb://localhost:27017/")
    db = client[database_name]
    collection = db[collection_name]
    # Test connection
    client.admin.command('ping')
    logging.info("Successfully connected to MongoDB.")
except Exception as e:
    logging.critical(f"Failed to connect to MongoDB: {e}")
    exit() # Exit if DB connection fails

# Load the pipeline
# The pipeline handles model and tokenizer loading internally.
logging.info(f"Loading model '{model_name}' on device '{device}'...")
pipe = pipeline(
    "text-generation",
    model=model_name,
    token=hf_token,
    torch_dtype=torch.bfloat16,
    device=device,
)
logging.info("Model loaded successfully.")

# --- FUNCTIONS ---

def update_record_in_mongodb(record_id, enrichment_data):
    """Update the MongoDB record with NER enrichments."""
    try:
        collection.update_one(
            {"_id": record_id},
            {"$push": {"enrichments": enrichment_data}}
        )
    except Exception as e:
        logging.error(f"MongoDB update error for record {record_id}: {e}")

def parse_model_response(response_str: str):
    """Safely parse the model's string output into a Python list."""
    # The model might add prose like "Here is the list: [...]". We try to find the list itself.
    start_index = response_str.find('[')
    end_index = response_str.rfind(']')
    if start_index != -1 and end_index != -1:
        list_str = response_str[start_index : end_index + 1]
        try:
            # Use ast.literal_eval for safely evaluating a string containing a Python literal.
            return ast.literal_eval(list_str)
        except (ValueError, SyntaxError):
            logging.warning(f"Could not parse model output as a list. Storing raw string: {list_str}")
            return response_str # Fallback to the extracted string part
    return response_str # Fallback to the raw response

# --- MAIN PROCESSING LOGIC ---

error_count = 0
file_count = 0
start_doc_index = 10
num_docs_to_process = 180

# CHANGED: Use skip() and limit() for efficient database pagination.
cursor = collection.find({"word_count": {"$gt": 20}}).skip(start_doc_index).limit(num_docs_to_process)
# print(list(cursor))
logging.info(f"Starting processing of {num_docs_to_process} documents...")


for record in cursor:
    file_name = record.get("file_name", "N/A") # Use .get() for safety
    text = record.get("extracted_text")
    print(text)

    if not text:
        logging.warning(f"Skipping record {record['_id']} (file: {file_name}) due to missing 'extracted_text'.")
        continue

    # Get first 2000 characters for better context, Gemma can handle it.
    text_start = text[:min(2000, len(text))]

    # Define the chat messages for the prompt
    messages = [
        {
            "role": "system",
            "content": "You are a precise entity extraction system. Your task is to identify and extract names of countries, people, and languages from the given text. Provide the output as a clean Python list of dictionaries, like this: `[{'person': 'John Doe'}, {'country': 'Canada'}, {'language': 'French'}]`. Do not add any explanations or introductory text."
        },
        {
            "role": "user",
            "content": f"Please extract all countries, people, and languages from the following text:\n\nText:\n{text_start}"
        }
    ]

    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        outputs = pipe(
            prompt,
            max_new_tokens=500, # Increased slightly for potentially longer lists
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )

        raw_response = outputs[0]["generated_text"][len(prompt):]
        parsed_entities = parse_model_response(raw_response)

        logging.info(f"Successfully processed file: {file_name}")
        print(f"Extracted Entities:\n{parsed_entities}\n--------------------------------")

        if parsed_entities:
            enrichment_data = {
                "model_used": model_name,
                "enrichment_date": datetime.datetime.utcnow().isoformat(),
                "entities": parsed_entities # Storing the parsed, structured data
            }
            update_record_in_mongodb(record["_id"], enrichment_data)

        file_count += 1

    except Exception as e:
        logging.error(f"An unexpected error occurred while processing file {file_name}: {e}")
        error_count += 1
        continue

# --- FINALIZATION ---
logging.info("Script finished.")
logging.info(f"Total files processed: {file_count}")
logging.info(f"Total errors encountered: {error_count}")

# ADDED: Close the MongoDB connection
client.close()
logging.info("MongoDB connection closed.")