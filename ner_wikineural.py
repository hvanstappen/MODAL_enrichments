# Find named entities in text using Wikineural model

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import datetime
import logging
from pymongo import MongoClient
from collections import Counter
import torch

# Database config
DB_NAME = "MODAL_data"
COLLECTION_NAME = "collection_name"

# Setup logging
LOG_FILE = "logs/" + COLLECTION_NAME + "_error_log.txt"
logging.basicConfig(filename=LOG_FILE, level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer w/ FIX OF ## ISSUE
tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")

# Move model to the appropriate device
model.to(device)

nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=False)

def get_records_with_text(wordlength):
    """Fetch records with word count greater than wordlength."""
    try:
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        query = ({"$and": [{"$nor": [{"enrichments.model_used": "Wikineural"}]}, {"word_count":{"$gte":50}}, {"word_count": {"$lte": 9999999}}]}) # change as needed
        records = list(collection.find(query))
        client.close()
        return records
    except Exception as e:
        logging.error(f"Database error: {e}")
        return []

def clean_token(token):
    """Remove subword prefix (e.g., "##") and handle spacing."""
    if token.startswith("##"):
        return token[2:], False  # no leading space
    return token, True  # leading space required

def chunk_text(text, tokenizer, max_length=512, stride=50):
    """Split text into overlapping chunks that respect token limits."""
    encodings = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False, return_attention_mask=False)
    input_ids = encodings["input_ids"]
    offsets = encodings["offset_mapping"]
    print("chunking..")

    chunks = []
    start = 0
    while start < len(input_ids):
        end = min(start + max_length, len(input_ids))
        chunk_offsets = offsets[start:end]
        chunk_start_char = chunk_offsets[0][0]
        chunk_end_char = chunk_offsets[-1][1]
        chunk_text2 = text[chunk_start_char:chunk_end_char]
        chunks.append((chunk_text2, chunk_start_char))
        if end == len(input_ids):
            break
        start += max_length - stride  # move forward with overlap
    return chunks

def ner_extractor(text):
    """Extract named entities (persons, organizations, locations) from text."""
    print("ner_extractor starting..")
    try:
        chunks = chunk_text(text, tokenizer)
        grouped_entities = []

        for chunked_text, chunk_offset in chunks:
            ner_results = nlp(chunked_text)
            current_entity = None
            current_tokens = []
            current_score_total = 0.0
            current_score_count = 0

            for ent in ner_results:
                entity_type = ent["entity"]
                word = ent["word"]
                score = ent["score"]

                if entity_type.startswith("B-") or (current_entity and entity_type != f"I-{current_entity}"):
                    # Save previous entity
                    if current_entity and current_tokens:
                        full_entity = ""
                        for tok in current_tokens:
                            token_text, is_leading = clean_token(tok)
                            if full_entity and is_leading:
                                full_entity += " "
                            full_entity += token_text
                        avg_score = current_score_total / current_score_count
                        grouped_entities.append({
                            "entity": current_entity,
                            "word": full_entity,
                            "score": avg_score,
                        })

                    # Start new entity
                    current_entity = entity_type[2:] if entity_type.startswith("B-") else entity_type
                    current_tokens = [word]
                    current_score_total = score
                    current_score_count = 1

                elif entity_type.startswith("I-") and current_entity:
                    current_tokens.append(word)
                    current_score_total += score
                    current_score_count += 1
                else:
                    # Not a recognized pattern, reset
                    current_entity = None
                    current_tokens = []
                    current_score_total = 0.0
                    current_score_count = 0

            # Add last entity from chunk
            if current_entity and current_tokens:
                full_entity = ""
                for tok in current_tokens:
                    token_text, is_leading = clean_token(tok)
                    if full_entity and is_leading:
                        full_entity += " "
                    full_entity += token_text
                avg_score = current_score_total / current_score_count
                grouped_entities.append({
                    "entity": current_entity,
                    "word": full_entity,
                    "score": avg_score,
                })

        # Categorize
        entities = {
            "NER_persons": Counter(ent["word"] for ent in grouped_entities if ent["entity"] in ["PER"] and ent["score"] >= 0.60),
            "NER_organisations": Counter(ent["word"] for ent in grouped_entities if ent["entity"] in ["ORG"] and ent["score"] >= 0.60),
            "NER_locations": Counter(ent["word"] for ent in grouped_entities if ent["entity"] == "LOC" and ent["score"] >= 0.60),
            "NER_miscellaneous": Counter(ent["word"] for ent in grouped_entities if ent["entity"] == "MISC" and ent["score"] >= 0.60),
        }

        for key in entities:
            entities[key] = [entity for entity, _ in entities[key].most_common()]

        return entities

    except Exception as e:
        logging.error(f"NER extraction error: {e}")
        print("ner_extractor error")
        return {"NER_persons": [], "NER_organisations": [], "NER_locations": [], "NER_miscellaneous":[]}

def update_record_in_mongodb(record_id, enrichments):
    """Update the MongoDB record with NER enrichments."""
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        # Ensure "enrichments" array exists and append new data
        collection.update_one(
            {"_id": record_id},
            {"$push": {"enrichments": enrichments}},
            upsert=True  # Ensures the document exists before appending
        )

        client.close()
    except Exception as e:
        logging.error(f"MongoDB update error for record {record_id}: {e}")


# Fetch records from MongoDB
records = get_records_with_text(50)

if not records:
    print("No records found or database connection failed. Check log file.")
else:
    number_of_records = len(records)
    number_of_processed_records = 0
    for record in records:
        number_of_processed_records += 1
        print(f"\n===========\nProcessing record {record['_id']} - {number_of_processed_records}/{number_of_records}")
        try:
            text = record.get("extracted_text", "").strip()

            if not text:
                logging.warning(f"Skipping record {record['_id']} - Empty text field.")
                continue

            # Extract NER entities
            entities = ner_extractor(text)
            print(entities)

            # Ensure we only update records with extracted entities
            # Prepare enrichment data
            enrichment_data = {
                "model_used": "Wikineural",
                "enrichment_date": datetime.datetime.utcnow().isoformat(),
                **entities  # Merge extracted entities into JSON structure
            }

            # Update MongoDB record
            update_record_in_mongodb(record["_id"], enrichment_data)

        except Exception as e:
            logging.error(f"Error processing record {record['_id']}: {e}")

print("\n✅ Script completed. Check error_log.txt for any issues.")
