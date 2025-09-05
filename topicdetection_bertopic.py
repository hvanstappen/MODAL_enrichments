from datetime import datetime
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from pymongo import MongoClient
import numpy as np
from bson import ObjectId
from tools.text_functions import remove_stopwords, remove_numbers
from topic_representation2subject import topic_representation2subject_openai
import os
from sentence_transformers import SentenceTransformer

# SET DB COLLECTION:
database_name = "MODAL_data"
collection_name = "collection_name"

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client[database_name]
collection = db[collection_name]

# Select text documents with enough words to be considered for topic detection:
documents = collection.find(
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
            {"word_count": {"$gte": 100}}
        ]
    },
    {"_id": 1, "extracted_text": 1, "language": 1}
)

# Extract text and IDs correctly and remove stopwords
docs = []
doc_ids = []
for doc in documents:
    if "extracted_text" in doc:
        cleaned_text = remove_stopwords(doc["extracted_text"], doc["language"])  # remove stopwords
        cleaned_text = remove_numbers(cleaned_text)
        cleaned_text = cleaned_text[:min(10000, len(cleaned_text))]
        docs.append(cleaned_text)
        doc_id = str(doc["_id"])  # Ensure ID is a string
        doc_ids.append(doc_id)

num_docs = len(docs)
num_enrichments = 0
num_no_topic_found = 0
print(f"Number of documents: {num_docs}")

# Set the number of topics, between 10 and 100
min_topic_size = int(num_docs / 500)
if min_topic_size < 10:
    min_topic_size = 10
    print(f"Minimum topic size set to {min_topic_size} because the number of documents is too small.")
if min_topic_size > 100:
    min_topic_size = 100
    print(f"Minimum topic size set to {min_topic_size} because the number of documents is too large.")

if not docs:
    raise ValueError("No documents found in the database.")

# Initialize BERTopic
print("loading model...")
# Force the SentenceTransformer embedding model to run on the CPU
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
representation_model = KeyBERTInspired()
topic_model = BERTopic(embedding_model=embedding_model,
                       representation_model=representation_model,
                       language="multilingual",
                       calculate_probabilities=True,
                       verbose=True,
                       min_topic_size=min_topic_size)

print("detecting topics...")
topics, probs = topic_model.fit_transform(docs)

# Ensure probs is not None
if probs is None:
    print("Warning: No probabilities were calculated. Proceeding without them.")
    probs = [None] * len(topics)

#get topics and write to CSV file:
info = topic_model.get_topic_info()
csv_path = "data/Bert_topics_" + collection_name + ".csv"
info.to_csv(csv_path, index=False)

# Get topics for each document and write to database
print("creating subjects...")
if 'subject' not in info.columns:
    info['subject'] = None

for index, row in info.iterrows():
    topic_representation = row['Representation']
    print(topic_representation)
    keywords = "'" + ", ".join(item for item in topic_representation) + "'"
    print(keywords)
    subject = topic_representation2subject_openai(keywords) # create label
    print(subject)
    info.at[index, 'subject'] = subject

info.to_csv(csv_path, index=False)

print(info)

current_date = datetime.now().isoformat()


# Helper to handle doc_prob
def extract_scalar_probability(doc_prob):
    if isinstance(doc_prob, np.ndarray):
        if doc_prob.size == 1:
            return float(doc_prob[0])
        else:
            average = np.mean(doc_prob)
            print(f"Warning: doc_prob is a multi-element array: {doc_prob}. Using the average: {average}")
            return float(average)
    elif isinstance(doc_prob, (float, int)):
        return float(doc_prob)
    return None

topic_labels_dict = dict(zip(info["Topic"], topic_model.generate_topic_labels()))

for doc_id, doc_topic, doc_prob in zip(doc_ids, topics, probs):
    if doc_topic != -1:
        num_enrichments += 1
        topic_representation = topic_model.get_topic(doc_topic) or []
        topic_name = topic_labels_dict.get(doc_topic, f"Topic {doc_topic}")
        topic_keywords = [word for word, _ in topic_representation]
        subject = info.loc[info['Topic'] == doc_topic, 'subject'].values[0] if not info.loc[
            info['Topic'] == doc_topic, 'subject'].empty else None
        if subject is None:
            subject = f"no subject"
        topic_probability = float(doc_prob[doc_topic]) if doc_prob is not None and doc_topic != -1 else None  #FIX 1

    else:
        num_no_topic_found += 1
        topic_representation = topic_model.get_topic(doc_topic) or []
        topic_keywords = [word for word, _ in topic_representation]
        topic_name = "No Topic Found"
        topic_probability = None
        subject = info.loc[info['Topic'] == doc_topic, 'subject'].values[0] if not info.loc[
            info['Topic'] == doc_topic, 'subject'].empty else None
        if subject is None:
            subject = f"no subject"

    enrichment_data = {
        "model_used": "BERTopic",
        "enrichment_date": current_date,
        "Topic_representation": topic_keywords,
        "Topic_label": subject,
        "Topic_name": topic_name,
        "Topic_probability": topic_probability
    }

    # Check if the document already has enrichments, modify the logic accordingly
    existing_doc = collection.find_one({"_id": ObjectId(doc_id) if len(doc_id) == 24 else doc_id})
    if not existing_doc or 'enrichments' not in existing_doc:
        # If no enrichments exist, initialize the field
        collection.update_one(
            {"_id": ObjectId(doc_id) if len(doc_id) == 24 else doc_id},
            {"$set": {"enrichments": [enrichment_data]}},  # Initialize with the first enrichment data
            upsert=True
        )
    else:
        # If enrichments exist, append to the existing list
        collection.update_one(
            {"_id": ObjectId(doc_id) if len(doc_id) == 24 else doc_id},
            {"$push": {"enrichments": enrichment_data}}
        )



print(f'Documents with topics found: {num_enrichments}, documents without topics found: {num_no_topic_found}')
print("Topics with probabilities have been written back to the database.")
