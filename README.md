# MODAL enrichment

This set of scripts was created as a deliverable for the [project MODAL](https://advn.be/nl/over-advn/projecten/modal-project).

The research project "Metadata and Access to Digital Archives Using Large Language Models," or MODAL for short, launched in the fall of 2024. MODAL's goal is to investigate the potential applications of generative artificial intelligence (GenAI) for the cultural heritage sector and to disseminate this knowledge within the sector.

This a part of a suite of scripts:

* [scripts for text extraction and metadata storage]()
* scripts for enrichments (this set of scripts)
* [scripts for browsing and retrieval of documents]()

## NER
The script `ner_wikineural.py` handles the Named Entity recognition.

The script processes text documents stored in MongoDB by performing Named Entity Recognition (NER) using the [Wikineural](https://huggingface.co/Babelscape/wikineural-multilingual-ner) multilingual model. Specifically, it:

1. Fetches documents from MongoDB that haven't been processed yet (with word count ≥ 50)
2. For each document, extracts named entities in four categories:
   - Persons (PER)
   - Organizations (ORG)
   - Locations (LOC)
   - Miscellaneous (MISC)
3. Updates each MongoDB document with the extracted entities, along with timestamp and model information
4. Handles long texts by splitting them into manageable chunks and processes them sequentially

