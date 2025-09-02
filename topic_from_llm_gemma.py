from pymongo import MongoClient
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

# Connect to the MongoDB server
client = MongoClient("mongodb://localhost:27017/")  # Update URI as needed
db = client["MODAL_testdata"]  # Replace with  database name
collection = db["LH_HH_71_Kristien_Hemmerechts"]  # Replace with  collection name

# Specify the model name
model_name = "google/gemma-2-2b-it"  # Using 2B since 1B might have limited availability

# Load the tokenizer and model
token = "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"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=token)

instruction = "Beantwoord de volgende vraag in het Nederlands, met enkel een antwoord, gebaseerd op de context. Antwoord in json formaat met een betrouwbaarheidsscore tussen 0 en 1"
# instruction = "Geef een bondig antwoord op de volgende vraag, gebaseerd op de context."

def extract_text_after_recipients(mailtext):
    """
    Extracts the text starting two lines after the word 'recipients' in a given text.

    Args:
        mailtext (str): The input text.

    Returns:
        str: The extracted text, or None if 'recipients' is not found
             or there are fewer than two lines after it.
    """
    lines = mailtext.splitlines()
    try:
        recipients_index = -1
        for i, line in enumerate(lines):
            if 'recipients' in line.lower():  # Case-insensitive search
                recipients_index = i
                break

        if recipients_index != -1 and recipients_index + 2 < len(lines):
            extracted_lines = lines[recipients_index + 2:]
            return "\n".join(extracted_lines)
        else:
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def ask_gemma_it(instruction, context, question):
    """Asks the Gemma instruction-tuned model a question based on the context
    using a structured prompt."""
    prompt = f"user {instruction}\n\nContext: {context}\n\nQuestion: {question}"
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**input_ids, max_new_tokens=200, num_return_sequences=1)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # The model's response starts after the <start_of_turn>model token
    answer_start = answer.find("Answer:") + len("Answer:")
    return answer[answer_start:].strip()


for record in collection.find({"word_count": {"$gt": 20}})[:3]:
    file_name = record["file_name"]
    text = record["extracted_text"]
    text = extract_text_after_recipients(text)
    text_start = text[:1000]
    print(f"\n\nFile name: {file_name}")
    print(f"Text fragment: {text_start}")

    question = "Wat is het onderwerp van deze tekst?"
    answer = ask_gemma_it(instruction, text_start, question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print("--------------------------------")

