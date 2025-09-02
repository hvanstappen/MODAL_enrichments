import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
from mistralai import Mistral
from pydantic import BaseModel

class Subject(BaseModel):
    subject: str
from pymongo import MongoClient

# # Connect to MongoDB
# client = MongoClient("mongodb://localhost:27017/")
# db = client["MODAL_testdata"]
# collection = db["ADVN_VEA260_VUJO"]

# documents = collection.find(
#     {
# "enrichments.model_used": "BERTopic"
#     },
#     {"_id": 1, "extracted_text": 1, "language": 1}
# )
openaikey = "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"
client = OpenAI(api_key=openaikey)


def topic_representation2subject_openai(topic_representation):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"Antwoord uitsluitend met een onderwerp dat uit slechts enkele woorden bestaat.",
            },
            {
                "role": "user",
                # "content": f"The following is a topic representation. Please generate a subject from it.\n\nTopic representation: {topic_representation}\n\nSubject:",
                "content": f"Vat de volgende topic representation samen in een onderwerp: {topic_representation}\n\nSubject:",
            }
        ],
        model="gpt-4o",
    )
    return chat_completion.choices[0].message.content.strip()

# def topic_representation2subject_mistral(topic_representation):

def topic_representation2subject_mistral(topic_representation):
    # topic_representation_cleaned = ", ".join([word.strip("' ") for word in eval(topic_representation)])
    # print(f'cleaned: {topic_representation_cleaned}')
    api_key = os.environ["MISTRAL_API_KEY"]
    # model = "open-mistral-7b"
    # model = "open-mistral-nemo"
    model = "mistral-small-latest"

    client = Mistral(api_key=api_key)

    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"Antwoord uitsluitend met een onderwerp dat uit slechts enkele woorden bestaat.",
            },
            {
                "role": "user",
                # "content": f"The following is a topic representation. Please generate a subject from it.\n\nTopic representation: {topic_representation}\n\nSubject:",
                "content": f"Vat de volgende topic representation samen in een onderwerp: {topic_representation}\n\nSubject:",
            }
        ],
        temperature=0
    )

    # chat_response = client.chat.parse(
    #     model=model,
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": "Antwoord uitsluitend in het Nederlands.",
    #         },
    #         {
    #             "role": "system",
    #             "content": "Vat de trefwoorden samen tot een eenvoudige onderwerpsterm.",
    #         },
    #         {
    #             "role": "user",
    #             "content": f"{topic_representation_cleaned}",
    #         }
    #     ],
    #     temperature=0,
    #     response_format=Subject
    # )

    response = chat_response.choices[0].message.content
    return response


# topic_representation = "'detention', 'prison', 'detained', 'sentenced', 'november', 'appeals', 'china', 'beijing', 'internationalpen', 'email'"
# subject_mistral = topic_representation2subject_mistral(topic_representation)
# print(subject_mistral)
#
# subject_ai = topic_representation2subject_openai(topic_representation)
# print(subject_ai)
