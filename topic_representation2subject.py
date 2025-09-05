import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
from mistralai import Mistral
from pydantic import BaseModel

class Subject(BaseModel):
    subject: str

openaikey = os.environ["OPENAI_API_KEY"]

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
                "content": f"Vat de volgende topic representation samen in een onderwerp: {topic_representation}\n\nSubject:",
            }
        ],
        model="gpt-4o",
    )
    return chat_completion.choices[0].message.content.strip()

def topic_representation2subject_mistral(topic_representation):
    api_key = os.environ["MISTRAL_API_KEY"]
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


    response = chat_response.choices[0].message.content
    return response
