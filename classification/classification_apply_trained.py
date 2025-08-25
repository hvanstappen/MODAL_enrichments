from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
import torch
import pandas as pd
import numpy as np

# 1. Laad de nieuwe data
df = pd.read_csv('./data/email_dataset_extra.csv')

text_col = "text_short"  # Pas aan als je CSV een andere kolomnaam gebruikt

if text_col not in df.columns:
    raise ValueError(f"CSV moet een kolom '{text_col}' bevatten.")

# 2. Laad het gefinetunede model en tokenizer
model_dir = "./bert-base-dutch-cased_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Zorg dat model in eval modus staat
model.eval()

# 3. Tokenize de nieuwe teksten
def preprocess(texts):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

encodings = preprocess(df[text_col].tolist())

# 4. Voorspel labels
with torch.no_grad():
    outputs = model(**encodings)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).numpy()

# 5. Zet integer voorspellingen terug naar labels
id2label = model.config.id2label
predicted_labels = [id2label[i] for i in predictions]

# 6. Voeg voorspellingen toe aan dataframe
df['predicted_label'] = predicted_labels

# 7. Sla de resultaten op
df.to_csv('data/test_data_classified.csv', index=False)
print("Classificatie klaar. Resultaten opgeslagen in 'test_data_classified.csv'.")


# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from datasets import Dataset
# import torch
# import pandas as pd
# import numpy as np
# import torch.nn.functional as F
#
# # 1. Laad de nieuwe data
# df = pd.read_csv('data/test_data.csv')
#
# text_col = "text_short"  # Pas aan als nodig
# if text_col not in df.columns:
#     raise ValueError(f"CSV moet een kolom '{text_col}' bevatten.")
#
# # 2. Laad het gefinetunede model en tokenizer
# model_dir = "./bert-base-dutch-cased_finetuned"
# tokenizer = AutoTokenizer.from_pretrained(model_dir)
# model = AutoModelForSequenceClassification.from_pretrained(model_dir)
# model.eval()
#
# # 3. Tokenize de nieuwe teksten
# def preprocess(texts):
#     return tokenizer(
#         texts,
#         padding="max_length",
#         truncation=True,
#         max_length=512,
#         return_tensors="pt"
#     )
#
# encodings = preprocess(df[text_col].tolist())
#
# # 4. Voorspel labels met softmax scores
# with torch.no_grad():
#     outputs = model(**encodings)
#     logits = outputs.logits
#     probs = F.softmax(logits, dim=-1).numpy()  # convert logits to probabilities
#     predictions = np.argmax(probs, axis=-1)
#
# # 5. Zet integer voorspellingen terug naar labels
# id2label = model.config.id2label
# predicted_labels = [id2label[i] for i in predictions]
#
# # 6. Voeg voorspellingen en scores toe aan dataframe
# df['predicted_label'] = predicted_labels
#
# # Voeg kolommen voor de scores per label toe
# for idx, label in id2label.items():
#     df[f'score_{label}'] = probs[:, idx]
#
# # 7. Optioneel: print gemiddelde score per label
# print("Gemiddelde scores per label:")
# for idx, label in id2label.items():
#     print(f"{label}: {probs[:, idx].mean():.4f}")
#
# # 8. Sla de resultaten op
# df.to_csv('data/test_data_classified.csv', index=False)
# print("Klassificatie klaar. Resultaten opgeslagen in 'test_data_classified.csv'.")
#
