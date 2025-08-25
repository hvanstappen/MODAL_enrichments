from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import evaluate
import numpy as np
import pandas as pd

# Load training dataset
df = pd.read_csv('data/email_dataset.csv')

print("Kolommen in de dataset:", df.columns)

# Check columns
text_col = "text_short"  # edit if column name is different
label_col = "label"      # edit if column name is different

if text_col not in df.columns or label_col not in df.columns:
    raise ValueError(f"Verwacht kolommen '{text_col}' en '{label_col}' in de CSV.")

# Automatic label mapping
unique_labels = sorted(df[label_col].unique())
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

print("Gevonden labels:", unique_labels)
print("label2id mapping:", label2id)

# Map labels to integers
df["labels"] = df[label_col].map(label2id)

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Train/valid/test splitting
split_dataset = dataset.train_test_split(test_size=0.2)
split_validation = split_dataset['test'].train_test_split(test_size=0.5)

train_dataset = split_dataset['train']
eval_dataset = split_validation['train']
test_dataset = split_validation['test']

# Tokenize
model_name = "wietsedv/bert-base-dutch-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    # Tokenize the texts and return a dictionary with the required fields
    tokenized = tokenizer(
        examples[text_col],
        padding="max_length",
        truncation=True,
        max_length=512,  # Add explicit max_length
        return_tensors=None  # Ensure we don't get PyTorch/TF tensors here
    )

    # Make sure we have the labels in the correct format
    tokenized["labels"] = examples["labels"]
    return tokenized

# Apply preprocessing with remove_columns to avoid duplicate columns
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names  # Remove original columns
)
eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=eval_dataset.column_names
)
test_dataset = test_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=test_dataset.column_names
)

print("Kolommen na tokenization:", train_dataset.column_names)

# Load Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)

# TrainingArguments
training_args = TrainingArguments(
    output_dir="finetuned_model",
    eval_strategy="epoch",       # als je oudere transformers hebt, kan 'evaluation_strategy' nodig zijn
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    no_cuda=True,  # CPU-forcing, zet op False voor GPU
)

# Metrics
metric = evaluate.load("accuracy")

def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Training
print("Training start...")
trainer.train()

# Evaluation
print("Evaluatie start...")
eval_results = trainer.evaluate()
print(f"Evaluatie resultaten: {eval_results}")

# Testset predictions
predictions = trainer.predict(test_dataset)
print("Voorspellingen:", predictions.predictions.argmax(-1))
print("Labels:", predictions.label_ids)

test_metrics = compute_metrics((predictions.predictions, predictions.label_ids))
print(f"Test metrics: {test_metrics}")

# Save Model
best_checkpoint = trainer.state.best_model_checkpoint
if best_checkpoint:
    model.save_pretrained("./bert-base-dutch-cased_finetuned")
    tokenizer.save_pretrained("./bert-base-dutch-cased_finetuned")
    print(f"Beste model opgeslagen in: {best_checkpoint}")
else:
    print("Geen beste model gevonden, controleer de checkpoints.")

# Verify the dataset format
print("Dataset format after preprocessing:", train_dataset.features)