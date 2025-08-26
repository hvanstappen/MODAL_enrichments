from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import evaluate
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import shutil

# Configuration
csv_training_file = "data/training_data.csv"  # Set the name of the CSV file with training data
model_name = "GroNLP/bert-base-dutch-cased"  # Set the model name here
finetuned_model_name = "bert-base-dutch-cased_finetuned"  # Choose a name for the fine tuned model


def load_and_validate_data(csv_file, text_col="text_short", label_col="label"):
    """Load and validate the training dataset."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    print("Columns in dataset:", df.columns.tolist())

    # Check for required columns
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Expect '{text_col}' and '{label_col}' in CSV.")

    # Check for missing values
    if df[text_col].isnull().any():
        print(f"Warning: {df[text_col].isnull().sum()} missing values in '{text_col}' column")
        df = df.dropna(subset=[text_col])

    if df[label_col].isnull().any():
        print(f"Warning: {df[label_col].isnull().sum()} missing values in '{label_col}' column")
        df = df.dropna(subset=[label_col])

    print(f"Dataset size after cleaning: {len(df)} samples")
    return df


def create_label_mappings(df, label_col):
    """Create label mappings and validate labels."""
    unique_labels = sorted(df[label_col].unique())

    if len(unique_labels) < 2:
        raise ValueError("Dataset must contain at least 2 different labels for classification")

    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    print("Gevonden labels:", unique_labels)
    print("label2id mapping:", label2id)

    return label2id, id2label, unique_labels


def preprocess_function(examples, tokenizer, text_col, max_length=512):
    """Tokenize the texts and return a dictionary with the required fields."""
    tokenized = tokenizer(
        examples[text_col],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=None
    )

    # Ensure labels are in the correct format
    tokenized["labels"] = examples["labels"]
    return tokenized


def main():
    # Load and validate data
    df = load_and_validate_data(csv_training_file)

    # Create label mappings
    text_col = "text_short"
    label_col = "label"
    label2id, id2label, unique_labels = create_label_mappings(df, label_col)

    # Map labels to integers
    df["labels"] = df[label_col].map(label2id)

    # Verify all labels were mapped successfully
    if df["labels"].isnull().any():
        unmapped_labels = df[df["labels"].isnull()][label_col].unique()
        raise ValueError(f"Some labels could not be mapped: {unmapped_labels}")

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Improved train/validation/test split with better validation
    if len(dataset) < 10:
        raise ValueError("Dataset too small for proper train/validation/test split")

    # Alternative stratified split using sklearn for more reliable stratification
    from sklearn.model_selection import train_test_split

    # Get indices for stratified split
    indices = list(range(len(dataset)))
    labels_for_split = df["labels"].values

    # First split: 70% train, 30% temp (which will be split into val and test)
    train_indices, temp_indices = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=labels_for_split
    )

    # Second split: 15% validation, 15% test from the 30% temp
    temp_labels = labels_for_split[temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=42, stratify=temp_labels
    )

    # Create dataset splits using indices
    train_dataset = dataset.select(train_indices)
    eval_dataset = dataset.select(val_indices)
    test_dataset = dataset.select(test_indices)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Apply preprocessing with remove_columns to avoid duplicate columns
    def preprocess_with_tokenizer(examples):
        return preprocess_function(examples, tokenizer, text_col)

    train_dataset = train_dataset.map(
        preprocess_with_tokenizer,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        preprocess_with_tokenizer,
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    test_dataset = test_dataset.map(
        preprocess_with_tokenizer,
        batched=True,
        remove_columns=test_dataset.column_names
    )

    print("Columns after tokenization:", train_dataset.column_names)

    # Load Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id
    )

    # Create output directory
    output_dir = "finetuned_model"
    os.makedirs(output_dir, exist_ok=True)

    # TrainingArguments with improved settings
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=5, # change up to 4 or 5
        weight_decay=0.01,
        warmup_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        save_total_limit=1,  # Only keep 1 best checkpoint during training
        seed=42,
        # Remove use_cpu parameter - let it auto-detect GPU availability
        use_cpu=True,  # Uncomment this line to force CPU usage
    )

    # Load metrics
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)

        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)

        # Add additional metrics
        from sklearn.metrics import precision_recall_fscore_support, classification_report

        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

        return {
            "accuracy": accuracy["accuracy"],
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,  # Add tokenizer to trainer
    )

    # Training
    print("Start training...")
    train_result = trainer.train()

    # Print training summary
    print("Training completed!")

    # Evaluation on validation set
    print("Start evaluation on validation set...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Test set predictions
    print("Evaluating on test set...")
    predictions = trainer.predict(test_dataset)
    predicted_labels = predictions.predictions.argmax(-1)
    true_labels = predictions.label_ids

    print("Sample predictions vs true labels:")
    for i in range(min(10, len(predicted_labels))):
        pred_label = id2label[predicted_labels[i]]
        true_label = id2label[true_labels[i]]
        print(f"  Predicted: {pred_label}, True: {true_label}")

    # Calculate test metrics
    test_metrics = compute_metrics((predictions.predictions, predictions.label_ids))
    print(f"Test metrics: {test_metrics}")

    # Create the final model folder and clean up any existing content
    finetuned_model_folder = "./" + finetuned_model_name

    # Remove existing model folder if it exists to ensure clean save
    import shutil
    if os.path.exists(finetuned_model_folder):
        shutil.rmtree(finetuned_model_folder)
    os.makedirs(finetuned_model_folder)

    # Save the best model to the final location
    trainer.save_model(finetuned_model_folder)
    tokenizer.save_pretrained(finetuned_model_folder)

    # Save label mappings for future use
    import json
    with open(os.path.join(finetuned_model_folder, "label_mappings.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, indent=2)

    # Clean up all training checkpoints from the output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Cleaned up training checkpoints from: {output_dir}")

    print(f"Best model saved to: {finetuned_model_folder}")
    print("All training checkpoints have been removed.")
    print("Training completed successfully!")

    # Print dataset format for verification
    print("Dataset format after preprocessing:", train_dataset.features)

    return trainer, test_metrics


if __name__ == "__main__":
    trainer, test_metrics = main()