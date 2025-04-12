import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset, load_dataset
import json
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the PubMedQA dataset
with open("pubmedqa_with_rationale.json", "r") as f:
    data = json.load(f)

examples = []
for example_id, entry in data.items():
    examples.append({
        "id": example_id,
        "question": entry["question"],
        "context": entry["context"],
        "label": entry["label"],
        "rationale": entry["rationale"],
    })

dataset = Dataset.from_list(examples)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Initialize the T5 tokenizer and model
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move the model to GPU if available
model = model.to(device)

def preprocess_function(examples):
    inputs = []
    targets = []
    for question, context, label in zip(examples["question"], examples["context"], examples["label"]):
        input_text = f"question: {question}, context: {context}. Answer yes, no, or maybe"
        inputs.append(input_text)
        targets.append(label)
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=10, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, output_dir="metrics_logs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.train_log_path = os.path.join(self.output_dir, "train_metrics.jsonl")
        self.eval_log_path = os.path.join(self.output_dir, "eval_metrics.jsonl")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        # Determine if it's training or evaluation step
        if "loss" in logs and "eval_loss" not in logs:
            with open(self.train_log_path, "a") as f:
                f.write(json.dumps({"step": state.global_step, **logs}) + "\n")
        elif "eval_loss" in logs:
            with open(self.eval_log_path, "a") as f:
                f.write(json.dumps({"step": state.global_step, **logs}) + "\n")

# Preprocess and tokenize the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./logit_checkpoints",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    weight_decay=0.01,
    save_total_limit=1,
    report_to="none",  # disable wandb or others unless you want logging
    fp16=True,         # <--- Enable mixed precision (faster training on modern GPUs)
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    callbacks=[MetricsLoggerCallback(output_dir="./metrics_logs_response")],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./response_result")
tokenizer.save_pretrained("./response_result")
