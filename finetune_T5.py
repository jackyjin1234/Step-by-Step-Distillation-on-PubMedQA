import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the PubMedQA dataset
dataset = load_dataset("pubmed_qa", "pqa_labeled")
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

# Initialize the T5 tokenizer and model
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move the model to GPU if available
model = model.to(device)

def preprocess_function(examples):
    inputs = []
    targets = []
    for question, context, label in zip(examples["question"], examples["context"], examples["final_decision"]):
        input_text = f"question: {question}, context: {context}. Answer yes, no, or maybe"
        inputs.append(input_text)
        targets.append(label)
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=10, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess and tokenize the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
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
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./finetuned_t5_pubmedqa")
tokenizer.save_pretrained("./finetuned_t5_pubmedqa")
