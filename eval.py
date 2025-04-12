import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import random
import numpy as np

# 1. Load dataset
dataset = load_dataset("pubmed_qa", "pqa_labeled")["train"]

# 2. Load tokenizers and models
response_tokenizer = AutoTokenizer.from_pretrained("./response_result")
response_model = AutoModelForSeq2SeqLM.from_pretrained("./response_result").to("cuda" if torch.cuda.is_available() else "cpu")

step_tokenizer = AutoTokenizer.from_pretrained("./step_result")
step_model = AutoModelForSeq2SeqLM.from_pretrained("./step_result").to("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Sample 100 items
sampled_dataset = dataset.shuffle(seed=42).select(range(100))

# 4. Preprocessing function (same as before)
def preprocess_function(examples, tokenizer):
    inputs = []
    for question, context in zip(examples["question"], examples["context"]):
        input_text = f"question: {question}, context: {context}. Answer yes, no, or maybe"
        inputs.append(input_text)
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    return model_inputs

# 5. Prediction function
def generate_predictions(model, tokenizer, inputs):
    with torch.no_grad():
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10,
            do_sample=False,
            decoder_start_token_id=model.config.decoder_start_token_id if model.config.decoder_start_token_id is not None else tokenizer.pad_token_id,
        )

        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        preds = [pred.strip().lower() for pred in preds]
    return preds

# 6. Prepare inputs for both models
inputs_response = preprocess_function(sampled_dataset, response_tokenizer)
inputs_step = preprocess_function(sampled_dataset, step_tokenizer)

# 7. Make predictions
response_preds = generate_predictions(response_model, response_tokenizer, inputs_response)
step_preds = generate_predictions(step_model, step_tokenizer, inputs_step)

# 8. Get ground truth
labels = [label.strip().lower() for label in sampled_dataset["final_decision"]]

# 9. Calculate accuracy
def compute_accuracy(preds, labels):
    # print(preds[:3])
    # preds = [pred.split()[0] for pred in preds]  # Only take the first token like "yes" or "no"
    correct = sum([p == l for p, l in zip(preds, labels)])
    return correct / len(labels)

response_accuracy = compute_accuracy(response_preds, labels)
step_accuracy = compute_accuracy(step_preds, labels)

# 10. Print results
print(f"✅ Response-based model accuracy: {response_accuracy:.4f}")
print(f"✅ Step-by-step model accuracy: {step_accuracy:.4f}")
