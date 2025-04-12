import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Trainer, TrainerCallback, GenerationConfig
from datasets import Dataset, load_dataset
import json
import os
import numpy as np

class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, output_dir="metrics_logs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.train_log_path = os.path.join(self.output_dir, "train_metrics.jsonl")
        self.eval_log_path = os.path.join(self.output_dir, "eval_metrics.jsonl")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        record = {"step": state.global_step}

        # Log training loss if present
        if "loss" in logs:
            record["train_loss"] = logs["loss"]
            with open(self.train_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        # Log eval loss and accuracy if present
        if "eval_loss" in logs:
            record["eval_loss"] = logs["eval_loss"]
            if "eval_accuracy" in logs:
                record["eval_accuracy"] = logs["eval_accuracy"]
            with open(self.eval_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    # print("Shape of predictions:", predictions.shape)
    # print("Shape of labels:", labels.shape)

    # print("First prediction:", predictions[0])
    # print("First label:", labels[0])
    # predictions, labels = eval_pred
    # print(predictions)
    # exit(0)
    # if isinstance(predictions, (list, tuple)):
    #     predictions = predictions[0]

    # if isinstance(predictions, tuple):
    #     predictions = predictions[0]

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels (special ignore index) with pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip and lower case
    decoded_preds = [pred.strip().lower() for pred in decoded_preds]
    decoded_labels = [label.strip().lower() for label in decoded_labels]

    # Compute exact match
    acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

    return {"accuracy": acc}

# 1. Load your data
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
dataset = dataset.train_test_split(test_size=0.4, seed=42)

# 2. Load tokenizer and model
model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 3. Preprocessing: prepare BOTH prediction and explanation inputs
def preprocess_function(examples):
    question_context = [q + " [SEP] " + c for q, c in zip(examples["question"], examples["context"])]
    
    pred_targets = [l.lower() for l in examples["label"]]
    expl_targets = [r for r in examples["rationale"]]

    pred_inputs = tokenizer(
        ["predict: " + txt for txt in question_context],
        max_length=512,
        padding="max_length",
        truncation=True
    )
    pred_labels = tokenizer(
        pred_targets,
        max_length=5,
        padding="max_length",
        truncation=True
    )

    expl_inputs = tokenizer(
        ["explain: " + txt for txt in question_context],
        max_length=512,
        padding="max_length",
        truncation=True
    )
    expl_labels = tokenizer(
        expl_targets,
        max_length=128,
        padding="max_length",
        truncation=True
    )

    return {
        "pred_input_ids": pred_inputs["input_ids"],
        "pred_attention_mask": pred_inputs["attention_mask"],
        "pred_labels": pred_labels["input_ids"],
        "expl_input_ids": expl_inputs["input_ids"],
        "expl_attention_mask": expl_inputs["attention_mask"],
        "expl_labels": expl_labels["input_ids"],
    }

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
tokenized_datasets.set_format("torch")

# 4. Custom Data Collator
class TwoTaskDataCollator:
    def __call__(self, features):
        batch = {
            "pred": {
                "input_ids": torch.stack([f["pred_input_ids"] for f in features]),
                "attention_mask": torch.stack([f["pred_attention_mask"] for f in features]),
                "labels": torch.stack([f["pred_labels"] for f in features]),
            },
            "expl": {
                "input_ids": torch.stack([f["expl_input_ids"] for f in features]),
                "attention_mask": torch.stack([f["expl_attention_mask"] for f in features]),
                "labels": torch.stack([f["expl_labels"] for f in features]),
            }
        }
        return batch

# 5. Custom Trainer
class TwoTaskTrainer(Trainer):
    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha  # weight for prediction loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if "pred" in inputs and "expl" in inputs:
            pred_outputs = model(**inputs["pred"])
            expl_outputs = model(**inputs["expl"])

            loss = self.alpha * pred_outputs.loss + (1 - self.alpha) * expl_outputs.loss

            if return_outputs:
                return (loss, {"pred_outputs": pred_outputs, "expl_outputs": expl_outputs})
            else:
                return loss
        else:
            # Evaluation mode: only pred inputs (flat)
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, (outputs.logits, inputs.get("labels")))
            # return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        """ Custom prediction_step to handle pred + expl inputs. """
        return super().prediction_step(
            model,
            inputs["pred"],  # <=== Only use pred branch during evaluation
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
        )

# 6. TrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./step_checkpoints",
    evaluation_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    weight_decay=0.01,
    predict_with_generate=True,
    save_total_limit=1,
    logging_steps=10,
    fp16=True,
    remove_unused_columns=False
)

model.generation_config = GenerationConfig(
    temperature=0.3,
    do_sample=True,
    top_p=0.9,
    max_new_tokens=64
)

# 7. Trainer
trainer = TwoTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=TwoTaskDataCollator(),
    alpha=0.3,  # 0.5 for balanced loss
    compute_metrics=compute_metrics,
    callbacks=[MetricsLoggerCallback(output_dir="./metrics_logs_step")],
)

# 8. Fine-tune!
trainer.train()

# 9. Save model
model.save_pretrained("./step_result")
tokenizer.save_pretrained("./step_result")
