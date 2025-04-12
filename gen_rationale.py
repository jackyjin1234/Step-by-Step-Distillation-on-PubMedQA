import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
from datasets import load_dataset
import json
from tqdm import tqdm
import re

def detect_label_from_rationale(rationale_text):
    first_few_words = rationale_text.strip().lower()[:20]  # Look at first 20 characters
    
    match = re.search(r"\b(yes|no|maybe)\b", first_few_words)
    if match:
        return match.group(1)
    else:
        return "unknown"

class ForceStartTokensProcessor(LogitsProcessor):
    def __init__(self, tokenizer, allowed_start_words, prompt_length):
        self.tokenizer = tokenizer
        self.allowed_start_ids = [tokenizer(word, add_special_tokens=False).input_ids[0] for word in allowed_start_words]
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):
        # Only restrict at the first generation step (after prompt)
        if input_ids.shape[1] == self.prompt_length:
            mask = torch.full_like(scores, float('-inf'))  # Mask everything
            mask[:, self.allowed_start_ids] = 0  # Allow only specific start words
            scores = scores + mask
        return scores

# Define the generation function
def generate_rationale(question, context, max_input_length=512, max_new_tokens=64):
    # Construct a prompt with both question and context.
    # This prompt can be tuned as needed.
    prompt = (
        f"Question: {question}\n"
        f"Context: {context}\n"
        "First, answer with exactly one of 'yes', 'no', or 'maybe'.\n"
        "Then explain your reasoning based only on the context above.\n"
        "Answer:"
    )
    
    # print(prompt)

    # Tokenize the prompt; note that truncation is applied to avoid oversize inputs
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    input_length = inputs["input_ids"].shape[1]
    forced_start = ForceStartTokensProcessor(tokenizer, ["yes", "no", "maybe"], input_length)

    # Generate output with parameters tuned for quality reasoning; adjust temperature/sampling if needed
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.95,  # nucleus sampling
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            logits_processor=[forced_start],
        )
    
    # Decode the generated output, skipping special tokens
    output_text = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
    # print(output_text)
    label = detect_label_from_rationale(output_text)
    return output_text.strip(), label

def load_pubmedqa_data(file_path):
    """Load the PubMedQA dataset from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    # Load the model and tokenizer from the local storage directory.
    local_save_directory = "./local_model_storage/mmed-llama-3"
    tokenizer = AutoTokenizer.from_pretrained(local_save_directory)
    # For generation use a model suitable for causal language modeling:
    model = AutoModelForCausalLM.from_pretrained(local_save_directory, torch_dtype=torch.float16)
    
    # Move model to GPU if available and set to evaluation mode.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Load the PubMedQA dataset (using the "pqa_labeled" configuration).
    data_path = "ori_pqau.json"
    dataset = load_pubmedqa_data(data_path)

    keys = list(dataset.keys())[:800]
    dataset = {k: dataset[k] for k in keys}

    # Process each example and collect outputs in a list.
    # The PubMedQA context field is a list of dictionaries; we join the "contexts" lists.
    results = {}
    save_every = 200
    output_json_path = "pubmedqa_with_rationale.json"

    for idx, (pmid, example) in enumerate(tqdm(dataset.items(), desc="Processing examples")):
        question = example["QUESTION"]
        context_entries = " ".join(example["CONTEXTS"])
        
        # Generate rationale and label
        rationale, label = generate_rationale(question, context_entries)
        
        if label == "unknown":
            print(f"⚠️ Unknown label detected at {pmid}!")
            print(f"Rationale:\n{rationale}")
            print(f"Detected label: {label}")
            print("-" * 80)
            continue  # Skip saving this entry
        
        # Save to results if label is good
        results[pmid] = {
            "question": question,
            "context": context_entries,
            "rationale": rationale,
            "label": label,
        }
        
        # Every 100 examples, save the current progress
        if (idx + 1) % save_every == 0:
            print(f"💾 Saving checkpoint at {idx + 1} examples...")
            with open(output_json_path, "w") as f:
                json.dump(results, f, indent=2)

    # Final save after all done
    print("✅ Final save completed.")
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)
