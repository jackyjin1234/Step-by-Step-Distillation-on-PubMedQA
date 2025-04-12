# First-time loading from Hugging Face
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

if __name__ == "__main__":
    # Initial loading from Hugging Face
    model_name = "Henrychur/MMed-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    # Save model and tokenizer to a local directory
    local_save_directory = "./local_model_storage/mmed-llama-3"
    model.save_pretrained(local_save_directory)
    tokenizer.save_pretrained(local_save_directory)
    print(f"Model and tokenizer saved to {local_save_directory}")