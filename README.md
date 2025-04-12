# Step-by-Step-Distillation-on-PubMedQA

1. Download.py: download MMed-Llama 3
2. gen_retionale.py: Generate rationale data from PubMedQA unlabeled dataset
3. Distill_logit.py: response-based distillation on generated dataset
4. Distill_step.py: step by step distillation on generated dataset
5. finetune_t5.py: finetune t5 on PubMedQA labeled dataset
6. eval.py: evaluate on PubMedQA labeled dataset
