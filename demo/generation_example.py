from models import MLPMemory, MistralMLPModel

import transformers
from transformers import AutoModelForCausalLM, AutoConfig
from loguru import logger

"""
generation_example.py

Usage:
    Run this demo with:
        python -m demo.generation_example
"""

# ================================
# ⚙️ Model paths configuration
# ================================
base_lm_path = "/path/to/models/Mistral-7B-v0.3"
# MLP Memory can be downloaded in https://huggingface.co/Rubin-Wei/MLPMemory-Mistral-wikipedia
knn_generator_path = "/path/to/mlp/memory"

# ================================
# 🧠 Load tokenizer & models
# ================================
logger.info("Loading tokenizer and base LM...")
tokenizer = transformers.AutoTokenizer.from_pretrained(base_lm_path)
base_lm = AutoModelForCausalLM.from_pretrained(base_lm_path)

logger.info("Loading KNN generator...")
config = AutoConfig.from_pretrained(knn_generator_path)
knn_generator = MistralMLPModel.from_pretrained(knn_generator_path, config=config, input_dim=config.hidden_size, output_dim=config.hidden_size)

base_lm.resize_token_embeddings(len(tokenizer))
base_lm.eval()
knn_generator.eval()

# ================================
# 🧩 Combine into MLP Memory model
# ================================
joint = MLPMemory(base_lm, knn_generator, lmbda=0.75, knn_temp=1.0).to("cuda")

# ================================
# 🧾 Inference example
# ================================
prompt = f"Answer the questions:\n\nQuestion: who sings i can't take my eyes off of you?? The answer is:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

out_ids = joint.generate(
    **inputs,
    max_new_tokens=20,
    do_sample=False
)
logger.info(f"MLP Memory output: {tokenizer.decode(out_ids[0], skip_special_tokens=True)}")
# Expected output: Answer the questions:\n\nQuestion: who sings i can't take my eyes off of you?? The answer is: Frankie Valli. ;) \n\nQuestion: who sings i can't take my eyes

out_ids = base_lm.generate(
    **inputs,
    max_new_tokens=20,
    do_sample=False
)
logger.info(f"Base Model output: {tokenizer.decode(out_ids[0], skip_special_tokens=True)}")
# Expected output: Answer the questions:\n\nQuestion: who sings i can't take my eyes off of you?? The answer is: Andy Williams \n\nQuestion: who sings the song "I'm a believer"??

