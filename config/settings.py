import os

# Models directory
MODEL_ROOT = os.environ.get("MODEL_ROOT", "model_weights")

# Model seed
MODEL_SEED = int(os.environ.get("MODEL_SEED", 42))

# Drug names (HIV protease inhibitors)
DRUG_NAMES = ["LPV", "IDV", "NFV", "SQV", "APV"]

# Maximum sequence length
MAX_SEQ_LENGTH = 512

# Device configuration - don't import torch here to avoid circular imports
DEVICE = "cuda" if os.environ.get("USE_CPU", "0") != "1" else "cpu"

# Tokenizer model name
TOKENIZER_NAME = "Rostlab/prot_bert"

# Output format
OUTPUT_FORMAT = "table"  # 'table' or 'csv'