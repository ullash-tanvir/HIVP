import torch
import re

def preprocess_sequence(seq, tokenizer, max_length=512):
    """
    Preprocess a protein sequence for input to ProtBert.
    
    Args:
        seq (str): Amino acid sequence
        tokenizer: The ProtBert tokenizer
        max_length (int): Maximum sequence length
        
    Returns:
        torch.Tensor: Tokenized sequence as tensor
    """
    # Clean the sequence - strip whitespace and convert to uppercase
    seq = seq.strip().upper()
    
    # Add spaces between amino acids
    spaced = " ".join(list(seq))
    
    # Encode the sequence
    encoded = tokenizer.encode(
        spaced, 
        add_special_tokens=True, 
        truncation=True, 
        max_length=max_length
    )
    
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

def parse_input_samples(input_text):
    """
    Parse multiple input samples from a text format.
    
    Expected format:
    SAMPLE_ID SEQUENCE
    SAMPLE_ID SEQUENCE
    ...
    
    Args:
        input_text (str): Input text with multiple samples
        
    Returns:
        dict: Dictionary mapping sample IDs to sequences
    """
    samples = {}
    lines = input_text.strip().split('\n')
    
    for line in lines:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            sample_id, sequence = parts
            samples[sample_id] = sequence
    
    return samples

def extract_drug_from_sample_id(sample_id):
    """
    Extract drug name from sample ID if present.
    Example: "88_APV" -> "APV", "NFV_P302" -> "NFV"
    
    Args:
        sample_id (str): Sample identifier
        
    Returns:
        str or None: Extracted drug name if found, None otherwise
    """
    # Common drug names in HIV protease inhibitor studies
    drug_names = ["APV", "IDV", "LPV", "NFV", "SQV"]
    
    for drug in drug_names:
        if drug in sample_id:
            return drug
    
    return None