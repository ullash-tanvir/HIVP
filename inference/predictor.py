import os
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertConfig
from functools import lru_cache

from models.architecture import EnhancedProtBert
from utils.preprocessing import preprocess_sequence
from config.settings import MODEL_ROOT, MODEL_SEED, DRUG_NAMES, TOKENIZER_NAME, MAX_SEQ_LENGTH

class ProteinPredictor:
    """
    Predictor class for HIV protease inhibitor resistance prediction
    """
    def __init__(self, model_root=MODEL_ROOT, seed=MODEL_SEED, device=None):
        """
        Initialize the predictor with models for all drugs
        
        Args:
            model_root (str): Directory containing model folders
            seed (int): Seed used for model training
            device (str): Device to run inference on ('cpu' or 'cuda')
        """
        self.model_root = model_root
        self.seed = seed
        # Determine device here instead of from settings
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.drug_names = DRUG_NAMES
        self.tokenizer = self.load_tokenizer()
        self.models = self.load_models()
        
    @staticmethod
    @lru_cache(maxsize=1)
    def load_tokenizer():
        """Load and cache the ProtBert tokenizer"""
        return BertTokenizer.from_pretrained(TOKENIZER_NAME, do_lower_case=False)
    
    def load_models(self):
        """
        Load models for all drugs
        
        Returns:
            dict: Dictionary mapping drug names to models
        """
        config = BertConfig.from_pretrained(TOKENIZER_NAME, num_labels=2)
        models = {}
        
        for name in self.drug_names:
            try:
                dir_path = os.path.join(self.model_root, name)
                params_file = os.path.join(dir_path, f"best_params_{self.seed}.txt")
                model_file = os.path.join(dir_path, f"best_model_{self.seed}.pt")
                
                # Load architecture parameters
                with open(params_file, 'r') as f:
                    arch = eval(f.read())["architecture"]
                
                # Initialize model with architecture
                model = EnhancedProtBert(config, arch)
                
                # Load model weights
                state = torch.load(model_file, map_location="cpu")
                model.load_state_dict(state, strict=False)
                model.eval()
                model.to(self.device)
                
                models[name] = model
                
            except Exception as e:
                print(f"Error loading model {name}: {e}")
        
        return models
    
    def preprocess_sequence(self, sequence):
        """
        Preprocess a sequence for model input
        
        Args:
            sequence (str): Amino acid sequence
            
        Returns:
            torch.Tensor: Preprocessed input tensor
        """
        seq = sequence.strip().upper()
        spaced = " ".join(list(seq))
        encoded = self.tokenizer.encode(spaced, add_special_tokens=True, truncation=True, max_length=MAX_SEQ_LENGTH)
        return torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(self.device)
    
    def predict_sequence(self, sequence):
        """
        Predict resistance for a single sequence across all drugs
        
        Args:
            sequence (str): Amino acid sequence
            
        Returns:
            dict: Predictions for each drug
        """
        input_ids = self.preprocess_sequence(sequence)
        results = {}
        
        for name, model in self.models.items():
            with torch.no_grad():
                logits = model(input_ids)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                pred = int(np.argmax(probs))
                results[name] = {"prediction": pred, "probability": float(probs[1])}
        
        return results
    
    def predict_batch(self, samples_dict):
        """
        Predict resistance for multiple sequences
        
        Args:
            samples_dict (dict): Dictionary mapping sample IDs to sequences
            
        Returns:
            pd.DataFrame: DataFrame with predictions for all samples and drugs
        """
        results = {}
        
        for sample_id, sequence in samples_dict.items():
            predictions = self.predict_sequence(sequence)
            results[sample_id] = {drug: data["prediction"] for drug, data in predictions.items()}
            
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(results, orient='index')
        
        # Make sure all drug columns exist
        for drug in self.drug_names:
            if drug not in df.columns:
                df[drug] = 0  # Default to 0 (susceptible) if missing
        
        # Reorder columns to match required output format
        df = df[self.drug_names]
        
        return df
    
    def format_results(self, results_df, format_type='table'):
        """
        Format prediction results
        
        Args:
            results_df (pd.DataFrame): DataFrame with prediction results
            format_type (str): Output format ('table' or 'csv')
            
        Returns:
            str: Formatted results
        """
        # Reset index to have sample_id as a column
        df_output = results_df.reset_index().rename(columns={'index': 'HIVP_Mutant'})
        
        if format_type == 'table':
            # Add header row with drug names
            header = f"**HIVP_Mutant {' '.join(self.drug_names)}**"
            rows = [header]
            
            # Add data rows
            for _, row in df_output.iterrows():
                row_str = f"{row['HIVP_Mutant']} {' '.join(map(str, row[self.drug_names].values))}"
                rows.append(row_str)
                
            return '\n'.join(rows)
            
        elif format_type == 'csv':
            return df_output.to_csv(index=False)
            
        else:
            return str(df_output)