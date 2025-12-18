import streamlit as st
import os
import torch
import numpy as np
from transformers import BertTokenizer, BertModel, BertConfig
import matplotlib.pyplot as plt
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Try to import seaborn, use matplotlib if not available
try:
    import seaborn as sns
    USE_SEABORN = True
except ImportError:
    USE_SEABORN = False
    st.warning("Seaborn not found. Install with: pip install seaborn")

# ------------------ Model Classes ------------------
class AttentionLayer(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        weights = self.attention(x)
        output = torch.sum(weights * x, dim=1)
        return output, weights

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, activation, dropout_rate):
        super().__init__()
        activations = {
            'relu': torch.nn.ReLU(), 'leaky_relu': torch.nn.LeakyReLU(),
            'gelu': torch.nn.GELU(), 'selu': torch.nn.SELU(), 'elu': torch.nn.ELU()
        }
        self.activation = activations[activation]
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.BatchNorm1d(out_dim),
            self.activation,
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(out_dim, out_dim),
            torch.nn.BatchNorm1d(out_dim),
        )
        self.shortcut = torch.nn.Linear(in_dim, out_dim) if in_dim != out_dim else torch.nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.layers(x)
        return self.activation(out + identity)

class EnhancedProtBert(torch.nn.Module):
    def __init__(self, config, arch_params):
        super().__init__()
        self.bert = BertModel.from_pretrained("Rostlab/prot_bert")
        for p in self.bert.parameters(): p.requires_grad = False
        self.hidden_dims = arch_params['hidden_dims']
        self.dropout_rate = arch_params['dropout_rate']
        self.activation = arch_params['activation']
        self.use_attention = arch_params['use_attention']
        self.num_heads = arch_params.get('num_heads', 1)
        
        if self.use_attention:
            self.attention_heads = torch.nn.ModuleList([
                torch.nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=self.num_heads,
                                             dropout=self.dropout_rate)
                for _ in range(self.num_heads)
            ])
            self.att_pool = AttentionLayer(config.hidden_size)

        layers = []
        input_dim = config.hidden_size * (self.num_heads if self.use_attention else 1)
        for h in self.hidden_dims:
            layers.append(ResidualBlock(input_dim, h, self.activation, self.dropout_rate))
            input_dim = h
        self.feature_layers = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Linear(self.hidden_dims[-1], config.num_labels)

    def forward(self, input_ids):
        with torch.no_grad():
            seq_out = self.bert(input_ids).last_hidden_state
        if self.use_attention:
            heads = []
            for att in self.attention_heads:
                perm = seq_out.permute(1,0,2)
                att_out, _ = att(perm, perm, perm)
                heads.append(att_out.permute(1,0,2))
            seq_out = torch.cat(heads, dim=2) if len(heads)>1 else heads[0]
            pooled, _ = self.att_pool(seq_out)
        else:
            pooled = seq_out.mean(dim=1)
        feats = self.feature_layers(pooled)
        return self.classifier(feats)

# ------------------ Loading Tokenizer & Models ------------------
@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

@st.cache_resource
def load_models(model_root="model_weights", seed=42):
    config = BertConfig.from_pretrained("Rostlab/prot_bert", num_labels=2)
    tokenizer = load_tokenizer()
    models = {}
    for name in ["APV","IDV","LPV","NFV","SQV"]:
        dir_path = os.path.join(model_root, name)
        params_file = os.path.join(dir_path, f"best_params_{seed}.txt")
        model_file = os.path.join(dir_path, f"best_model_{seed}.pt")
        with open(params_file) as f:
            arch = eval(f.read())["architecture"]
        model = EnhancedProtBert(config, arch)
        state = torch.load(model_file, map_location="cpu")
        model.load_state_dict(state, strict=False)
        model.eval()
        models[name] = model
    return models

# ------------------ FASTA Parser ------------------
def parse_fasta(fasta_text):
    """
    Parse FASTA format text and return a dictionary of {id: sequence}
    """
    sequences = {}
    current_id = None
    current_seq = []
    
    lines = fasta_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            # Save previous sequence if exists
            if current_id is not None:
                sequences[current_id] = ''.join(current_seq)
            # Start new sequence
            current_id = line[1:]  # Remove '>' character
            current_seq = []
        elif line:
            # Add to current sequence
            current_seq.append(line)
    
    # Don't forget the last sequence
    if current_id is not None:
        sequences[current_id] = ''.join(current_seq)
    
    return sequences

def is_fasta_format(text):
    """
    Check if the input text is in FASTA format
    """
    text = text.strip()
    return text.startswith('>')

# ------------------ Preprocess & Predict ------------------

def preprocess(seq, tokenizer):
    seq = seq.strip().upper()
    spaced = " ".join(list(seq))
    encoded = tokenizer.encode(spaced, add_special_tokens=True, truncation=True, max_length=512)
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

@st.cache_data
def predict_sequence(seq, _models, _tokenizer):
    input_ids = preprocess(seq, _tokenizer)
    results = {}
    for name, model in _models.items():
        with torch.no_grad():
            logits = model(input_ids)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred = int(np.argmax(probs))
            results[name] = {"prediction": pred, "probability": float(probs[1])}
    return results

def predict_multiple_sequences(sequences_dict, _models, _tokenizer):
    """
    Predict for multiple sequences and return results in a structured format
    """
    all_results = []
    
    for seq_id, sequence in sequences_dict.items():
        results = predict_sequence(sequence, _models, _tokenizer)
        
        # Create a row for each sequence
        row = {"Sequence_ID": seq_id}
        for model_name, result in results.items():
            row[f"{model_name}_Prediction"] = result["prediction"]
            # row[f"{model_name}_Probability"] = round(result["probability"], 4)
        
        all_results.append(row)
    
    return pd.DataFrame(all_results)

# ------------------ Streamlit App ------------------

def main():
    # Load models and tokenizer
    models = load_models()
    tokenizer = load_tokenizer()
    st.title("Protein Sequence Classification Ensemble")
    st.markdown("### Input Options:")
    st.markdown("- **Single sequence**: Enter amino acid sequence directly")
    st.markdown("- **FASTA format**: Enter sequences in FASTA format with >ID headers")
    
    seq_input = st.text_area(
        "Enter amino acid sequence(s):",
        height=200,
        placeholder="""Example single sequence:
MELANIN

Example FASTA format:
>88_APV
PQITLWQRPIVPIRIEGQLTEALLDTGADDTVLEEINLSGRWKPKMIGGIGGFVKVRQYDQVTIEICDHKVIGTVLVGPTPANIIGRNLLTQLGCTLNF
>NFV_P302
PQITLWQRPFVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKIIGGVGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"""
    )
    
    if st.button("Predict"):
        if not seq_input:
            st.warning("Please enter at least one amino acid sequence.")
            return
        
        # Check if input is FASTA format or single sequence
        if is_fasta_format(seq_input):
            st.info("FASTA format detected. Processing multiple sequences...")
            sequences = parse_fasta(seq_input)
            
            if not sequences:
                st.error("No valid sequences found in FASTA input.")
                return
            
            st.success(f"Found {len(sequences)} sequences")
            
            # Show parsed sequences
            # with st.expander("Parsed Sequences"):
            #     for seq_id, seq in sequences.items():
            #         st.write(f"**{seq_id}**: {seq[:50]}{'...' if len(seq) > 50 else ''}")
            
            # Predict for all sequences
            results_df = predict_multiple_sequences(sequences, models, tokenizer)
            
            st.markdown("### Prediction Results")
            
            # Display results table
            st.dataframe(results_df.set_index("Sequence_ID"), use_container_width=True)
            
            # Create a summary visualization
        #     st.markdown("### Summary Visualization")
            
        #     # Prepare data for visualization
        #     viz_data = []
        #     for _, row in results_df.iterrows():
        #         seq_id = row["Sequence_ID"]
        #         for model in ["APV", "IDV", "LPV", "NFV", "SQV"]:
        #             viz_data.append({
        #                 "Sequence_ID": seq_id,
        #                 "Model": model,
        #                 "Prediction": row[f"{model}_Prediction"],
        #                 "Probability": row[f"{model}_Probability"]
        #             })
            
        #     viz_df = pd.DataFrame(viz_data)
            
        #     # Create two columns for charts
        #     col1, col2 = st.columns(2)
            
        #     with col1:
        #         st.markdown("#### Predictions by Model")
        #         pred_summary = viz_df.groupby(['Model', 'Prediction']).size().reset_index(name='Count')
        #         fig, ax = plt.subplots(figsize=(8, 6))
                
        #         if USE_SEABORN:
        #             sns.barplot(data=pred_summary, x='Model', y='Count', hue='Prediction', ax=ax)
        #         else:
        #             # Fallback to matplotlib
        #             models = pred_summary['Model'].unique()
        #             predictions = pred_summary['Prediction'].unique()
        #             width = 0.35
        #             x = np.arange(len(models))
                    
        #             for i, pred in enumerate(predictions):
        #                 data = pred_summary[pred_summary['Prediction'] == pred]
        #                 counts = [data[data['Model'] == model]['Count'].iloc[0] if len(data[data['Model'] == model]) > 0 else 0 for model in models]
        #                 ax.bar(x + i*width, counts, width, label=f'Prediction {pred}')
                    
        #             ax.set_xlabel('Model')
        #             ax.set_xticks(x + width/2)
        #             ax.set_xticklabels(models)
        #             ax.legend()
                
        #         ax.set_title('Prediction Distribution by Model')
        #         ax.set_ylabel('Number of Sequences')
        #         st.pyplot(fig)
            
        #     with col2:
        #         st.markdown("#### Average Probabilities")
        #         avg_probs = viz_df.groupby('Model')['Probability'].mean().reset_index()
        #         fig, ax = plt.subplots(figsize=(8, 6))
                
        #         if USE_SEABORN:
        #             sns.barplot(data=avg_probs, x='Model', y='Probability', ax=ax)
        #         else:
        #             # Fallback to matplotlib
        #             ax.bar(avg_probs['Model'], avg_probs['Probability'])
                
        #         ax.set_title('Average Positive Class Probability by Model')
        #         ax.set_ylabel('Average Probability')
        #         ax.set_ylim(0, 1)
        #         st.pyplot(fig)
            
        # else:
        #     # Single sequence prediction (original functionality)
        #     st.info("Single sequence detected. Processing...")
        #     res = predict_sequence(seq_input, models, tokenizer)
        #     df = pd.DataFrame([
        #         {"Model": k, "Prediction": v["prediction"], "Pos_Probability": v["probability"]}
        #         for k,v in res.items()
        #     ])
        #     st.markdown("### Prediction Results")
        #     st.table(df.set_index("Model"))

if __name__ == "__main__":
    main()