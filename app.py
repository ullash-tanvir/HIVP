#!/usr/bin/env python3
"""
Streamlit web application for HIV protease inhibitor resistance prediction.
This is a modular version that works with GPU.
"""

import streamlit as st
import os
import warnings

# Fix for torch._classes error
warnings.filterwarnings("ignore", message=".*torch._classes.*")
os.environ["STREAMLIT_WATCHDOG_DISABLE"] = "true"

# Page configuration
st.set_page_config(
    page_title="HIV Protease Inhibitor Resistance Prediction",
    page_icon="🧬",
    layout="wide"
)

# Import modules (only when needed)
def import_modules():
    import pandas as pd
    import torch
    import numpy as np
    from utils.preprocessing import parse_input_samples
    from inference.predictor import ProteinPredictor
    from config.settings import MODEL_ROOT, MODEL_SEED, DRUG_NAMES
    
    return {
        'pd': pd,
        'torch': torch,
        'np': np,
        'parse_input_samples': parse_input_samples,
        'ProteinPredictor': ProteinPredictor,
        'MODEL_ROOT': MODEL_ROOT,
        'MODEL_SEED': MODEL_SEED,
        'DRUG_NAMES': DRUG_NAMES
    }

# Initialize predictor only once
@st.cache_resource
def get_predictor():
    modules = import_modules()
    return modules['ProteinPredictor'](
        model_root=modules['MODEL_ROOT'],
        seed=modules['MODEL_SEED'],
        device="cuda" if modules['torch'].cuda.is_available() else "cpu"
    )

def parse_fasta(input_text):
    """
    Parse multiple input samples from FASTA-like format.
    
    Expected format:
    >SAMPLE1
    SEQUENCE1
    >SAMPLE2
    SEQUENCE2
    ...
    
    Args:
        input_text (str): Input text with multiple samples
        
    Returns:
        dict: Dictionary mapping sample IDs to sequences
    """
    samples = {}
    lines = input_text.strip().split('\n')
    
    current_id = None
    current_seq = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('>'):
            # If we have a previous sequence, add it to samples
            if current_id is not None and current_seq:
                samples[current_id] = ''.join(current_seq)
                current_seq = []
                
            # Set the new ID (remove '>' prefix)
            current_id = line[1:].strip()
        else:
            # Add this line to the current sequence
            if current_id is not None:
                current_seq.append(line)
    
    # Add the last sequence if there is one
    if current_id is not None and current_seq:
        samples[current_id] = ''.join(current_seq)
    
    return samples

def is_fasta_format(text):
    """
    Check if the input text is in FASTA format
    """
    text = text.strip()
    return text.startswith('>')

def render_header():
    """Render the application header"""
    st.title("HIV Protease Inhibitor Resistance Prediction")
    st.markdown("""
    This application predicts HIV protease inhibitor resistance based on amino acid sequences.
    You can input a single sequence or multiple sequences in FASTA format.
    """)
    st.markdown("### Input Options:")
    st.markdown("- **Single sequence**: Enter amino acid sequence directly")
    st.markdown("- **FASTA format**: Enter sequences in FASTA format with >ID headers")

def main():
    """Main application function"""
    try:
        render_header()
        
        # Single input area for both modes
        seq_input = st.text_area(
            "Enter amino acid sequence(s):",
            height=200,
            placeholder="""Example single sequence:
PQITLWQRPIVPIRIEGQLTEALLDTGADDTVLEEINLSGRWKPKMIGGIGGFVKVRQYDQVTIEICDHKVIGTVLVGPTPANIIGRNLLTQLGCTLNF

Example FASTA format:
>88_APV
PQITLWQRPIVPIRIEGQLTEALLDTGADDTVLEEINLSGRWKPKMIGGIGGFVKVRQYDQVTIEICDHKVIGTVLVGPTPANIIGRNLLTQLGCTLNF
>NFV_P302
PQITLWQRPFVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKIIGGVGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF""",
            help="Input a single sequence or multiple sequences in FASTA format (>ID followed by sequence)"
        )
        
        if st.button("Predict"):
            if not seq_input.strip():
                st.warning("Please enter at least one amino acid sequence.")
                return
            
            modules = import_modules()
            predictor = get_predictor()
            
            # Check if input is FASTA format or single sequence
            if is_fasta_format(seq_input):
                st.info("FASTA format detected. Processing multiple sequences...")
                
                # Parse input samples
                samples = parse_fasta(seq_input)
                
                if not samples:
                    st.warning("No valid sequences found. Please check the format.")
                    st.info("Make sure each sequence starts with a line beginning with '>' followed by an ID.")
                    return
                
                st.success(f"Found {len(samples)} sequences")
                
                # Run prediction
                results_df = predictor.predict_batch(samples)
                
                # Display results
                # st.subheader("Prediction Results")
                
                # Show formatted table
                # formatted_results = predictor.format_results(results_df, 'table')
                # st.code(formatted_results, language="markdown")
                
                # Show detailed results
                st.subheader("Detailed Results Table")
                st.dataframe(results_df, use_container_width=True)
                
                # Option to download results
                # csv = results_df.reset_index().rename(columns={'index': 'HIVP_Mutant'}).to_csv(index=False)
                # st.download_button(
                #     label="Download Results as CSV",
                #     data=csv,
                #     file_name="hiv_protease_predictions.csv",
                #     mime="text/csv"
                # )
                
            else:
                # Single sequence prediction
                st.info("Single sequence detected. Processing...")
                
                results = predictor.predict_sequence(seq_input)
                
                # Create DataFrame for display
                df = modules['pd'].DataFrame([
                    {"Drug": drug, "Prediction": data["prediction"], "Probability": f"{data['probability']:.4f}"}
                    for drug, data in results.items()
                ])
                
                # Display results
                st.subheader("Prediction Results")
                st.dataframe(df.set_index("Drug"), use_container_width=True)
                
                # Create a color-coded heatmap
                st.subheader("Resistance Heatmap")
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.dataframe(
                        modules['pd'].DataFrame(
                            {"Resistant": [data["prediction"] for drug, data in results.items()]},
                            index=results.keys()
                        ).style.background_gradient(cmap="RdYlGn_r"),
                        use_container_width=True
                    )
                
                with col2:
                    st.info("0 = Susceptible, 1 = Resistant")
        
        # Footer
        st.markdown("---")
        st.markdown(
            "HIV Protease Inhibitor Resistance Prediction Tool | Powered by ProtBERT"
        )
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()