#!/usr/bin/env python3
"""
Command-line application for HIV protease inhibitor resistance prediction.
This standalone version doesn't use Streamlit to avoid torch._classes errors.
"""

import argparse
import sys
import os
import pandas as pd
import warnings
import traceback

# Suppress warnings
warnings.filterwarnings("ignore", message=".*torch._classes.*")
warnings.filterwarnings("ignore", category=UserWarning)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="HIV Protease Inhibitor Resistance Prediction Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--input",
        help="Input file with sequences in FASTA format (>ID followed by sequence)",
        type=str
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file for predictions",
        type=str
    )
    
    parser.add_argument(
        "-f", "--format",
        help="Output format (table, csv, or json)",
        choices=["table", "csv", "json"],
        default="table"
    )
    
    parser.add_argument(
        "-m", "--model-dir",
        help="Directory containing model folders",
        default="models"
    )
    
    parser.add_argument(
        "-s", "--seed",
        help="Seed used for model training",
        type=int,
        default=42
    )
    
    parser.add_argument(
        "-b", "--batch",
        help="Process sequences in batch mode without interactive interface",
        action="store_true"
    )
    
    parser.add_argument(
        "--use-cuda",
        help="Use CUDA for inference if available",
        action="store_true"
    )
    
    return parser.parse_args()

def parse_input_samples(input_text):
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

def interactive_menu():
    """Interactive menu for command-line interface"""
    print("=" * 60)
    print("HIV Protease Inhibitor Resistance Prediction Tool")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("1. Predict single sequence")
        print("2. Predict multiple sequences (batch mode)")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            process_single_sequence()
        elif choice == '2':
            process_batch_sequences()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

def process_single_sequence():
    """Process a single sequence input"""
    print("\n--- Single Sequence Prediction ---")
    print("Enter amino acid sequence (Ctrl+D when done):")
    
    try:
        seq_lines = []
        while True:
            try:
                line = input()
                seq_lines.append(line)
            except EOFError:
                break
        
        sequence = ''.join(seq_lines).strip()
        
        if not sequence:
            print("Error: Empty sequence.")
            return
        
        # Lazy import to avoid torch._classes error
        from utils.preprocessing import preprocess_sequence
        import torch
        from inference.predictor import ProteinPredictor
        from config.settings import MODEL_ROOT, MODEL_SEED, DEVICE
        
        predictor = ProteinPredictor(MODEL_ROOT, MODEL_SEED, DEVICE)
        results = predictor.predict_sequence(sequence)
        
        # Display results
        print("\nPrediction Results:")
        print("-" * 40)
        print(f"{'Drug':<10} {'Prediction':<15} {'Probability':<15}")
        print("-" * 40)
        
        for drug, data in results.items():
            pred = "Resistant" if data["prediction"] == 1 else "Susceptible"
            prob = f"{data['probability']:.4f}"
            print(f"{drug:<10} {pred:<15} {prob:<15}")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

def process_batch_sequences():
    """Process multiple sequences in batch mode"""
    print("\n--- Batch Sequence Prediction ---")
    print("Enter sequences in FASTA format (>ID followed by sequence).")
    print("Example:")
    print(">SAMPLE1")
    print("PQITLWQRP...")
    print(">SAMPLE2")
    print("PQITLWQRP...")
    print("\nEnter sequences (Ctrl+D when done):")
    
    try:
        input_lines = []
        while True:
            try:
                line = input()
                input_lines.append(line)
            except EOFError:
                break
        
        input_text = '\n'.join(input_lines)
        
        if not input_text.strip():
            print("Error: No input provided.")
            return
        
        # Parse input samples
        samples = parse_input_samples(input_text)
        
        if not samples:
            print("Error: No valid sequences found in input.")
            return
        
        # Lazy import to avoid torch._classes error
        import torch
        from inference.predictor import ProteinPredictor
        from config.settings import MODEL_ROOT, MODEL_SEED, DEVICE
        
        # Run prediction
        predictor = ProteinPredictor(MODEL_ROOT, MODEL_SEED, DEVICE)
        results_df = predictor.predict_batch(samples)
        
        # Format and display results
        formatted_results = predictor.format_results(results_df, 'table')
        print("\nPrediction Results:")
        print(formatted_results)
        
        # Ask if user wants to save results
        save = input("\nSave results to file? (y/n): ")
        if save.lower() == 'y':
            filename = input("Enter filename: ")
            if not filename:
                filename = "results.txt"
            
            with open(filename, 'w') as f:
                f.write(formatted_results)
            print(f"Results saved to {filename}")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

def main():
    """Main function"""
    args = parse_arguments()
    
    try:
        if args.batch or args.input:
            # Non-interactive batch mode
            import torch
            from inference.predictor import ProteinPredictor
            
            # Determine device
            device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
            
            # Initialize predictor
            predictor = ProteinPredictor(
                model_root=args.model_dir,
                seed=args.seed,
                device=device
            )
            
            # Get input sequences
            if args.input:
                # Read from file
                with open(args.input, 'r') as f:
                    input_text = f.read()
            else:
                # Read from stdin
                print("Enter sequences in FASTA format (>ID followed by sequence). Press Ctrl+D (Unix) or Ctrl+Z (Windows) when done:")
                input_text = sys.stdin.read()
            
            # Parse input samples
            samples = parse_input_samples(input_text)
            
            if not samples:
                print("Error: No valid sequences found in input.")
                return 1
            
            # Predict resistance
            results_df = predictor.predict_batch(samples)
            
            # Format results
            formatted_results = predictor.format_results(results_df, args.format)
            
            # Output results
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(formatted_results)
                print(f"Results written to {args.output}")
            else:
                print("\nResults:")
                print(formatted_results)
        else:
            # Interactive mode
            interactive_menu()
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())