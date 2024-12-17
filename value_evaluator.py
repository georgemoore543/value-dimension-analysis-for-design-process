import pandas as pd
import os
from openai import OpenAI
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

def load_api_key():
    """Load OpenAI API key from environment variable or .env file"""
    # First try to get from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    
    # If not found, try loading from .env file
    if not api_key and os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if line.startswith('OPENAI_API_KEY='):
                    api_key = line.strip().split('=')[1].strip()
                    break
    
    return api_key

def initialize_client():
    """Initialize OpenAI client with API key"""
    api_key = load_api_key()
    
    if not api_key:
        print("Error: No OpenAI API key found in environment or .env file")
        return None
        
    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        print(f"Error initializing OpenAI client: {str(e)}")
        return None

def validate_file_structure(df, file_type):
    """Validate the structure of uploaded files."""
    if file_type == "design_process":
        required_cols = ["Prompt"]
        if not all(col in df.columns for col in required_cols):
            return False, "Design Process Output file must contain a 'Prompt' column"
    
    elif file_type == "value_dimensions":
        required_cols = ["value dimensions", "dim_definitions"]
        if not all(col in df.columns for col in required_cols):
            return False, "Value Dimensions file must contain 'value dimensions' and 'dim_definitions' columns"
    
    elif file_type == "scale_definitions":
        if "scale_point" not in df.columns:
            return False, "Scale Definitions file must contain a 'scale_point' column"
        
    return True, "Valid file structure"

def read_excel_files():
    """Read the input Excel files using file dialog."""
    try:
        # Create root window and hide it
        root = tk.Tk()
        root.withdraw()
        
        # Dictionary to store file paths and their descriptions
        file_requests = [
            ("Design Process Output", "design_process"),
            ("Value Dimensions", "value_dimensions"),
            ("Scale Definitions for Value Dimensions", "scale_definitions")
        ]
        
        files_data = {}
        
        # Request each file
        for desc, file_type in file_requests:
            while True:
                filepath = filedialog.askopenfilename(
                    title=f"Select {desc} file (.xlsx)",
                    filetypes=[("Excel files", "*.xlsx")]
                )
                
                if not filepath:  # User cancelled
                    print("File selection cancelled. Exiting program.")
                    return None, None, None
                
                try:
                    df = pd.read_excel(filepath)
                    is_valid, message = validate_file_structure(df, file_type)
                    
                    if not is_valid:
                        print(f"Error: {message}")
                        continue
                    
                    files_data[file_type] = df
                    break
                    
                except Exception as e:
                    print(f"Error reading {desc} file: {e}")
                    continue
        
        # Process the files
        prompts = files_data["design_process"]
        value_dims = files_data["value_dimensions"]
        scale_defs = files_data["scale_definitions"]
        
        # Create dimension-specific scale descriptions
        dimension_scales = {}
        for dimension in value_dims['value dimensions']:
            if dimension in scale_defs.columns:
                dimension_scales[dimension] = dict(zip(scale_defs['scale_point'], scale_defs[dimension]))
            else:
                print(f"Warning: Scale definitions missing for dimension '{dimension}'")
        
        # Convert value dimensions and their definitions to dictionary
        dim_definitions = dict(zip(value_dims['value dimensions'], value_dims['dim_definitions']))
        
        return prompts, dim_definitions, dimension_scales
        
    except Exception as e:
        print(f"Error processing files: {e}")
        return None, None, None

def main():
    # Initialize OpenAI client
    client = initialize_client()
    if not client:
        return
    
    print("OpenAI client initialized successfully!")
    
    # Read Excel files
    prompts, dim_definitions, dimension_scales = read_excel_files()
    if prompts is None or dim_definitions is None or dimension_scales is None:
        return
    
    print("Files loaded successfully!")

if __name__ == "__main__":
    main() 