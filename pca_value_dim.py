print("=" * 50)
print("LOADING PCA_VALUE_DIM.PY")
print("File path:", __file__)
print("Module name:", __name__)
print("=" * 50)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
import os
import sys
from typing import List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.offline as pyo
import webbrowser  # For opening the plot in browser if needed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
import seaborn as sns

print("File loaded, ValueDimensionPCA will be defined at:", __name__)

print("Defining ValueDimensionPCA class...")
class ValueDimensionPCA:
    def __init__(self):
        print("ValueDimensionPCA initialized")
        self.ratings_data = None
        self.value_dims = None
        self.pca_ratings = None
        self.original_dims = None
        self.pca = None
        self.explained_variance_ratio_ = None
        self.components_ = None
        self.prompts = None
        self.cosine_ratings = None
        self.current_ratings = None
        
    def validate_data(self, ratings_dfs: List[pd.DataFrame], dims_dfs: List[pd.DataFrame]) -> Tuple[bool, str]:
        """Validate the loaded data"""
        try:
            # First, handle non-numeric columns in ratings data
            print("\nChecking ratings data format...")
            for i in range(len(ratings_dfs)):
                df = ratings_dfs[i]
                non_numeric_cols = df.select_dtypes(exclude=['int64', 'float64']).columns
                if not non_numeric_cols.empty:
                    print(f"\nRatings file {i+1} contains non-numeric columns:")
                    print("\n".join([f"- {col}" for col in non_numeric_cols]))
                    print("These columns will be removed before PCA analysis.")
                    
                    # Store non-numeric data if needed (you can add this functionality later)
                    # self.store_metadata(df[non_numeric_cols])
                    
                    # Remove non-numeric columns
                    ratings_dfs[i] = df.select_dtypes(include=['int64', 'float64'])
                    print(f"Shape after removing non-numeric columns: {ratings_dfs[i].shape}")
            
            # Continue with rest of validation...
            missing_info = []
            
            # Check ratings files for missing values
            print("\nChecking ratings files for missing values...")
            for i, df in enumerate(ratings_dfs):
                missing = df.isnull().sum()
                if missing.any():
                    cols_with_missing = missing[missing > 0]
                    missing_info.append(f"\nRatings file {i+1}:")
                    for col, count in cols_with_missing.items():
                        missing_info.append(f"- Column '{col}': {count} missing values")
            
            # Check dimensions files for missing values
            print("\nChecking dimensions files for missing values...")
            for i, df in enumerate(dims_dfs):
                missing = df.isnull().sum()
                if missing.any():
                    cols_with_missing = missing[missing > 0]
                    # If 'dim_definitions' column has missing values
                    if 'dim_definitions' in cols_with_missing:
                        response = messagebox.askyesno(
                            "Missing Dimension Definitions",
                            f"Dimensions file {i+1} has missing values in the 'dim_definitions' column.\n\n"
                            "Would you like to proceed without the dimension definitions?"
                        )
                        if response:
                            # Remove dim_definitions column
                            dims_dfs[i] = df.drop(columns=['dim_definitions'])
                            print(f"Removed 'dim_definitions' column from dimensions file {i+1}")
                            # Remove this from cols_with_missing
                            cols_with_missing = cols_with_missing.drop('dim_definitions')
                    
                    # Check if there are still other missing values
                    if cols_with_missing.any():
                        missing_info.append(f"\nDimensions file {i+1}:")
                        for col, count in cols_with_missing.items():
                            missing_info.append(f"- Column '{col}': {count} missing values")
            
            if missing_info:
                message = "Missing values detected:\n" + "\n".join(missing_info)
                print(message)
                
                # Ask user what to do about remaining missing values
                response = messagebox.askyesno(
                    "Missing Values Detected",
                    f"{message}\n\nWould you like to proceed anyway?\n"
                    "(Missing values will be handled by removing rows with any missing data)"
                )
                
                if response:
                    print("User chose to proceed with missing values")
                    return True, "Proceeding with missing values"
                else:
                    print("User chose not to proceed")
                    return False, "Operation cancelled by user"
            
            # Validate required columns ('value dimension' is required)
            print("\nValidating column names...")
            required_cols = ['value dimensions']
            for i, df in enumerate(dims_dfs):
                print(f"Columns in dimensions file {i+1}:", df.columns.tolist())
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    message = f"Dimensions file {i+1} is missing required column: 'value dimensions'"
                    print(message)
                    return False, message
            
            return True, "Data validation successful"
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    def load_data(self, ratings_paths: List[str], dims_paths: List[str]):
        """Load and validate data files"""
        try:
            print("\nDEBUG: Starting data load process")
            print("Loading dimensions files...")
            dims_dfs = []
            for path in dims_paths:
                print(f"Reading dimensions file: {path}")
                df = pd.read_excel(path)
                print(f"Dimensions shape: {df.shape}")
                print(f"Dimensions columns: {df.columns.tolist()}")
                dims_dfs.append(df)
            
            self.value_dims = pd.concat(dims_dfs, axis=0)
            print(f"Combined dimensions shape: {self.value_dims.shape}")
            print(f"Available columns: {self.value_dims.columns.tolist()}")
            
            # Load ratings files
            print("Loading ratings files...")
            ratings_dfs = []
            for path in ratings_paths:
                print(f"\nReading {path}")
                df = pd.read_excel(path)
                print(f"Original shape: {df.shape}")
                
                # Store prompts before removing non-numeric columns
                if 'Prompt' in df.columns:
                    print("Storing prompts...")
                    self.prompts = df['Prompt'].to_dict()
                    
                    # Remove all non-numeric columns
                    numeric_df = df.select_dtypes(include=['int64', 'float64'])
                    print(f"Shape after removing non-numeric columns: {numeric_df.shape}")
                    print(f"Numeric columns: {numeric_df.columns.tolist()}")  # Debug print
                    ratings_dfs.append(numeric_df)
                else:
                    print("Warning: No 'Prompt' column found")
                    ratings_dfs.append(df)
            
            # Process validated data
            print("\nProcessing validated data...")
            self.ratings_data = pd.concat(ratings_dfs, axis=1)
            print(f"Final ratings data shape: {self.ratings_data.shape}")
            
            # Store original dimension names
            self.original_dims = self.ratings_data.columns.tolist()
            print(f"Stored original dimensions: {self.original_dims}")  # Debug print
            
            # Initialize current_ratings
            self.current_ratings = self.ratings_data.copy()
            print("Current ratings initialized")
            
            return True, "Data loaded successfully"
            
        except Exception as e:
            import traceback
            print(f"\nError loading data: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())
            return False, f"Error loading data: {str(e)}"
    
    def perform_pca(self):
        """Perform PCA on the current ratings data"""
        try:
            print("\nDEBUG: Starting PCA process")
            
            # Filter out '#' columns BEFORE storing original dimensions
            valid_columns = [col for col in self.current_ratings.columns if col != '#']
            filtered_data = self.current_ratings[valid_columns]
            
            # Store only valid dimensions
            self.original_dims = valid_columns
            print(f"Original dimensions (filtered): {self.original_dims}")
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(filtered_data)
            
            # Perform PCA
            self.pca = PCA()
            self.pca_ratings = self.pca.fit_transform(scaled_data)
            
            # Store PCA results
            self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
            self.components_ = self.pca.components_
            
            print("\nDEBUG: Calculating contributions")
            
            # Get loadings and explained variance
            loadings = self.components_.T
            exp_var = self.explained_variance_ratio_
            
            # Calculate importance scores for each variable
            importance_scores = np.zeros(len(valid_columns))
            
            for i in range(len(valid_columns)):
                # Weight the absolute loadings by explained variance
                weighted_loadings = np.abs(loadings[i, :]) * exp_var
                # Take the sum of the top 3 contributions
                top_contributions = np.sort(weighted_loadings)[-3:]
                importance_scores[i] = np.sum(top_contributions)
                
                print(f"\nDEBUG: {valid_columns[i]}:")
                print(f"Top 3 weighted loadings: {top_contributions}")
                print(f"Importance score: {importance_scores[i]:.4f}")
            
            # Convert to percentages
            total_importance = np.sum(importance_scores)
            variance_percentages = (importance_scores / total_importance) * 100
            
            self.variance_contributions = pd.Series(
                variance_percentages,
                index=valid_columns
            )
            
            print("\nDEBUG: Final variance contributions:")
            sorted_contributions = self.variance_contributions.sort_values(ascending=False)
            for dim, contrib in sorted_contributions.items():
                print(f"{dim}: {contrib:.2f}%")
            
            print("\nDEBUG: Validation:")
            print(f"Sum of contributions: {self.variance_contributions.sum():.2f}%")
            print(f"Min contribution: {self.variance_contributions.min():.2f}%")
            print(f"Max contribution: {self.variance_contributions.max():.2f}%")
            
            return True
            
        except Exception as e:
            print(f"Error performing PCA: {str(e)}")
            print("Full error details:")
            import traceback
            traceback.print_exc()
            return False

    def get_prompt_for_component(self, component_index: int, n_top: int = 5) -> List[Tuple[str, float]]:
        """Get the top contributing prompts for a given principal component"""
        loadings = self.components_[component_index]
        # Get indices of top absolute loadings
        top_indices = np.abs(loadings).argsort()[-n_top:][::-1]
        
        results = []
        for idx in top_indices:
            loading = loadings[idx]
            prompt = self.prompts.get(self.original_dims[idx], "Unknown prompt")
            results.append((prompt, loading))
        
        return results

    def calculate_cosine_similarity(self, prompts, value_dims):
        """Calculate cosine similarity between prompts and value dimensions"""
        try:
            print("\nDEBUG: Starting cosine similarity calculation")
            if not prompts:
                raise ValueError("No prompts available")
            if value_dims is None or len(value_dims) == 0:
                raise ValueError("No value dimensions available")
            
            print(f"Number of prompts: {len(prompts)}")
            
            # Handle value_dims whether it's a Series or DataFrame
            if isinstance(value_dims, pd.DataFrame):
                print("Value dimensions DataFrame detected")
                # If we have definitions, combine them with dimension names
                if 'dim_definitions' in value_dims.columns:
                    print("Using dimension names and definitions")
                    dim_texts = [f"{dim} - {def_}" if pd.notna(def_) else dim 
                               for dim, def_ in zip(value_dims['value dimensions'], 
                                                  value_dims['dim_definitions'])]
                else:
                    print("Using dimension names only")
                    dim_texts = value_dims['value dimensions'].tolist()
            else:
                print("Value dimensions Series detected")
                dim_texts = value_dims.tolist()
            
            print(f"Value dimensions available: {dim_texts}")
            
            # Convert prompts dictionary to list of texts
            prompt_texts = list(prompts.values())
            
            print("Initializing TF-IDF vectorizer...")
            vectorizer = TfidfVectorizer(stop_words='english')
            
            print("Vectorizing texts...")
            all_texts = prompt_texts + dim_texts
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            print("Calculating similarities...")
            n_prompts = len(prompt_texts)
            prompt_vectors = tfidf_matrix[:n_prompts]
            dim_vectors = tfidf_matrix[n_prompts:]
            
            similarity_matrix = cosine_similarity(prompt_vectors, dim_vectors)
            print(f"Similarity matrix shape: {similarity_matrix.shape}")
            
            # Create DataFrame with original dimension names as columns
            if isinstance(value_dims, pd.DataFrame):
                dim_names = value_dims['value dimensions'].tolist()
            else:
                dim_names = value_dims.tolist()
            
            self.cosine_ratings = pd.DataFrame(
                similarity_matrix,
                index=self.ratings_data.index,
                columns=[f"{dim}_Rating" for dim in dim_names]
            )
            
            print("Cosine similarity calculation complete")
            print("Sample of cosine similarities:")
            print(self.cosine_ratings.head())
            
            return True
            
        except Exception as e:
            print(f"Error calculating cosine similarity: {str(e)}")
            print(f"Prompts available: {bool(prompts)}")
            print(f"Value dims available: {value_dims is not None}")
            return False

    def perform_clustering(self, n_clusters=3):
        """Perform clustering on PCA results"""
        try:
            print("\nDEBUG: Starting clustering analysis")
            
            # Use top 3 PCs for clustering
            clustering_data = self.pca_ratings[:, :3]
            print(f"Clustering data shape: {clustering_data.shape}")
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.cluster_labels = kmeans.fit_predict(clustering_data)
            print(f"Unique clusters: {np.unique(self.cluster_labels)}")
            
            # Perform LDA
            lda = LinearDiscriminantAnalysis()
            self.lda_transformed = lda.fit_transform(clustering_data, self.cluster_labels)
            print("LDA transformation complete")
            
            # Store cluster centers and explained variance
            self.cluster_centers = kmeans.cluster_centers_
            self.lda_explained_variance = lda.explained_variance_ratio_
            
            print("Clustering analysis complete")
            return True
            
        except Exception as e:
            print(f"Error in clustering: {str(e)}")
            return False

    def calculate_dimension_variance(self):
        """Return the pre-calculated variance contributions"""
        try:
            if hasattr(self, 'variance_contributions'):
                return self.variance_contributions
            else:
                print("DEBUG: Variance contributions not yet calculated")
                return None
            
        except Exception as e:
            print("DEBUG: Error accessing variance contributions:", str(e))
            return None

# Then, the GUI class
class ValueDimensionPCAGui:
    # Add the PCA class as a class attribute
    PCA_Class = ValueDimensionPCA
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Value Dimension PCA Analysis")
        self.pca_instance = None
        self.ratings_type = tk.StringVar(value="original")
        
        # Initialize data structures
        self.ratings_paths = []
        self.dims_paths = []
        self.ratings_count = tk.StringVar(value="1")
        self.dims_count = tk.StringVar(value="1")
        
        # Create initial widgets
        self.create_initial_widgets()
        
    def create_initial_widgets(self):
        # File count frame
        count_frame = ttk.LabelFrame(self.root, text="Number of Files", padding="10")
        count_frame.pack(fill="x", padx=10, pady=5)
        
        # Ratings count
        ttk.Label(count_frame, text="Number of ratings spreadsheets:").grid(row=0, column=0, padx=5, pady=5)
        ratings_spinbox = ttk.Spinbox(
            count_frame, 
            from_=1, 
            to=10, 
            width=5,
            textvariable=self.ratings_count
        )
        ratings_spinbox.grid(row=0, column=1, padx=5, pady=5)
        
        # Dimensions count
        ttk.Label(count_frame, text="Number of dimension spreadsheets:").grid(row=1, column=0, padx=5, pady=5)
        dims_spinbox = ttk.Spinbox(
            count_frame, 
            from_=1, 
            to=10, 
            width=5,
            textvariable=self.dims_count
        )
        dims_spinbox.grid(row=1, column=1, padx=5, pady=5)
        
        # Next button
        ttk.Button(count_frame, text="Next", command=self.setup_file_selection).grid(row=2, column=0, columnspan=2, pady=10)
    
    def setup_file_selection(self):
        # Get the values
        num_ratings = int(self.ratings_count.get())
        num_dims = int(self.dims_count.get())
        
        # Clear initial widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Initialize the paths lists
        self.ratings_paths = [None] * num_ratings
        self.dims_paths = [None] * num_dims
        
        # Create file selection widgets
        self.create_file_selection_widgets()
    
    def create_file_selection_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # Ratings files section
        ratings_frame = ttk.LabelFrame(main_frame, text="Ratings Spreadsheets", padding="10")
        ratings_frame.pack(fill="x", pady=5)
        
        for i in range(len(self.ratings_paths)):
            self.create_file_selector(ratings_frame, f"Ratings file {i+1}", "ratings", i)
        
        # Dimensions files section
        dims_frame = ttk.LabelFrame(main_frame, text="Dimensions Spreadsheets", padding="10")
        dims_frame.pack(fill="x", pady=5)
        
        for i in range(len(self.dims_paths)):
            self.create_file_selector(dims_frame, f"Dimensions file {i+1}", "dims", i)
        
        # Proceed button
        self.proceed_button = ttk.Button(main_frame, text="Proceed with Analysis", 
                                       command=self.proceed, state="disabled")
        self.proceed_button.pack(pady=10)
    
    def create_file_selector(self, parent, label, file_type, index):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=2)
        
        ttk.Label(frame, text=label).pack(side="left", padx=5)
        path_var = tk.StringVar()
        path_entry = ttk.Entry(frame, textvariable=path_var, width=50)
        path_entry.pack(side="left", padx=5)
        
        def browse():
            filename = filedialog.askopenfilename(
                filetypes=[("Excel files", "*.xlsx *.xls")]
            )
            if filename:
                path_var.set(filename)
                self.verify_and_update_file(filename, file_type, index)
        
        ttk.Button(frame, text="Browse", command=browse).pack(side="left", padx=5)
    
    def verify_and_update_file(self, path, file_type, index):
        try:
            # Verify file can be read
            pd.read_excel(path, nrows=1)
            
            # Store path in appropriate list
            if file_type == "ratings":
                self.ratings_paths[index] = path
            else:
                self.dims_paths[index] = path
            
            self.check_all_files_selected()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file: {str(e)}")
    
    def check_all_files_selected(self):
        # Enable proceed button if all files are selected
        if all(self.ratings_paths) and all(self.dims_paths):
            self.proceed_button['state'] = 'normal'
    
    def load_and_prepare_data(self):
        """Load data and prepare both ratings types"""
        try:
            print("Starting data preparation...")
            self.pca_instance = ValueDimensionPCA()
            
            # Load initial data
            success, message = self.pca_instance.load_data(self.ratings_paths, self.dims_paths)
            if not success:
                messagebox.showerror("Error", message)
                return False

            # Load dimensions data first
            print("\nLoading dimensions data...")
            dims_dfs = []
            for path in self.dims_paths:
                print(f"Reading dimensions file: {path}")
                df = pd.read_excel(path)
                print(f"Dimensions shape: {df.shape}")
                print(f"Dimensions columns: {df.columns.tolist()}")
                dims_dfs.append(df)
            
            # Concatenate dimensions data
            self.pca_instance.value_dims = pd.concat(dims_dfs, axis=0)
            print(f"Combined dimensions shape: {self.pca_instance.value_dims.shape}")

            # Calculate cosine similarities
            print("Calculating cosine similarities...")
            if 'value dimensions' not in self.pca_instance.value_dims.columns:
                print("Available columns:", self.pca_instance.value_dims.columns.tolist())
                messagebox.showerror("Error", "Could not find 'value dimensions' column in dimensions file")
                return False
            
            success = self.pca_instance.calculate_cosine_similarity(
                self.pca_instance.prompts,
                self.pca_instance.value_dims['value dimensions']
            )
            if not success:
                messagebox.showerror("Error", "Failed to calculate cosine similarities")
                return False

            # Ask user which ratings to use
            response = messagebox.askyesno(
                "Select Ratings Type",
                "Data prepared successfully!\n\n"
                "Would you like to use cosine similarity scores?\n"
                "(No will use original ratings)"
            )
            
            # Set initial ratings type based on user choice
            self.ratings_type.set("cosine" if response else "original")
            self.pca_instance.current_ratings = (
                self.pca_instance.cosine_ratings if response 
                else self.pca_instance.ratings_data
            )
            
            return True
            
        except Exception as e:
            print(f"Error in data preparation: {str(e)}")
            print("Debug information:")
            print(f"- Paths available: {bool(self.dims_paths)}")
            if hasattr(self.pca_instance, 'value_dims'):
                print(f"- Value dims columns: {self.pca_instance.value_dims.columns.tolist()}")
            messagebox.showerror("Error", f"Failed to prepare data: {str(e)}")
            return False

    def proceed(self):
        """Modified proceed method"""
        try:
            print("Starting proceed method...")
            
            # Load and prepare all data first
            if not self.load_and_prepare_data():
                return
            
            # Perform initial PCA
            print("Performing initial PCA...")
            if not self.pca_instance.perform_pca():
                messagebox.showerror("Error", "PCA calculation failed")
                return
            
            # Show visualization with toggle option
            print("Showing visualization...")
            self.show_visualization(self.pca_instance)
            
        except Exception as e:
            print(f"Error in proceed: {str(e)}")
            messagebox.showerror("Error", f"Error processing files: {str(e)}")
    
    def show_visualization(self, pca):
        """Modified show_visualization to include summary panel"""
        try:
            print("Creating visualization window...")
            if not hasattr(self, 'viz_window'):
                self.viz_window = tk.Toplevel(self.root)
                self.viz_window.title("PCA Visualization")
            
            # Store PCA instance
            print("Storing PCA instance...")
            self.pca_instance = pca
            
            # Create main container
            main_frame = ttk.Frame(self.viz_window)
            main_frame.pack(fill="both", expand=True)
            
            # Create controls and panels
            print("Creating UI elements...")
            self.create_visualization_controls(main_frame)
            self.create_summary_panel(main_frame)
            
            # Create figure frame
            print("Creating figure frame...")
            self.fig_frame = ttk.Frame(main_frame)
            self.fig_frame.pack(fill="both", expand=True, pady=5)
            
            # Update displays
            print("Updating displays...")
            self.update_summary()
            self.update_plot(self.pca_instance)
            
            print("Visualization setup complete")
            
        except Exception as e:
            print(f"Error in show_visualization: {str(e)}")
            print(f"Debug info:")
            print(f"- PCA instance available: {hasattr(self, 'pca_instance')}")
            print(f"- Plot controls initialized: {hasattr(self, 'pc_x') and hasattr(self, 'pc_y') and hasattr(self, 'plot_type')}")
            raise
    
    def create_summary_panel(self, parent):
        """Create PCA results summary panel"""
        summary_frame = ttk.LabelFrame(parent, text="PCA Summary", padding="10")
        summary_frame.pack(fill="x", padx=10, pady=5)

        # Threshold selector
        threshold_frame = ttk.Frame(summary_frame)
        threshold_frame.pack(fill="x", pady=5)
        ttk.Label(threshold_frame, text="Variance explained threshold (%):").pack(side="left")
        self.variance_threshold = tk.StringVar(value="80")
        threshold_entry = ttk.Entry(threshold_frame, textvariable=self.variance_threshold, width=5)
        threshold_entry.pack(side="left", padx=5)
        ttk.Button(threshold_frame, text="Update", command=self.update_summary).pack(side="left", padx=5)

        # Create text widget for summary
        self.summary_text = tk.Text(summary_frame, height=15, width=60)
        self.summary_text.pack(fill="both", expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(summary_frame, orient="vertical", command=self.summary_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.summary_text.configure(yscrollcommand=scrollbar.set)

    def update_summary(self):
        """Update PCA summary with current results"""
        try:
            print("\nDEBUG: Starting summary update")
            
            # Check if PCA instance exists
            if not hasattr(self, 'pca_instance'):
                print("No PCA instance found")
                return
            print("PCA instance found")
            
            # Check if PCA object exists
            if not hasattr(self.pca_instance, 'pca'):
                print("No PCA object found in instance")
                return
            print("PCA object found")
            
            # Debug print for original_dims
            print(f"Original dims type: {type(self.pca_instance.original_dims)}")
            print(f"Original dims value: {self.pca_instance.original_dims}")
            
            # Clear existing text
            if not hasattr(self, 'summary_text'):
                print("No summary_text widget found")
                return
            self.summary_text.delete(1.0, tk.END)
            
            # Get threshold (with error checking)
            try:
                threshold = float(self.variance_threshold.get()) / 100
            except (ValueError, AttributeError):
                threshold = 0.8  # default to 80%
            print(f"Using threshold: {threshold}")

            # Data preprocessing info
            self.summary_text.insert(tk.END, "Data Preprocessing:\n", "heading")
            self.summary_text.insert(tk.END, "- Standardization: StandardScaler\n")
            
            # Add dimension variance contributions
            variance_contributions = self.pca_instance.calculate_dimension_variance()
            if variance_contributions is not None:
                self.summary_text.insert(tk.END, "\nVariance Contribution by Dimension:\n", "heading")
                # Sort dimensions by contribution
                sorted_contributions = variance_contributions.sort_values(ascending=False)
                for dim, contrib in sorted_contributions.items():
                    self.summary_text.insert(tk.END, f"- {dim}: {contrib:.2f}%\n")
                self.summary_text.insert(tk.END, "\n")
            
            # Debug print for components
            print(f"Components type: {type(self.pca_instance.pca.components_)}")
            print(f"Components shape: {self.pca_instance.pca.components_.shape}")
            
            # Debug print for explained variance
            print(f"Explained variance type: {type(self.pca_instance.pca.explained_variance_ratio_)}")
            print(f"Explained variance shape: {self.pca_instance.pca.explained_variance_ratio_.shape}")
            
            # Add input dimensions info with explicit type checking
            if hasattr(self.pca_instance, 'original_dims'):
                dims = self.pca_instance.original_dims
                if dims is not None:
                    try:
                        n_dims = len(dims)
                        self.summary_text.insert(tk.END, f"- Input dimensions: {n_dims}\n")
                    except TypeError:
                        print(f"Could not get length of original_dims: {dims}")
            
            self.summary_text.insert(tk.END, 
                f"- Data type: {'Cosine Similarity' if self.ratings_type.get() == 'cosine' else 'Original Ratings'}\n\n")

            # Check for explained variance ratios
            if hasattr(self.pca_instance.pca, 'explained_variance_ratio_'):
                print("Adding explained variance information")
                self.summary_text.insert(tk.END, "Explained Variance by Component:\n", "heading")
                var_ratios = self.pca_instance.pca.explained_variance_ratio_
                cumulative = np.cumsum(var_ratios)
                
                for i, (var, cum) in enumerate(zip(var_ratios, cumulative), 1):
                    self.summary_text.insert(tk.END, 
                        f"PC{i}: {var:.3f} ({cum:.3f} cumulative)\n")

                # Components needed for threshold
                n_components = np.sum(cumulative <= threshold) + 1
                self.summary_text.insert(tk.END, 
                    f"\nComponents needed for {threshold*100}% variance: {n_components}\n\n")

            # Check for components/loadings
            if hasattr(self.pca_instance.pca, 'components_'):
                print("Adding component loadings information")
                self.summary_text.insert(tk.END, "Top Component Loadings:\n", "heading")
                loadings = self.pca_instance.pca.components_
                
                if hasattr(self.pca_instance, 'original_dims'):
                    # For each component, show top contributing dimensions
                    for i in range(min(3, len(loadings))):  # Show first 3 components
                        self.summary_text.insert(tk.END, f"\nPrincipal Component {i+1}:\n")
                        # Get indices of top absolute loadings
                        top_indices = np.abs(loadings[i]).argsort()[-5:][::-1]  # Top 5
                        for idx in top_indices:
                            dim_name = self.pca_instance.original_dims[idx]
                            loading = loadings[i][idx]
                            self.summary_text.insert(tk.END, 
                                f"  {dim_name}: {loading:.3f}\n")

            # Software info
            self.summary_text.insert(tk.END, "\nSoftware Information:\n", "heading")
            self.summary_text.insert(tk.END, "- scikit-learn PCA\n")
            self.summary_text.insert(tk.END, "- pandas DataFrame\n")
            self.summary_text.insert(tk.END, "- numpy arrays\n")

            # Apply tags for formatting
            self.summary_text.tag_configure("heading", font=("TkDefaultFont", 10, "bold"))
            
            print("Summary update completed successfully")
            
        except Exception as e:
            print(f"Error updating summary: {str(e)}")
            print("Debug information:")
            print(f"- PCA instance exists: {hasattr(self, 'pca_instance')}")
            if hasattr(self, 'pca_instance'):
                print(f"- PCA object exists: {hasattr(self.pca_instance, 'pca')}")
                if hasattr(self.pca_instance, 'pca'):
                    print(f"- Components exist: {hasattr(self.pca_instance.pca, 'components_')}")
                    print(f"- Original dims exist: {hasattr(self.pca_instance, 'original_dims')}")
                    if hasattr(self.pca_instance, 'original_dims'):
                        print(f"- Original dims value: {self.pca_instance.original_dims}")
            messagebox.showerror("Error", f"Failed to update summary: {str(e)}")

    def create_visualization_controls(self, parent):
        """Update visualization controls to include heatmap option"""
        try:
            print("Creating control frame...")
            control_frame = ttk.Frame(parent)
            control_frame.pack(fill="x", padx=10, pady=5)
            
            # Ratings type toggle
            print("Creating ratings toggle...")
            ratings_frame = ttk.LabelFrame(control_frame, text="Ratings Type", padding="5")
            ratings_frame.pack(fill="x", pady=5)
            
            ttk.Radiobutton(
                ratings_frame,
                text="Original Ratings",
                variable=self.ratings_type,
                value="original",
                command=self.update_ratings_type
            ).pack(side="left", padx=5)
            
            ttk.Radiobutton(
                ratings_frame,
                text="Cosine Similarity Scores",
                variable=self.ratings_type,
                value="cosine",
                command=self.update_ratings_type
            ).pack(side="left", padx=5)
            
            # Plot controls
            print("Creating plot controls...")
            plot_controls = ttk.LabelFrame(control_frame, text="Plot Controls", padding="5")
            plot_controls.pack(fill="x", pady=5)
            
            # PC selection (only needed for scatter plot)
            self.pc_x = tk.StringVar(value="1")
            self.pc_y = tk.StringVar(value="2")
            
            ttk.Label(plot_controls, text="PC X:").pack(side="left", padx=5)
            ttk.Entry(plot_controls, textvariable=self.pc_x, width=3).pack(side="left")
            ttk.Label(plot_controls, text="PC Y:").pack(side="left", padx=5)
            ttk.Entry(plot_controls, textvariable=self.pc_y, width=3).pack(side="left")
            
            # Point size control
            self.point_size = tk.StringVar(value="50")
            ttk.Label(plot_controls, text="Point Size:").pack(side="left", padx=5)
            ttk.Entry(plot_controls, textvariable=self.point_size, width=4).pack(side="left")
            
            # Update plot type selection
            self.plot_type = tk.StringVar(value="scatter")
            plot_types = ["scatter", "matrix", "heatmap"]  # Added heatmap option
            for plot_type in plot_types:
                ttk.Radiobutton(
                    plot_controls,
                    text=plot_type.capitalize(),
                    variable=self.plot_type,
                    value=plot_type,
                    command=lambda: self.update_plot(self.pca_instance)
                ).pack(side="left", padx=5)
            
            print("Control frame created successfully")
            
        except Exception as e:
            print(f"Error creating visualization controls: {str(e)}")
            raise
    
    def update_plot(self, pca):
        """Update visualization based on current settings"""
        try:
            plot_type = self.plot_type.get()
            
            # Clear previous plot
            for widget in self.fig_frame.winfo_children():
                widget.destroy()
            
            if plot_type == "matrix":
                self.create_matrix_plot(pca)
            elif plot_type == "heatmap":
                self.create_heatmap_plot(pca)
            else:  # scatter
                pc1 = int(self.pc_x.get()) - 1
                pc2 = int(self.pc_y.get()) - 1
                self.create_scatter_plot(pca, pc1, pc2)
                
        except Exception as e:
            print(f"Error updating plot: {str(e)}")
            messagebox.showerror("Error", f"Error updating plot: {str(e)}")
    
    def run(self):
        self.root.mainloop()

    def create_scatter_plot(self, pca, pc1, pc2):
        """Create scatter plot of PCA results"""
        try:
            # Get point size with error handling
            try:
                point_size = float(self.point_size.get())
            except (ValueError, AttributeError):
                point_size = 50  # default size if there's an error
                
            # Create the scatter plot
            plt.figure(figsize=(10, 8))
            plt.scatter(
                pca.pca_ratings[:, pc1],
                pca.pca_ratings[:, pc2],
                s=point_size,
                alpha=0.6
            )
            
            # Add labels
            plt.xlabel(f'Principal Component {pc1 + 1}')
            plt.ylabel(f'Principal Component {pc2 + 1}')
            plt.title('PCA Results')
            
            # Add grid
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Show plot
            plt.tight_layout()
            
            # If we have a previous plot, clear it
            if hasattr(self, 'canvas'):
                self.canvas.get_tk_widget().destroy()
            
            # Create new canvas
            self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.fig_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            print(f"Error creating scatter plot: {str(e)}")
            raise

    def create_matrix_plot(self, pca):
        """Create 3x3 interactive scatterplot matrix of top 3 PCs"""
        try:
            print("Creating matrix plot...")
            
            # Create 3x3 subplot figure
            fig = make_subplots(
                rows=3, cols=3,
                subplot_titles=[f'PC{i+1} vs PC{j+1}' for i in range(3) for j in range(3)]
            )
            
            # Get top 3 PCs
            pc_data = pca.pca_ratings[:, :3]
            
            # Create scatterplots for each combination
            for i in range(3):
                for j in range(3):
                    # Get data for current plot
                    x = pc_data[:, i]
                    y = pc_data[:, j]
                    
                    # Create hover text
                    hover_text = [f"Prompt: {prompt}<br>PC{i+1}: {x_val:.3f}<br>PC{j+1}: {y_val:.3f}" 
                                for prompt, x_val, y_val in zip(pca.prompts.values(), x, y)]
                    
                    # Add scatter plot
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode='markers',
                            marker=dict(size=8),
                            hovertext=hover_text,
                            hoverinfo='text',
                            showlegend=False
                        ),
                        row=i+1, col=j+1
                    )
                    
                    # Update axes labels
                    fig.update_xaxes(title_text=f'PC{i+1}', row=i+1, col=j+1)
                    fig.update_yaxes(title_text=f'PC{j+1}', row=i+1, col=j+1)
            
            # Update layout
            fig.update_layout(
                height=800,
                width=800,
                title='PCA Components Matrix Plot',
                showlegend=False,
                template='plotly_white'
            )
            
            # If we have a previous plot, clear it
            if hasattr(self, 'matrix_frame'):
                self.matrix_frame.destroy()
            
            # Create frame for the plot
            self.matrix_frame = ttk.Frame(self.fig_frame)
            self.matrix_frame.pack(fill=tk.BOTH, expand=True)
            
            # Generate HTML and create a temporary file
            html_path = "pca_matrix_plot.html"
            pyo.plot(fig, filename=html_path, auto_open=False)
            
            # Create button to open plot in browser
            ttk.Button(
                self.matrix_frame,
                text="Open Interactive Plot in Browser",
                command=lambda: webbrowser.open(html_path)
            ).pack(pady=10)
            
            # Also display a static message
            ttk.Label(
                self.matrix_frame,
                text="Click the button above to view the interactive plot in your browser"
            ).pack(pady=5)
            
            print("Matrix plot created successfully")
            
        except Exception as e:
            print(f"Error creating matrix plot: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def update_ratings_type(self):
        """Handle ratings type selection"""
        try:
            print(f"\nDEBUG: Updating ratings type to: {self.ratings_type.get()}")
            
            if not hasattr(self, 'pca_instance'):
                print("No PCA instance available")
                return
            
            if self.ratings_type.get() == "original":
                print("Switching to original ratings")
                self.pca_instance.current_ratings = self.pca_instance.ratings_data
            else:
                print("Switching to cosine similarity ratings")
                self.pca_instance.current_ratings = self.pca_instance.cosine_ratings
            
            # Update PCA with new ratings
            print("Performing PCA with updated ratings")
            if self.pca_instance.perform_pca():
                print("PCA completed successfully")
                if hasattr(self, 'update_plot'):
                    print("Updating visualization")
                    self.update_plot(self.pca_instance)
            else:
                print("PCA calculation failed")
                messagebox.showerror("Error", "Failed to update PCA calculation")
            
        except Exception as e:
            print(f"Error updating ratings type: {str(e)}")
            messagebox.showerror("Error", f"Failed to update ratings type: {str(e)}")
            self.ratings_type.set("original")

    def create_heatmap_plot(self, pca):
        """Create interactive heatmap of LDA clustering results"""
        try:
            print("Creating heatmap plot...")
            
            # Perform clustering if not already done
            if not hasattr(pca, 'cluster_labels'):
                print("Performing clustering...")
                pca.perform_clustering()
            
            # Create the heatmap data
            print("Creating heatmap data...")
            unique_clusters = np.unique(pca.cluster_labels.astype(int))
            n_components = min(12, pca.pca.components_.shape[0])  # Limit to 12 components
            
            # Initialize cluster data array
            cluster_data = np.zeros((len(unique_clusters), n_components))
            
            # Calculate mean for each cluster
            for i, cluster in enumerate(unique_clusters):
                mask = pca.cluster_labels.astype(int) == cluster
                cluster_prompts = pca.pca_ratings[mask, :n_components]  # Limit to n_components
                cluster_data[i] = np.mean(cluster_prompts, axis=0)
            
            print(f"Cluster data shape: {cluster_data.shape}")
            
            # Create DataFrame for heatmap
            cluster_df = pd.DataFrame(
                cluster_data,
                columns=[f"PC{i+1}" for i in range(n_components)],
                index=[f"Cluster {i+1}" for i in range(len(unique_clusters))]
            )
            
            print("Creating plotly figure...")
            # Create plotly heatmap
            fig = go.Figure(data=go.Heatmap(
                z=cluster_df.values,
                x=cluster_df.columns,
                y=cluster_df.index,
                colorscale='RdBu',
                zmid=0,
                text=np.round(cluster_df.values, 3),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False,
                hovertemplate='Cluster: %{y}<br>Component: %{x}<br>Value: %{z:.3f}<extra></extra>'
            ))
            
            # Update layout
            fig.update_layout(
                title='Cluster Heatmap of PCA Components',
                height=600,
                width=800,
                xaxis_title='Principal Components',
                yaxis_title='Clusters',
                template='plotly_white'
            )
            
            # Generate HTML
            html_path = "pca_heatmap.html"
            print(f"Saving heatmap to: {html_path}")
            pyo.plot(fig, filename=html_path, auto_open=False)
            
            # Create instructions and button in GUI
            instruction_label = ttk.Label(
                self.fig_frame,
                text="The interactive heatmap has been generated.",
                font=('TkDefaultFont', 10, 'bold')
            )
            instruction_label.pack(pady=(20,5))
            
            # Add cluster information
            print("Adding cluster information...")
            cluster_info = self.create_cluster_info(pca)
            cluster_info.pack(pady=5)
            
            open_button = ttk.Button(
                self.fig_frame,
                text="Open Heatmap in Browser",
                command=lambda: webbrowser.open(html_path)
            )
            open_button.pack(pady=(5,20))
            
            print("Heatmap created successfully")
            
        except Exception as e:
            print(f"Error creating heatmap: {str(e)}")
            print("Debug information:")
            print(f"- Cluster labels shape: {pca.cluster_labels.shape if hasattr(pca, 'cluster_labels') else 'No labels'}")
            print(f"- PCA ratings shape: {pca.pca_ratings.shape if hasattr(pca, 'pca_ratings') else 'No ratings'}")
            print(f"- Number of components: {pca.pca.components_.shape if hasattr(pca, 'pca') else 'No components'}")
            raise

    def create_cluster_info(self, pca):
        """Create text widget with cluster information"""
        try:
            print("Creating cluster information panel...")
            info_frame = ttk.LabelFrame(self.fig_frame, text="Cluster Information", padding=10)
            
            text_widget = tk.Text(info_frame, height=6, width=50)
            text_widget.pack(padx=5, pady=5)
            
            # Convert cluster labels to integers and get unique clusters
            cluster_labels = pca.cluster_labels.astype(int)
            unique_clusters = sorted(np.unique(cluster_labels))
            
            print(f"Processing {len(unique_clusters)} clusters...")
            
            # Add cluster information
            for cluster in unique_clusters:
                # Create boolean mask for current cluster
                cluster_mask = cluster_labels == cluster
                
                # Get prompts for current cluster
                cluster_prompts = []
                for idx, is_in_cluster in enumerate(cluster_mask):
                    if is_in_cluster:
                        prompt = list(pca.prompts.values())[idx]
                        cluster_prompts.append(prompt)
                
                # Write cluster information
                text_widget.insert(tk.END, f"\nCluster {cluster + 1} ({len(cluster_prompts)} prompts):\n")
                sample_prompts = cluster_prompts[:3] if len(cluster_prompts) > 3 else cluster_prompts
                text_widget.insert(tk.END, f"Sample prompts: {', '.join(sample_prompts)}...\n")
            
            text_widget.config(state='disabled')
            print("Cluster information panel created successfully")
            return info_frame
            
        except Exception as e:
            print(f"Error creating cluster info: {str(e)}")
            print("Debug information:")
            print(f"- Cluster labels type: {type(pca.cluster_labels)}")
            print(f"- Cluster labels shape: {pca.cluster_labels.shape}")
            print(f"- Number of prompts: {len(pca.prompts)}")
            raise

# If you want to run directly from this file
if __name__ == "__main__":
    gui = ValueDimensionPCAGui()
    gui.run()
