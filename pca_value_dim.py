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
from run_pca_naming import generate_pca_names
from llm_handler import LLMHandler
from response_parser import ResponseParser
from sklearn.decomposition import FastICA
import scipy.stats as stats

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
            
            # Check for missing definitions after loading dimensions
            if 'dim_definitions' in self.value_dims.columns:
                missing_dims, missing_count = self.analyze_missing_definitions()
                if missing_count > 0:
                    # Return special status to trigger definition generation dialog
                    return True, "missing_definitions"
            
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
            
            # Ensure we have valid current_ratings
            if self.current_ratings is None:
                print("No current ratings available, using original ratings")
                self.current_ratings = self.ratings_data.copy()
            
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
                # Combine dimension names with their definitions
                dim_texts = []
                for _, row in value_dims.iterrows():
                    dim_name = row['value dimensions']
                    definition = row.get('dim_definitions', '')
                    if pd.notna(definition) and definition.strip():
                        combined_text = f"{dim_name}: {definition}"
                    else:
                        combined_text = dim_name
                    dim_texts.append(combined_text)
                    print(f"Combined text for {dim_name}: {combined_text}")
            else:
                print("Value dimensions Series detected")
                dim_texts = value_dims.tolist()
            
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

    def analyze_pc_patterns(self, pc_index):
        """Analyze patterns in prompts for a given PC"""
        # Get component loadings
        loadings = self.pca.components_[pc_index]
        # Get sorted indices of most influential dimensions
        sorted_indices = np.argsort(np.abs(loadings))[::-1]
        
        # Create description of patterns
        pattern_desc = []
        for idx in sorted_indices[:5]:  # Top 5 most influential dimensions
            dim_name = self.dims_data[idx]
            loading = loadings[idx]
            direction = "high" if loading > 0 else "low"
            pattern_desc.append(f"{dim_name} tends to be {direction}")
            
        return "\n".join(pattern_desc)

    def analyze_missing_definitions(self) -> Tuple[pd.DataFrame, int]:
        """Analyze dimensions file for missing definitions.
        
        Returns:
            Tuple containing:
            - DataFrame with missing definitions (value dimensions without definitions)
            - Count of missing definitions
        """
        try:
            print("\nAnalyzing dimensions file for missing definitions...")
            
            if self.value_dims is None:
                raise ValueError("No dimensions data loaded")
            
            # Check for both NaN and empty string values
            missing_mask = (
                self.value_dims['dim_definitions'].isna() | 
                (self.value_dims['dim_definitions'] == '')
            )
            
            missing_dims = self.value_dims[missing_mask].copy()
            missing_count = len(missing_dims)
            
            print(f"Found {missing_count} dimensions without definitions")
            
            return missing_dims, missing_count
            
        except Exception as e:
            print(f"Error analyzing missing definitions: {str(e)}")
            raise

    def generate_missing_definitions(self, llm_handler: LLMHandler) -> pd.DataFrame:
        """Generate definitions for dimensions that are missing them.
        
        Args:
            llm_handler: Instance of LLMHandler for generating definitions
            
        Returns:
            DataFrame containing original dimensions and new definitions
        """
        try:
            # Get dimensions missing definitions
            missing_dims, count = self.analyze_missing_definitions()
            
            if count == 0:
                print("No missing definitions found")
                return pd.DataFrame()
            
            # Get list of dimension names needing definitions
            dims_to_define = missing_dims['value dimensions'].tolist()
            
            # Generate definitions using LLM
            print(f"Generating definitions for {len(dims_to_define)} dimensions...")
            generated_defs = llm_handler.generate_value_dimension_definitions(dims_to_define)
            
            # Create DataFrame with results
            results_df = pd.DataFrame(generated_defs)
            results_df.columns = ['value dimensions', 'generated_definition']
            
            # Merge with original missing dimensions to preserve order and other columns
            final_df = missing_dims.merge(
                results_df,
                on='value dimensions',
                how='left'
            )
            
            return final_df
            
        except Exception as e:
            print(f"Error generating missing definitions: {str(e)}")
            raise

class ValueDimensionICA:
    def __init__(self):
        self.ica = None
        self.ica_components = None
        self.mixing_matrix = None
        self.convergence_info = None
        self.kurtosis_scores = None
        
    def perform_ica(self, data, n_components=None):
        """Perform ICA on the data"""
        try:
            print("\nPerforming ICA analysis...")
            if data is None:
                raise ValueError("No data provided for ICA analysis")
                
            # Initialize FastICA
            self.ica = FastICA(
                n_components=n_components,
                random_state=42,
                max_iter=1000,
                tol=0.01
            )
            
            # Fit and transform the data
            print("Fitting ICA model...")
            self.ica_components = self.ica.fit_transform(data)
            self.mixing_matrix = self.ica.mixing_
            
            # Calculate kurtosis for each component
            print("Calculating kurtosis scores...")
            self.kurtosis_scores = [stats.kurtosis(comp) for comp in self.ica_components.T]
            
            # Store convergence information
            self.convergence_info = {
                'n_iter': self.ica.n_iter_,
                'converged': self.ica.n_iter_ < 1000  # True if converged before max_iter
            }
            
            print("ICA analysis completed successfully")
            return True
            
        except Exception as e:
            print(f"Error performing ICA: {str(e)}")
            return False

# Then, the GUI class
class ValueDimensionPCAGui(ValueDimensionPCA):
    # Add the PCA class as a class attribute
    PCA_Class = ValueDimensionPCA
    
    def __init__(self):
        super().__init__()
        self.root = tk.Tk()
        self.root.title("Value Dimension Analysis (PCA & ICA)")
        self.pca_instance = None
        self.ica_instance = ValueDimensionICA()
        self.ratings_type = tk.StringVar(value="original")
        
        # Initialize LLMHandler with configuration
        self.llm_handler = LLMHandler({
            'openai_api_key': os.getenv('OPENAI_API_KEY'),  # Make sure this env variable is set
            'model': 'gpt-3.5-turbo',
            'temperature': 0.7,
            'max_tokens': 150
        })
        
        # Add plot type variables
        self.pca_plot_type = tk.StringVar(value="scatter")
        self.ica_plot_type = tk.StringVar(value="scatter")
        
        # Create main container for side-by-side analysis
        self.analysis_frame = ttk.Frame(self.root)
        self.analysis_frame.pack(fill="both", expand=True)
        
        # Create frames for PCA and ICA
        self.pca_frame = ttk.LabelFrame(self.analysis_frame, text="PCA Analysis")
        self.pca_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        self.ica_frame = ttk.LabelFrame(self.analysis_frame, text="ICA Analysis")
        self.ica_frame.pack(side="right", fill="both", expand=True, padx=5)
        
        # Initialize data structures
        self.ratings_paths = []
        self.dims_paths = []
        self.ratings_count = tk.StringVar(value="1")
        self.dims_count = tk.StringVar(value="1")
        
        # Create initial widgets
        self.create_initial_widgets()

    def load_data(self):
        """Handle data loading for both PCA and ICA"""
        try:
            # Initialize instances if not already done
            if self.pca_instance is None:
                self.pca_instance = ValueDimensionPCA()
            if self.ica_instance is None:
                self.ica_instance = ValueDimensionICA()
            
            # Load data for PCA
            success_pca, message_pca = self.pca_instance.load_data(
                self.ratings_paths, 
                self.dims_paths
            )
            
            if success_pca:
                # Perform ICA on the same data
                success_ica = self.ica_instance.perform_ica(
                    self.pca_instance.current_ratings
                )
                
                if success_ica:
                    messagebox.showinfo("Success", "Data loaded and analyzed successfully")
                    self.run_button.config(state=tk.NORMAL)
                else:
                    messagebox.showerror("Error", "ICA analysis failed")
            else:
                messagebox.showerror("Error", f"Failed to load data: {message_pca}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")

    def show_visualization(self, pca):
        """Modified to show both PCA and ICA visualizations"""
        try:
            print("Creating visualization window...")
            if not hasattr(self, 'viz_window'):
                self.viz_window = tk.Toplevel(self.root)
                self.viz_window.title("Analysis Visualization")
                
                # Add radio buttons for ratings type selection
                ratings_frame = ttk.LabelFrame(self.viz_window, text="Ratings Type")
                ratings_frame.pack(fill="x", padx=10, pady=5)
                
                def on_ratings_change():
                    try:
                        if self.ratings_type.get() == "cosine":
                            print("Switching to cosine similarity scores...")
                            # Calculate cosine similarities if not already done
                            if not hasattr(self.pca_instance, 'cosine_ratings') or self.pca_instance.cosine_ratings is None:
                                success = self.pca_instance.calculate_cosine_similarity(
                                    self.pca_instance.prompts,
                                    self.pca_instance.value_dims
                                )
                                if not success:
                                    raise ValueError("Failed to calculate cosine similarities")
                            
                            # Update current_ratings with cosine similarities
                            self.pca_instance.current_ratings = self.pca_instance.cosine_ratings.copy()
                        else:
                            print("Switching to original ratings...")
                            self.pca_instance.current_ratings = self.pca_instance.ratings_data.copy()
                        
                        # Perform PCA with updated ratings
                        if self.pca_instance.perform_pca():
                            self.update_plot(self.pca_instance)
                        else:
                            raise ValueError("Failed to perform PCA with new ratings")
                            
                    except Exception as e:
                        print(f"Error switching ratings type: {str(e)}")
                        messagebox.showerror("Error", f"Failed to switch ratings type: {str(e)}")
                        # Revert to original ratings if there's an error
                        self.ratings_type.set("original")
                    
                ttk.Radiobutton(
                    ratings_frame,
                    text="Original Ratings",
                    variable=self.ratings_type,
                    value="original",
                    command=on_ratings_change
                ).pack(side="left", padx=5)
                
                ttk.Radiobutton(
                    ratings_frame,
                    text="Cosine Similarity Scores",
                    variable=self.ratings_type,
                    value="cosine",
                    command=on_ratings_change
                ).pack(side="left", padx=5)
            
            # ... rest of visualization code ...
            
            # Create main scrollable container
            main_canvas = tk.Canvas(self.viz_window)
            scrollbar = ttk.Scrollbar(self.viz_window, orient="vertical", command=main_canvas.yview)
            scrollable_frame = ttk.Frame(main_canvas)
            
            # Configure scrolling
            scrollable_frame.bind(
                "<Configure>",
                lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
            )
            main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            main_canvas.configure(yscrollcommand=scrollbar.set)
            
            # Create side-by-side frames
            pca_viz_frame = ttk.LabelFrame(scrollable_frame, text="PCA Results")
            pca_viz_frame.pack(side="left", fill="both", expand=True, padx=5)
            
            ica_viz_frame = ttk.LabelFrame(scrollable_frame, text="ICA Results")
            ica_viz_frame.pack(side="right", fill="both", expand=True, padx=5)
            
            # Store instances
            self.pca_instance = pca
            
            # Create controls and panels for both analyses
            self.create_visualization_controls(pca_viz_frame, analysis_type="pca")
            self.create_visualization_controls(ica_viz_frame, analysis_type="ica")
            
            self.create_summary_panel(pca_viz_frame, analysis_type="pca")
            self.create_summary_panel(ica_viz_frame, analysis_type="ica")
            
            # Create figure frames
            self.pca_fig_frame = ttk.Frame(pca_viz_frame)
            self.pca_fig_frame.pack(fill="both", expand=True, pady=5)
            
            self.ica_fig_frame = ttk.Frame(ica_viz_frame)
            self.ica_fig_frame.pack(fill="both", expand=True, pady=5)
            
            # Update displays
            self.update_summary(analysis_type="pca")
            self.update_summary(analysis_type="ica")
            self.update_plot(self.pca_instance)
            
            # Pack scrollbar and canvas
            scrollbar.pack(side="right", fill="y")
            main_canvas.pack(side="left", fill="both", expand=True)
            
            # Configure window size
            self.viz_window.geometry("1600x800")  # Wider to accommodate both analyses
            
            # Enable mouse wheel scrolling
            def _on_mousewheel(event):
                main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            
        except Exception as e:
            print(f"Error in show_visualization: {str(e)}")
            raise

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
        """Modified proceed method to handle missing definitions"""
        try:
            print("Starting proceed method...")
            
            # Initialize PCA instance if it doesn't exist
            if not hasattr(self, 'pca_instance') or self.pca_instance is None:
                self.pca_instance = ValueDimensionPCA()
            
            # Load and prepare all data first
            success, message = self.pca_instance.load_data(self.ratings_paths, self.dims_paths)
            
            if not success:
                messagebox.showerror("Error", message)
                return
                
            # Check if we need to handle missing definitions
            if message == "missing_definitions":
                response = messagebox.askyesno(
                    "Missing Definitions",
                    "Some value dimensions are missing definitions. Would you like to generate them using AI?"
                )
                if response:
                    self.show_definition_generation_dialog()
                    return
            
            # Continue with normal PCA process
            print("Performing initial PCA...")
            if not self.pca_instance.perform_pca():
                messagebox.showerror("Error", "PCA calculation failed")
                return
                
            # Initialize and perform ICA
            if not hasattr(self, 'ica_instance') or self.ica_instance is None:
                self.ica_instance = ValueDimensionICA()
            
            print("Performing ICA...")
            if not self.ica_instance.perform_ica(self.pca_instance.current_ratings):
                messagebox.showerror("Error", "ICA calculation failed")
                return
            
            # Show visualization
            print("Showing visualization...")
            self.show_visualization(self.pca_instance)
            
        except Exception as e:
            print(f"Error in proceed: {str(e)}")
            messagebox.showerror("Error", f"Error processing files: {str(e)}")
    
    def create_summary_panel(self, parent, analysis_type="pca"):
        """Create summary panel for both PCA and ICA results"""
        try:
            title = "PCA Summary" if analysis_type == "pca" else "ICA Summary"
            summary_frame = ttk.LabelFrame(parent, text=title, padding="10")
            summary_frame.pack(fill="x", padx=10, pady=5)

            # Create text widget for summary
            text_widget = tk.Text(summary_frame, height=15, width=60)
            text_widget.pack(fill="both", expand=True)
            
            # Add scrollbar
            scrollbar = ttk.Scrollbar(summary_frame, orient="vertical", command=text_widget.yview)
            scrollbar.pack(side="right", fill="y")
            text_widget.configure(yscrollcommand=scrollbar.set)

            # Store text widget reference
            setattr(self, f"{analysis_type}_summary_text", text_widget)

            # Add component naming section
            names_frame = ttk.Frame(summary_frame)
            names_frame.pack(fill="x", pady=5)
            
            # Add status label
            status_label = ttk.Label(
                names_frame,
                text="Ready to generate names",
                foreground="gray"
            )
            status_label.pack(pady=2)
            setattr(self, f"{analysis_type}_status_label", status_label)
            
            # Generate Names button
            name_gen_button = ttk.Button(
                names_frame,
                text=f"Generate {analysis_type.upper()} Names",
                command=lambda: self.generate_component_names(analysis_type)
            )
            name_gen_button.pack(pady=5)
            setattr(self, f"{analysis_type}_name_gen_button", name_gen_button)
            
            # Text widget for displaying generated names
            names_text = tk.Text(summary_frame, height=10, width=60)
            names_text.pack(fill="both", expand=True)
            names_text.pack_forget()  # Initially hidden
            setattr(self, f"{analysis_type}_names_text", names_text)

        except Exception as e:
            print(f"Error creating summary panel: {str(e)}")
            raise

    def update_summary(self, analysis_type="pca"):
        """Update summary with current results"""
        try:
            # Get appropriate text widget
            text_widget = getattr(self, f"{analysis_type}_summary_text")
            text_widget.delete(1.0, tk.END)
            
            if analysis_type == "pca":
                self.update_pca_summary(text_widget)
            else:
                self.update_ica_summary(text_widget)
                
        except Exception as e:
            print(f"Error updating summary: {str(e)}")
            raise

    def update_ica_summary(self, text_widget):
        """Update ICA-specific summary information"""
        try:
            # Clear existing text
            text_widget.delete(1.0, tk.END)
            
            # Data preprocessing info
            text_widget.insert(tk.END, "Data Preprocessing:\n", "heading")
            text_widget.insert(tk.END, "- Standardization: StandardScaler\n")
            text_widget.insert(tk.END, f"- Input dimensions: {len(self.pca_instance.original_dims)}\n\n")

            # Convergence information
            text_widget.insert(tk.END, "Convergence Information:\n", "heading")
            if hasattr(self.ica_instance, 'convergence_info') and self.ica_instance.convergence_info:
                conv_info = self.ica_instance.convergence_info
                text_widget.insert(tk.END, f"- Iterations: {conv_info.get('n_iter', 'N/A')}\n")
                text_widget.insert(tk.END, f"- Converged: {conv_info.get('converged', 'N/A')}\n\n")
            else:
                text_widget.insert(tk.END, "- Convergence information not available\n\n")

            # Component contributions section
            text_widget.insert(tk.END, "Component Contributions from Mixing Matrix:\n", "heading")
            n_top = int(self.ica_top_n.get())
            
            # Get mixing matrix and ensure it's properly oriented
            mixing_matrix = self.ica_instance.mixing_matrix
            n_components = mixing_matrix.shape[1]
            
            for i in range(n_components):
                # Get mixing coefficients for this component
                coefficients = mixing_matrix[:, i]
                kurt = self.ica_instance.kurtosis_scores[i]
                
                text_widget.insert(tk.END, f"\nIC{i+1} (kurtosis: {kurt:.3f}):\n", "subheading")
                
                # Sort indices by absolute coefficient values
                sorted_indices = np.argsort(np.abs(coefficients))[::-1]
                
                # Separate positive and negative contributions
                pos_indices = [idx for idx in sorted_indices[:n_top] if coefficients[idx] > 0]
                neg_indices = [idx for idx in sorted_indices[:n_top] if coefficients[idx] < 0]
                
                # Show positive contributions
                if pos_indices:
                    text_widget.insert(tk.END, "  Positive contributions:\n", "italic")
                    for idx in pos_indices:
                        if idx < len(self.pca_instance.original_dims):  # Ensure index is valid
                            dim = self.pca_instance.original_dims[idx]
                            coef = coefficients[idx]
                            text_widget.insert(tk.END, f"    {dim}: {coef:.3f}\n")
                
                # Show negative contributions
                if neg_indices:
                    text_widget.insert(tk.END, "  Negative contributions:\n", "italic")
                    for idx in neg_indices:
                        if idx < len(self.pca_instance.original_dims):  # Ensure index is valid
                            dim = self.pca_instance.original_dims[idx]
                            coef = coefficients[idx]
                            text_widget.insert(tk.END, f"    {dim}: {coef:.3f}\n")

            # Kurtosis summary
            text_widget.insert(tk.END, "\nKurtosis Summary:\n", "heading")
            text_widget.insert(tk.END, f"- Mean kurtosis: {np.mean(self.ica_instance.kurtosis_scores):.3f}\n")
            text_widget.insert(tk.END, f"- Max kurtosis: {np.max(self.ica_instance.kurtosis_scores):.3f}\n")
            text_widget.insert(tk.END, f"- Min kurtosis: {np.min(self.ica_instance.kurtosis_scores):.3f}\n\n")

            # Software information
            text_widget.insert(tk.END, "Software Information:\n", "heading")
            text_widget.insert(tk.END, "- scikit-learn FastICA\n")
            text_widget.insert(tk.END, "- pandas DataFrame\n")
            text_widget.insert(tk.END, "- numpy arrays\n")

            # Apply tags for formatting
            text_widget.tag_configure("heading", font=("TkDefaultFont", 10, "bold"))
            text_widget.tag_configure("subheading", font=("TkDefaultFont", 9, "bold"))
            text_widget.tag_configure("italic", font=("TkDefaultFont", 9, "italic"))
            
        except Exception as e:
            print(f"Error updating ICA summary: {str(e)}")
            text_widget.insert(tk.END, f"Error updating ICA summary: {str(e)}\n")

    def generate_component_names(self, analysis_type="pca"):
        """Generate names for components using run_pca_naming module"""
        try:
            # Update status
            status_label = getattr(self, f"{analysis_type}_status_label")
            status_label.config(text="Generating component names...")
            self.root.update()

            # Initialize the appropriate instance
            if analysis_type == "pca":
                if not hasattr(self, 'pca_instance') or self.pca_instance is None:
                    raise ValueError("PCA analysis has not been performed yet")
                instance = self.pca_instance
            else:  # ICA
                if not hasattr(self, 'ica_instance') or self.ica_instance is None:
                    raise ValueError("ICA analysis has not been performed yet")
                instance = self.ica_instance

            # Generate names based on analysis type
            if analysis_type == "pca":
                # Format PCA results as expected by generate_pca_names
                results_dict = {}
                for i, comp in enumerate(instance.pca.components_):
                    sorted_indices = abs(comp).argsort()[::-1]
                    top_dims = [instance.original_dims[idx] for idx in sorted_indices[:5]]
                    
                    high_indices = comp.argsort()[::-1][:5]
                    low_indices = comp.argsort()[:5]
                    
                    results_dict[f'pc_{i+1}_top_dims'] = ', '.join(map(str, top_dims))
                    results_dict[f'pc_{i+1}_high_prompts'] = ', '.join([instance.original_dims[idx] for idx in high_indices])
                    results_dict[f'pc_{i+1}_low_prompts'] = ', '.join([instance.original_dims[idx] for idx in low_indices])
                
                prompts_df = pd.DataFrame({'prompt': instance.original_dims})
                
                from run_pca_naming import generate_pca_names
                results = generate_pca_names(
                    pca_results=results_dict,
                    prompts_df=prompts_df,
                    n_components=min(8, len(instance.pca.components_))
                )
            else:  # ICA
                # Format ICA results
                results_dict = {}
                n_features = len(self.pca_instance.original_dims)
                
                for i in range(instance.mixing_matrix.shape[1]):
                    coeffs = instance.mixing_matrix[:, i]
                    kurt_score = instance.kurtosis_scores[i]
                    
                    # Get indices for top dimensions (both positive and negative)
                    sorted_indices = abs(coeffs).argsort()[::-1]
                    sorted_indices = sorted_indices[sorted_indices < n_features]
                    top_dims = [f"{self.pca_instance.original_dims[idx]} ({coeffs[idx]:.3f})" 
                               for idx in sorted_indices[:5]]
                    
                    # Get indices for high and low loading prompts
                    high_indices = coeffs.argsort()[::-1][:5]
                    low_indices = coeffs.argsort()[:5]
                    
                    # Store formatted results
                    results_dict[f'ic_{i+1}_top_dims'] = ', '.join(top_dims)
                    results_dict[f'ic_{i+1}_high_prompts'] = ', '.join(
                        [f"{self.pca_instance.original_dims[idx]} ({coeffs[idx]:.3f})" 
                         for idx in high_indices if idx < n_features]
                    )
                    results_dict[f'ic_{i+1}_low_prompts'] = ', '.join(
                        [f"{self.pca_instance.original_dims[idx]} ({coeffs[idx]:.3f})" 
                         for idx in low_indices if idx < n_features]
                    )
                    results_dict[f'ic_{i+1}_kurtosis'] = kurt_score
                
                prompts_df = pd.DataFrame({'prompt': self.pca_instance.original_dims})
                
                from run_pca_naming import generate_ica_names
                results = generate_ica_names(
                    ica_results=results_dict,
                    prompts_df=prompts_df,
                    n_components=min(8, instance.mixing_matrix.shape[1])
                )

            if results is not None:
                self.display_component_names(results, analysis_type)
                messagebox.showinfo(
                    "Success", 
                    f"Generated names for {len(results)} {analysis_type.upper()} components"
                )
            else:
                messagebox.showerror(
                    "Error", 
                    f"Failed to generate {analysis_type.upper()} component names"
                )

        except Exception as e:
            print(f"Error generating component names: {str(e)}")
            print("\nDebug information:")
            print(f"Analysis type: {analysis_type}")
            
            if analysis_type == "ica":
                if hasattr(self, 'ica_instance') and self.ica_instance is not None:
                    print("ICA instance exists")
                    if hasattr(self.ica_instance, 'ica_components'):
                        print(f"ICA components type: {type(self.ica_instance.ica_components)}")
                        print(f"ICA components shape: {self.ica_instance.ica_components.shape}")
                    if hasattr(self.ica_instance, 'mixing_matrix'):
                        print(f"Mixing matrix type: {type(self.ica_instance.mixing_matrix)}")
                        print(f"Mixing matrix shape: {self.ica_instance.mixing_matrix.shape}")
                    if hasattr(self.ica_instance, 'kurtosis_scores'):
                        print(f"Kurtosis scores type: {type(self.ica_instance.kurtosis_scores)}")
                        print(f"Kurtosis scores length: {len(self.ica_instance.kurtosis_scores)}")
                else:
                    print("ICA instance not found")
            
            messagebox.showerror("Error", f"Name generation failed: {str(e)}")
        finally:
            status_label.config(text="Ready")

    def create_visualization_controls(self, parent, analysis_type="pca"):
        """Update visualization controls to handle both PCA and ICA"""
        try:
            print(f"Creating control frame for {analysis_type.upper()}...")
            control_frame = ttk.Frame(parent)
            control_frame.pack(fill="x", padx=10, pady=5)
            
            # Add controls for number of top variables to show
            var_control_frame = ttk.LabelFrame(control_frame, text="Summary Controls", padding="5")
            var_control_frame.pack(fill="x", pady=5)
            
            ttk.Label(var_control_frame, text="Top variables to show:").pack(side="left", padx=5)
            
            # Create and store the variable
            if analysis_type == "pca":
                self.pca_top_n = tk.StringVar(value="5")
                top_n_var = self.pca_top_n
            else:
                self.ica_top_n = tk.StringVar(value="5")
                top_n_var = self.ica_top_n
            
            top_n_entry = ttk.Entry(var_control_frame, textvariable=top_n_var, width=3)
            top_n_entry.pack(side="left")
            
            # Add update button
            ttk.Button(
                var_control_frame,
                text="Update Summary",
                command=lambda: self.update_summary(analysis_type)
            ).pack(side="left", padx=5)
            
            # Ratings type toggle (only for PCA)
            if analysis_type == "pca":
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
            
            # Component selection
            if analysis_type == "pca":
                self.pc_x = tk.StringVar(value="1")
                self.pc_y = tk.StringVar(value="2")
                prefix = "PC"
                plot_type_var = self.pca_plot_type
            else:
                self.ic_x = tk.StringVar(value="1")
                self.ic_y = tk.StringVar(value="2")
                prefix = "IC"
                plot_type_var = self.ica_plot_type
            
            x_var = self.pc_x if analysis_type == "pca" else self.ic_x
            y_var = self.pc_y if analysis_type == "pca" else self.ic_y
            
            ttk.Label(plot_controls, text=f"{prefix} X:").pack(side="left", padx=5)
            ttk.Entry(plot_controls, textvariable=x_var, width=3).pack(side="left")
            ttk.Label(plot_controls, text=f"{prefix} Y:").pack(side="left", padx=5)
            ttk.Entry(plot_controls, textvariable=y_var, width=3).pack(side="left")
            
            # Point size control
            self.point_size = tk.StringVar(value="50")
            ttk.Label(plot_controls, text="Point Size:").pack(side="left", padx=5)
            ttk.Entry(plot_controls, textvariable=self.point_size, width=4).pack(side="left")
            
            # Plot type selection
            plot_types = ["scatter", "matrix", "heatmap"]
            if analysis_type == "ica":
                plot_types.extend(["kurtosis", "signals", "mixing"])
            
            for plot_type in plot_types:
                ttk.Radiobutton(
                    plot_controls,
                    text=plot_type.capitalize(),
                    variable=plot_type_var,
                    value=plot_type,
                    command=lambda t=analysis_type: self.update_plot(t)
                ).pack(side="left", padx=5)
            
            print("Control frame created successfully")
            
        except Exception as e:
            print(f"Error creating visualization controls: {str(e)}")
            raise

    def update_plot(self, analysis_type):
        """Update visualization based on current settings and analysis type"""
        try:
            # Get plot type using the correct attribute name
            plot_type = self.pca_plot_type.get() if analysis_type == "pca" else self.ica_plot_type.get()
            
            # Clear previous plot
            frame = self.pca_fig_frame if analysis_type == "pca" else self.ica_fig_frame
            for widget in frame.winfo_children():
                widget.destroy()
            
            instance = self.pca_instance if analysis_type == "pca" else self.ica_instance
            
            if analysis_type == "pca":
                if plot_type == "scatter":
                    pc1 = int(self.pc_x.get()) - 1
                    pc2 = int(self.pc_y.get()) - 1
                    self.create_scatter_plot(instance, pc1, pc2)
                elif plot_type == "matrix":
                    self.create_matrix_plot(instance, "pca")
                elif plot_type == "heatmap":
                    self.create_heatmap_plot(instance, "pca")
            else:  # ICA
                if plot_type == "scatter":
                    ic1 = int(self.ic_x.get()) - 1
                    ic2 = int(self.ic_y.get()) - 1
                    self.create_ica_scatter_plot(ic1, ic2)
                elif plot_type == "matrix":
                    self.create_matrix_plot(instance, "ica")
                elif plot_type == "heatmap":
                    self.create_heatmap_plot(instance, "ica")
                elif plot_type == "kurtosis":
                    self.create_kurtosis_plot(instance)
                elif plot_type == "signals":
                    self.create_signals_plot(instance)
                elif plot_type == "mixing":
                    self.create_mixing_plot(instance)
                
        except Exception as e:
            print(f"Error updating plot: {str(e)}")
            messagebox.showerror("Error", f"Error updating plot: {str(e)}")

    def run(self):
        """Main method to run the application"""
        try:
            print("\nStarting Value Dimension Analysis application...")
            
            # Initialize logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            logging.info("Application started")
            
            # Set error handling
            def handle_exception(exc_type, exc_value, exc_traceback):
                logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
                messagebox.showerror("Error", f"An unexpected error occurred: {str(exc_value)}")
            
            sys.excepthook = handle_exception
            
            # Run the main loop
            self.root.mainloop()
            
        except Exception as e:
            logging.error(f"Error in main run method: {str(e)}")
            raise
        finally:
            logging.info("Application closed")

    def perform_analysis(self):
        """Perform both PCA and ICA analysis"""
        try:
            print("\nPerforming analysis...")
            
            # Perform PCA
            if not self.pca_instance.perform_pca():
                raise Exception("PCA analysis failed")
            
            # Perform ICA
            if not self.ica_instance.perform_ica(self.pca_instance.current_ratings):
                raise Exception("ICA analysis failed")
            
            # Show visualizations
            self.show_visualization(self.pca_instance)
            
            print("Analysis completed successfully")
            return True
            
        except Exception as e:
            print(f"Error performing analysis: {str(e)}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            return False

    def display_component_names(self, results_df, analysis_type="pca"):
        """Display the generated names in the GUI"""
        try:
            # Get appropriate text widget
            names_text = getattr(self, f"{analysis_type}_names_text")
            names_text.pack(fill="both", expand=True)  # Make visible
            names_text.config(state="normal")  # Allow editing temporarily
            names_text.delete(1.0, tk.END)
            
            # Display results
            prefix = "PC" if analysis_type == "pca" else "IC"
            for _, row in results_df.iterrows():
                # Component number
                comp_num = row.get('component_number', row.get('pc_num', '?'))
                
                # Component name and description
                name = row.get('component_name', row.get('name', 'Unnamed Component'))
                description = row.get('description', row.get('explanation', 'No description available'))
                
                # Format and insert the component information
                names_text.insert(tk.END, f"\n{prefix} {comp_num}:", "bold")
                names_text.insert(tk.END, f" {name}\n", "bold")
                names_text.insert(tk.END, f"{description}\n")
                
                # Add loadings if available
                loadings = row.get('top_loadings', row.get('top_vars', []))
                if loadings:
                    names_text.insert(tk.END, "\nTop contributing variables:\n", "italic")
                    for loading in loadings:
                        names_text.insert(tk.END, f"- {loading}\n")
                
                names_text.insert(tk.END, "\n" + "-"*50 + "\n")
            
            # Configure tags
            names_text.tag_configure("bold", font=("TkDefaultFont", 10, "bold"))
            names_text.tag_configure("italic", font=("TkDefaultFont", 10, "italic"))
            
            # Make text widget read-only
            names_text.config(state="disabled")
            
        except Exception as e:
            print(f"Error displaying component names: {str(e)}")
            raise

    def update_ratings_type(self):
        """Update the current ratings based on user selection"""
        try:
            print(f"\nUpdating ratings type to: {self.ratings_type.get()}")
            
            # Update current ratings based on selection
            if self.ratings_type.get() == "cosine":
                self.pca_instance.current_ratings = self.pca_instance.cosine_ratings
                print("Using cosine similarity ratings")
            else:
                self.pca_instance.current_ratings = self.pca_instance.ratings_data
                print("Using original ratings")
            
            # Rerun PCA with new ratings
            if self.pca_instance.perform_pca():
                # Update visualizations
                self.update_summary(analysis_type="pca")
                self.update_plot("pca")
                print("Successfully updated analysis with new ratings")
            else:
                messagebox.showerror("Error", "Failed to update PCA with new ratings")
                
        except Exception as e:
            print(f"Error updating ratings type: {str(e)}")
            messagebox.showerror("Error", f"Failed to update ratings: {str(e)}")

    def update_pca_summary(self, text_widget):
        """Update PCA-specific summary information"""
        try:
            # Data preprocessing info
            text_widget.insert(tk.END, "Data Preprocessing:\n", "heading")
            text_widget.insert(tk.END, "- Standardization: StandardScaler\n")
            text_widget.insert(tk.END, f"- Input dimensions: {len(self.pca_instance.original_dims)}\n\n")

            # Explained variance information
            text_widget.insert(tk.END, "Explained Variance:\n", "heading")
            exp_var = self.pca_instance.explained_variance_ratio_
            cumulative_var = np.cumsum(exp_var)
            
            for i, (var, cum_var) in enumerate(zip(exp_var, cumulative_var)):
                text_widget.insert(tk.END, 
                    f"PC{i+1}: {var:.3f} ({cum_var:.3f} cumulative)\n")
            text_widget.insert(tk.END, "\n")

            # Variance contributions
            text_widget.insert(tk.END, "Variable Contributions:\n", "heading")
            var_contrib = self.pca_instance.variance_contributions
            if var_contrib is not None:
                sorted_contrib = var_contrib.sort_values(ascending=False)
                for dim, contrib in sorted_contrib.items():
                    text_widget.insert(tk.END, f"{dim}: {contrib:.2f}%\n")
            text_widget.insert(tk.END, "\n")

            # Component information
            text_widget.insert(tk.END, "Component Information:\n", "heading")
            text_widget.insert(tk.END, f"- Number of components: {self.pca_instance.pca.n_components_}\n")
            text_widget.insert(tk.END, f"- Shape of loadings: {self.pca_instance.components_.shape}\n\n")

            # Software information
            text_widget.insert(tk.END, "Software Information:\n", "heading")
            text_widget.insert(tk.END, "- scikit-learn PCA\n")
            text_widget.insert(tk.END, "- pandas DataFrame\n")
            text_widget.insert(tk.END, "- numpy arrays\n")

            # Apply tags for formatting
            text_widget.tag_configure("heading", font=("TkDefaultFont", 10, "bold"))
            
            # Add component contributions section
            text_widget.insert(tk.END, "\nComponent Contributions:\n", "heading")
            n_top = int(self.pca_top_n.get())
            
            for i, comp in enumerate(self.pca_instance.components_):
                var_exp = self.pca_instance.explained_variance_ratio_[i]
                text_widget.insert(tk.END, f"\nPC{i+1} ({var_exp:.2%} variance explained):\n", "subheading")
                
                # Get indices sorted by absolute loading values
                sorted_indices = np.argsort(np.abs(comp))[::-1]
                
                # Separate positive and negative contributions
                pos_indices = [idx for idx in sorted_indices[:n_top] if comp[idx] > 0]
                neg_indices = [idx for idx in sorted_indices[:n_top] if comp[idx] < 0]
                
                # Show positive contributions
                if pos_indices:
                    text_widget.insert(tk.END, "  Positive contributions:\n", "italic")
                    for idx in pos_indices:
                        dim = self.pca_instance.original_dims[idx]
                        loading = comp[idx]
                        text_widget.insert(tk.END, f"    {dim}: {loading:.3f}\n")
                
                # Show negative contributions
                if neg_indices:
                    text_widget.insert(tk.END, "  Negative contributions:\n", "italic")
                    for idx in neg_indices:
                        dim = self.pca_instance.original_dims[idx]
                        loading = comp[idx]
                        text_widget.insert(tk.END, f"    {dim}: {loading:.3f}\n")
            
        except Exception as e:
            print(f"Error updating PCA summary: {str(e)}")
            raise

    def create_scatter_plot(self, instance, x_idx, y_idx):
        """Create scatter plot for PCA components"""
        try:
            # Clear previous plot
            for widget in self.pca_fig_frame.winfo_children():
                widget.destroy()
            
            # Create figure
            fig = Figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            
            # Get data
            x_data = instance.pca_ratings[:, x_idx]
            y_data = instance.pca_ratings[:, y_idx]
            
            # Create scatter plot
            ax.scatter(x_data, y_data, alpha=0.6, s=float(self.point_size.get()))
            
            # Customize plot
            ax.set_xlabel(f"PC{x_idx + 1}")
            ax.set_ylabel(f"PC{y_idx + 1}")
            ax.set_title("PCA Component Scatter Plot")
            ax.grid(True, alpha=0.3)
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=self.pca_fig_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, self.pca_fig_frame)
            toolbar.update()
            
        except Exception as e:
            print(f"Error creating scatter plot: {str(e)}")
            raise

    def create_matrix_plot(self, instance, analysis_type="pca"):
        """Create correlation matrix plot for PCA or ICA"""
        try:
            frame = self.pca_fig_frame if analysis_type == "pca" else self.ica_fig_frame
            
            # Clear previous plot
            for widget in frame.winfo_children():
                widget.destroy()
            
            # Get appropriate data and dimensions
            if analysis_type == "pca":
                data = instance.pca.components_
                dims = instance.original_dims
            else:
                data = instance.mixing_matrix.T
                dims = self.pca_instance.original_dims  # Use PCA instance's dimensions
            
            # Debug print
            print(f"\nDebug information for {analysis_type} matrix plot:")
            print(f"Data shape: {data.shape}")
            print(f"Number of dimensions: {len(dims)}")
            
            # Create figure with subplots
            n_components = data.shape[0]
            n_cols = min(4, n_components)
            n_rows = (n_components + n_cols - 1) // n_cols
            
            fig = Figure(figsize=(12, 3*n_rows))
            
            for i in range(n_components):
                ax = fig.add_subplot(n_rows, n_cols, i+1)
                
                # Get component data
                component_data = data[i]
                
                # Ensure we don't exceed the number of dimensions
                valid_indices = np.where(np.arange(len(component_data)) < len(dims))[0]
                
                # Sort valid indices by absolute values
                sorted_indices = valid_indices[np.argsort(np.abs(component_data[valid_indices]))[::-1]]
                
                # Take top N dimensions (where N is min of 12 and available valid dimensions)
                n_dims = min(12, len(sorted_indices))
                top_indices = sorted_indices[:n_dims]
                
                if len(top_indices) == 0:
                    print(f"Warning: No valid indices for component {i+1}")
                    continue
                    
                # Plot bars
                bars = ax.bar(range(len(top_indices)), component_data[top_indices])
                
                # Color bars based on value
                for bar in bars:
                    if bar.get_height() < 0:
                        bar.set_color('red')
                    else:
                        bar.set_color('blue')
                
                # Customize plot
                ax.set_title(f"{'PC' if analysis_type == 'pca' else 'IC'}{i+1}")
                ax.set_xticks(range(len(top_indices)))
                ax.set_xticklabels([dims[idx] for idx in top_indices], rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, frame)
            toolbar.update()
            
        except Exception as e:
            print(f"Error creating {analysis_type.upper()} matrix plot: {str(e)}")
            print(f"Additional debug info:")
            if analysis_type == "ica":
                print(f"ICA mixing matrix shape: {instance.mixing_matrix.shape}")
                print(f"Number of dimensions in PCA: {len(self.pca_instance.original_dims)}")
            raise

    def create_kurtosis_plot(self, instance):
        """Create kurtosis plot for ICA components"""
        try:
            # Clear previous plot
            for widget in self.ica_fig_frame.winfo_children():
                widget.destroy()
            
            # Create figure
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            # Plot kurtosis values
            x = range(1, len(instance.kurtosis_scores) + 1)
            ax.bar(x, instance.kurtosis_scores)
            
            # Customize plot
            ax.set_xlabel("Independent Component")
            ax.set_ylabel("Kurtosis")
            ax.set_title("ICA Component Kurtosis Values")
            ax.grid(True, alpha=0.3)
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=self.ica_fig_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, self.ica_fig_frame)
            toolbar.update()
            
        except Exception as e:
            print(f"Error creating kurtosis plot: {str(e)}")
            raise

    def create_signals_plot(self, instance):
        """Create signals plot for ICA components"""
        try:
            # Clear previous plot
            for widget in self.ica_fig_frame.winfo_children():
                widget.destroy()
            
            # Create figure
            fig = Figure(figsize=(12, 8))
            
            # Plot each IC signal
            n_components = instance.ica_components.shape[1]
            for i in range(n_components):
                ax = fig.add_subplot(n_components, 1, i+1)
                ax.plot(instance.ica_components[:, i])
                ax.set_ylabel(f"IC{i+1}")
                ax.grid(True, alpha=0.3)
            
            # Adjust layout
            fig.suptitle("ICA Component Signals")
            fig.tight_layout()
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=self.ica_fig_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, self.ica_fig_frame)
            toolbar.update()
            
        except Exception as e:
            print(f"Error creating signals plot: {str(e)}")
            raise

    def create_mixing_plot(self, instance):
        """Create mixing matrix plot for ICA"""
        try:
            # Clear previous plot
            for widget in self.ica_fig_frame.winfo_children():
                widget.destroy()
            
            # Create figure
            fig = Figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            
            # Plot mixing matrix
            im = ax.imshow(instance.mixing_matrix, cmap='RdBu_r', aspect='auto')
            
            # Add colorbar
            fig.colorbar(im)
            
            # Customize plot
            ax.set_title("ICA Mixing Matrix")
            ax.set_xlabel("Independent Components")
            ax.set_ylabel("Features")
            
            # Set ticks
            ax.set_xticks(range(instance.mixing_matrix.shape[1]))
            ax.set_xticklabels([f"IC{i+1}" for i in range(instance.mixing_matrix.shape[1])])
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=self.ica_fig_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, self.ica_fig_frame)
            toolbar.update()
            
        except Exception as e:
            print(f"Error creating mixing matrix plot: {str(e)}")
            raise

    def create_ica_scatter_plot(self, ic1, ic2):
        """Create scatter plot for ICA components"""
        try:
            # Clear previous plot
            for widget in self.ica_fig_frame.winfo_children():
                widget.destroy()
            
            # Create figure
            fig = Figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            
            # Get data
            x_data = self.ica_instance.ica_components[:, ic1]
            y_data = self.ica_instance.ica_components[:, ic2]
            
            # Create scatter plot
            ax.scatter(x_data, y_data, alpha=0.6, s=float(self.point_size.get()))
            
            # Customize plot
            ax.set_xlabel(f"IC{ic1 + 1}")
            ax.set_ylabel(f"IC{ic2 + 1}")
            ax.set_title("ICA Component Scatter Plot")
            ax.grid(True, alpha=0.3)
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=self.ica_fig_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, self.ica_fig_frame)
            toolbar.update()
            
        except Exception as e:
            print(f"Error creating ICA scatter plot: {str(e)}")
            raise

    def create_heatmap_plot(self, instance, analysis_type="pca"):
        """Create heatmap plot for PCA or ICA"""
        try:
            frame = self.pca_fig_frame if analysis_type == "pca" else self.ica_fig_frame
            
            # Clear previous plot
            for widget in frame.winfo_children():
                widget.destroy()
            
            # Get appropriate data and dimensions
            if analysis_type == "pca":
                data = instance.pca.components_
                dims = instance.original_dims
                title = "PCA Components Heatmap"
                ylabel = "Principal Components"
            else:
                data = instance.mixing_matrix.T  # Transpose for consistent visualization
                dims = self.pca_instance.original_dims  # Use PCA instance's dimensions
                title = "ICA Mixing Matrix Heatmap"
                ylabel = "Independent Components"
            
            # Create figure
            fig = Figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            
            # Create heatmap
            im = ax.imshow(data, cmap='RdBu_r', aspect='auto')
            
            # Add colorbar
            fig.colorbar(im)
            
            # Customize plot
            ax.set_title(title)
            ax.set_xlabel("Dimensions")
            ax.set_ylabel(ylabel)
            
            # Set y-axis ticks (component labels)
            n_components = data.shape[0]
            ax.set_yticks(range(n_components))
            ax.set_yticklabels([f"{analysis_type.upper()}{i+1}" for i in range(n_components)])
            
            # Set x-axis ticks (dimension labels)
            if len(dims) <= 20:
                ax.set_xticks(range(len(dims)))
                ax.set_xticklabels(dims, rotation=45, ha='right')
            else:
                # Show fewer ticks for readability
                step = len(dims) // 10
                ax.set_xticks(range(0, len(dims), step))
                ax.set_xticklabels([dims[i] for i in range(0, len(dims), step)], rotation=45, ha='right')
            
            fig.tight_layout()
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, frame)
            toolbar.update()
            
        except Exception as e:
            print(f"Error creating {analysis_type.upper()} heatmap plot: {str(e)}")
            raise

    def show_definition_generation_dialog(self):
        """Show dialog for generating missing value dimension definitions"""
        try:
            # Create dialog window
            dialog = tk.Toplevel(self.root)
            dialog.title("Generate Value Dimension Definitions")
            dialog.geometry("800x600")
            
            # Create main frame with padding
            main_frame = ttk.Frame(dialog, padding="10")
            main_frame.pack(fill="both", expand=True)
            
            # Analysis section
            analysis_frame = ttk.LabelFrame(main_frame, text="Analysis", padding="5")
            analysis_frame.pack(fill="x", pady=(0, 10))
            
            # Get missing definitions
            missing_dims, missing_count = self.pca_instance.analyze_missing_definitions()
            
            if missing_count == 0:
                ttk.Label(
                    analysis_frame, 
                    text="No missing definitions found in the dimensions file.",
                    padding="5"
                ).pack()
                
                ttk.Button(
                    analysis_frame,
                    text="Close",
                    command=dialog.destroy
                ).pack(pady=5)
                return
            
            # Show analysis results
            ttk.Label(
                analysis_frame,
                text=f"Found {missing_count} dimensions without definitions:",
                padding="5"
            ).pack()
            
            # Create scrollable frame for missing dimensions
            canvas = tk.Canvas(analysis_frame)
            scrollbar = ttk.Scrollbar(analysis_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # List missing dimensions
            for dim in missing_dims['value dimensions']:
                ttk.Label(
                    scrollable_frame,
                    text=f"• {dim}",
                    padding="2"
                ).pack(anchor="w")
            
            canvas.pack(side="left", fill="both", expand=True, padx=(5, 0))
            scrollbar.pack(side="right", fill="y")
            
            # Generation controls
            control_frame = ttk.Frame(main_frame)
            control_frame.pack(fill="x", pady=10)
            
            def generate_definitions():
                try:
                    # Disable generate button
                    generate_btn.config(state="disabled")
                    status_label.config(text="Generating definitions...")
                    dialog.update()
                    
                    # Generate definitions
                    results_df = self.pca_instance.generate_missing_definitions(self.llm_handler)
                    
                    # Show preview dialog
                    self.show_definition_preview_dialog(results_df, dialog)
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to generate definitions: {str(e)}")
                    generate_btn.config(state="normal")
                    status_label.config(text="Ready")
            
            generate_btn = ttk.Button(
                control_frame,
                text="Generate Definitions",
                command=generate_definitions
            )
            generate_btn.pack(side="left", padx=5)
            
            ttk.Button(
                control_frame,
                text="Cancel",
                command=dialog.destroy
            ).pack(side="left")
            
            # Status label
            status_label = ttk.Label(main_frame, text="Ready")
            status_label.pack(pady=5)
            
            # Make dialog modal
            dialog.transient(self.root)
            dialog.grab_set()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error showing definition dialog: {str(e)}")

    def show_definition_preview_dialog(self, results_df: pd.DataFrame, parent_dialog: tk.Toplevel):
        """Show dialog for previewing and editing generated definitions"""
        try:
            # Create preview dialog
            preview = tk.Toplevel(self.root)
            preview.title("Preview Generated Definitions")
            preview.geometry("1000x700")
            
            # Create main frame with padding
            main_frame = ttk.Frame(preview, padding="10")
            main_frame.pack(fill="both", expand=True)
            
            # Instructions
            ttk.Label(
                main_frame,
                text="Review and edit generated definitions. Click 'Accept' when ready.",
                padding="5"
            ).pack(fill="x")
            
            # Create frame for definitions
            defs_frame = ttk.Frame(main_frame)
            defs_frame.pack(fill="both", expand=True, pady=10)
            
            # Headers
            headers_frame = ttk.Frame(defs_frame)
            headers_frame.pack(fill="x", pady=(0, 5))
            ttk.Label(headers_frame, text="Value Dimension", width=30).pack(side="left", padx=5)
            ttk.Label(headers_frame, text="Original Definition", width=30).pack(side="left", padx=5)
            ttk.Label(headers_frame, text="Generated Definition", width=30).pack(side="left", padx=5)
            
            # Create scrollable frame for definitions
            canvas = tk.Canvas(defs_frame)
            scrollbar = ttk.Scrollbar(defs_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Store text widgets for later access
            definition_widgets = {}
            
            # Add each dimension and its definitions
            for idx, row in results_df.iterrows():
                frame = ttk.Frame(scrollable_frame)
                frame.pack(fill="x", pady=2)
                
                # Dimension name
                ttk.Label(
                    frame, 
                    text=row['value dimensions'],
                    width=30,
                    wraplength=200
                ).pack(side="left", padx=5)
                
                # Original definition (if any)
                original_def = row.get('dim_definitions', '')
                ttk.Label(
                    frame,
                    text=original_def if pd.notna(original_def) else "(No definition)",
                    width=30,
                    wraplength=200
                ).pack(side="left", padx=5)
                
                # Generated definition (editable)
                text_widget = tk.Text(frame, height=4, width=30, wrap="word")
                text_widget.insert("1.0", row['generated_definition'])
                text_widget.pack(side="left", padx=5)
                
                # Store reference to text widget
                definition_widgets[row['value dimensions']] = text_widget
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Control buttons
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill="x", pady=10)
            
            def regenerate_definitions():
                """Regenerate definitions for all dimensions"""
                try:
                    preview.destroy()
                    parent_dialog.destroy()
                    self.show_definition_generation_dialog()
                except Exception as e:
                    messagebox.showerror("Error", f"Error regenerating definitions: {str(e)}")
            
            def accept_definitions():
                """Save accepted definitions and create new file"""
                try:
                    # Get edited definitions
                    final_definitions = {}
                    for dim, text_widget in definition_widgets.items():
                        final_definitions[dim] = text_widget.get("1.0", "end-1c")
                    
                    # Create new DataFrame with updated definitions
                    new_df = self.pca_instance.value_dims.copy()
                    for dim, definition in final_definitions.items():
                        new_df.loc[new_df['value dimensions'] == dim, 'dim_definitions'] = definition
                    
                    # Ask user where to save the file
                    default_name = os.path.splitext(os.path.basename(self.dims_paths[0]))[0]
                    file_path = filedialog.asksaveasfilename(
                        defaultextension=".xlsx",
                        initialfile=f"{default_name}_llm_generated_defs.xlsx",
                        filetypes=[("Excel files", "*.xlsx")]
                    )
                    
                    if file_path:
                        # Save the file
                        new_df.to_excel(file_path, index=False)
                        messagebox.showinfo(
                            "Success",
                            f"Definitions saved to:\n{file_path}"
                        )
                        
                        # Update current value_dims with new definitions
                        self.pca_instance.value_dims = new_df
                        
                        # Recalculate cosine similarities with updated definitions
                        if hasattr(self.pca_instance, 'prompts'):
                            success = self.pca_instance.calculate_cosine_similarity(
                                self.pca_instance.prompts,
                                self.pca_instance.value_dims
                            )
                            if success:
                                # Update current_ratings with new cosine similarities
                                self.pca_instance.current_ratings = self.pca_instance.cosine_ratings.copy()
                                
                                # Perform PCA with updated ratings
                                if self.pca_instance.perform_pca():
                                    print("PCA recalculated with new cosine similarities")
                                else:
                                    print("Failed to recalculate PCA")
                        
                        # Close dialogs
                        preview.destroy()
                        parent_dialog.destroy()
                        
                except Exception as e:
                    messagebox.showerror("Error", f"Error saving definitions: {str(e)}")
            
            ttk.Button(
                button_frame,
                text="Regenerate All",
                command=regenerate_definitions
            ).pack(side="left", padx=5)
            
            ttk.Button(
                button_frame,
                text="Accept and Save",
                command=accept_definitions
            ).pack(side="left", padx=5)
            
            ttk.Button(
                button_frame,
                text="Cancel",
                command=lambda: [preview.destroy(), parent_dialog.destroy()]
            ).pack(side="left", padx=5)
            
            # Make dialog modal
            preview.transient(self.root)
            preview.grab_set()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error showing preview dialog: {str(e)}")

# If you want to run directly from this file
if __name__ == "__main__":
    print("Creating GUI...")
    gui = ValueDimensionPCAGui()
    print("Running GUI...")
    gui.run()
