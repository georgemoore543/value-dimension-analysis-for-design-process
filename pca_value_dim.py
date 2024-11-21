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
            print(f"Attempting to load data from:")
            print(f"Ratings paths: {ratings_paths}")
            print(f"Dimensions paths: {dims_paths}")
            
            # Load data
            print("Loading ratings files...")
            ratings_dfs = []
            for path in ratings_paths:
                print(f"Reading {path}")
                df = pd.read_excel(path)
                print(f"Original shape: {df.shape}")
                
                # Store metadata if needed
                if 'Prompt' in df.columns:
                    self.prompts = df['Prompt'].to_dict()
                
                ratings_dfs.append(df)
            
            print("Loading dimensions files...")
            dims_dfs = []
            for path in dims_paths:
                print(f"Reading {path}")
                df = pd.read_excel(path)
                print(f"Shape: {df.shape}")
                dims_dfs.append(df)
            
            # Validate and clean data
            print("Validating data...")
            is_valid, message = self.validate_data(ratings_dfs, dims_dfs)
            if not is_valid:
                return False, message
            
            # Process validated data
            print("Processing validated data...")
            self.ratings_data = pd.concat(ratings_dfs, axis=1)
            self.value_dims = pd.concat(dims_dfs, axis=0)
            self.original_dims = self.ratings_data.columns.tolist()
            
            # Perform PCA
            print("Performing PCA...")
            self.perform_pca()
            return True, "Data loaded successfully"
            
        except Exception as e:
            import traceback
            print(f"Error loading data: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())
            return False, f"Error loading data: {str(e)}"
    
    def perform_pca(self):
        """Perform PCA on the ratings data"""
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.ratings_data)
        
        # Perform PCA
        self.pca = PCA()
        self.pca_ratings = self.pca.fit_transform(scaled_data)
        
        # Store PCA results
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.components_ = self.pca.components_

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

    def create_loading_plot(self, pc1: int, pc2: int):
        """Create loading plot with prompt labels"""
        # Your existing plotting code, but use prompts for labels
        loadings = self.components_
        for i, (x, y) in enumerate(zip(loadings[pc1], loadings[pc2])):
            prompt = self.prompts.get(self.original_dims[i], "Unknown")
            # Add prompt as label
            plt.annotate(
                prompt[:30] + "..." if len(prompt) > 30 else prompt,
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )

# Then, the GUI class
class ValueDimensionPCAGui:
    # Add the PCA class as a class attribute
    PCA_Class = ValueDimensionPCA
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PCA Analysis")
        self.root.geometry("800x600")
        
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
    
    def proceed(self):
        try:
            print("Starting proceed method...")
            # Use the class directly from this module
            from pca_value_dim import ValueDimensionPCA  # Import from this module
            pca = ValueDimensionPCA()
            if pca.load_data(self.ratings_paths, self.dims_paths)[0]:  # Note: load_data returns a tuple
                messagebox.showinfo("Success", "Files loaded successfully!")
                self.show_visualization(pca)
            else:
                messagebox.showerror("Error", "Failed to load data")
        except Exception as e:
            print(f"Error in proceed: {str(e)}")
            messagebox.showerror("Error", f"Error processing files: {str(e)}")
    
    def show_visualization(self, pca):
        # Clear current window
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Create visualization controls
        self.create_visualization_controls(pca)
    
    def create_visualization_controls(self, pca):
        """Create the visualization control panel"""
        control_frame = ttk.LabelFrame(self.root, text="Visualization Controls", padding="10")
        control_frame.pack(side="left", fill="y", padx=10, pady=5)
        
        # PC Selection
        pc_frame = ttk.LabelFrame(control_frame, text="Principal Components", padding="5")
        pc_frame.pack(fill="x", pady=5)
        
        self.pc_x = tk.StringVar(value="1")
        self.pc_y = tk.StringVar(value="2")
        
        ttk.Label(pc_frame, text="X-axis PC:").pack()
        ttk.Spinbox(pc_frame, from_=1, to=len(pca.components_), textvariable=self.pc_x).pack()
        
        ttk.Label(pc_frame, text="Y-axis PC:").pack()
        ttk.Spinbox(pc_frame, from_=1, to=len(pca.components_), textvariable=self.pc_y).pack()
        
        # Visualization Type
        viz_frame = ttk.LabelFrame(control_frame, text="Plot Type", padding="5")
        viz_frame.pack(fill="x", pady=5)
        
        self.plot_type = tk.StringVar(value="scatter")
        ttk.Radiobutton(viz_frame, text="Scatter Plot", value="scatter", 
                       variable=self.plot_type).pack()
        ttk.Radiobutton(viz_frame, text="Loading Plot", value="loading", 
                       variable=self.plot_type).pack()
        ttk.Radiobutton(viz_frame, text="Biplot", value="biplot", 
                       variable=self.plot_type).pack()
        
        # Display Options
        options_frame = ttk.LabelFrame(control_frame, text="Display Options", padding="5")
        options_frame.pack(fill="x", pady=5)
        
        self.show_labels = tk.BooleanVar(value=True)
        self.show_vectors = tk.BooleanVar(value=True)
        self.show_ellipses = tk.BooleanVar(value=False)
        self.show_legend = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(options_frame, text="Show Labels", 
                        variable=self.show_labels).pack()
        ttk.Checkbutton(options_frame, text="Show Loading Vectors", 
                        variable=self.show_vectors).pack()
        ttk.Checkbutton(options_frame, text="Show Confidence Ellipses", 
                        variable=self.show_ellipses).pack()
        ttk.Checkbutton(options_frame, text="Show Legend", 
                        variable=self.show_legend).pack()
        
        # Customization Options
        custom_frame = ttk.LabelFrame(control_frame, text="Customization", padding="5")
        custom_frame.pack(fill="x", pady=5)
        
        self.point_size = tk.StringVar(value="8")
        self.vector_scale = tk.StringVar(value="1.0")
        
        ttk.Label(custom_frame, text="Point Size:").pack()
        ttk.Entry(custom_frame, textvariable=self.point_size).pack()
        
        ttk.Label(custom_frame, text="Vector Scale:").pack()
        ttk.Entry(custom_frame, textvariable=self.vector_scale).pack()
        
        # Update Button
        ttk.Button(control_frame, text="Update Plot", 
                   command=lambda: self.update_plot(pca)).pack(pady=10)
    
    def update_plot(self, pca):
        """Update the visualization based on current settings"""
        try:
            # Get current settings
            pc1 = int(self.pc_x.get()) - 1
            pc2 = int(self.pc_y.get()) - 1
            plot_type = self.plot_type.get()
            
            # Create figure based on plot type
            if plot_type == "scatter":
                self.create_scatter_plot(pca, pc1, pc2)
            elif plot_type == "loading":
                self.create_loading_plot(pca, pc1, pc2)
            else:  # biplot
                self.create_biplot(pca, pc1, pc2)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error updating plot: {str(e)}")
    
    def run(self):
        self.root.mainloop()

    def create_scatter_plot(self, pca, pc1, pc2):
        """Create scatter plot of PCA scores"""
        try:
            # Clear existing plot frame if it exists
            if hasattr(self, 'plot_frame'):
                self.plot_frame.destroy()
            
            # Create new plot frame
            self.plot_frame = ttk.Frame(self.root)
            self.plot_frame.pack(side="right", fill="both", expand=True)
            
            # Create figure
            fig = Figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            
            # Plot scores
            scatter = ax.scatter(
                pca.pca_ratings[:, pc1],
                pca.pca_ratings[:, pc2],
                s=float(self.point_size.get()),
                alpha=0.6
            )
            
            # Add labels if requested
            if self.show_labels.get():
                for i, txt in enumerate(pca.ratings_data.index):
                    ax.annotate(
                        txt,
                        (pca.pca_ratings[i, pc1], pca.pca_ratings[i, pc2]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8
                    )
            
            # Add confidence ellipses if requested
            if self.show_ellipses.get():
                self.add_confidence_ellipses(ax, pca, pc1, pc2)
            
            # Set labels and title
            var_exp = pca.explained_variance_ratio_
            ax.set_xlabel(f'PC{pc1+1} ({var_exp[pc1]:.1%} explained variance)')
            ax.set_ylabel(f'PC{pc2+1} ({var_exp[pc2]:.1%} explained variance)')
            ax.set_title('PCA Score Plot')
            
            # Add legend if requested
            if self.show_legend.get():
                ax.legend()
            
            # Add plot to GUI
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
            toolbar.update()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error creating scatter plot: {str(e)}")

    def create_loading_plot(self, pca, pc1, pc2):
        """Create loading plot"""
        try:
            # Clear existing plot frame
            if hasattr(self, 'plot_frame'):
                self.plot_frame.destroy()
            
            # Create new plot frame
            self.plot_frame = ttk.Frame(self.root)
            self.plot_frame.pack(side="right", fill="both", expand=True)
            
            # Create figure
            fig = Figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            
            # Plot loadings
            loadings = pca.components_
            scale = float(self.vector_scale.get())
            
            # Add vectors if requested
            if self.show_vectors.get():
                for i, (x, y) in enumerate(zip(loadings[pc1], loadings[pc2])):
                    ax.arrow(
                        0, 0, x * scale, y * scale,
                        head_width=0.05,
                        head_length=0.1,
                        fc='red',
                        ec='red',
                        alpha=0.5
                    )
            
            # Add labels if requested
            if self.show_labels.get():
                for i, txt in enumerate(pca.original_dims):
                    ax.annotate(
                        txt,
                        (loadings[pc1, i] * scale, loadings[pc2, i] * scale),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8
                    )
            
            # Set equal aspect ratio
            ax.set_aspect('equal')
            
            # Add unit circle
            circle = plt.Circle((0,0), 1, fill=False, linestyle='--', color='gray')
            ax.add_artist(circle)
            
            # Set labels and title
            var_exp = pca.explained_variance_ratio_
            ax.set_xlabel(f'PC{pc1+1} ({var_exp[pc1]:.1%} explained variance)')
            ax.set_ylabel(f'PC{pc2+1} ({var_exp[pc2]:.1%} explained variance)')
            ax.set_title('PCA Loading Plot')
            
            # Set axis limits
            limit = 1.2 * scale
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            
            # Add plot to GUI
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
            toolbar.update()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error creating loading plot: {str(e)}")

    def create_biplot(self, pca, pc1, pc2):
        """Create biplot (combined scores and loadings)"""
        try:
            # Clear existing plot frame
            if hasattr(self, 'plot_frame'):
                self.plot_frame.destroy()
            
            # Create new plot frame
            self.plot_frame = ttk.Frame(self.root)
            self.plot_frame.pack(side="right", fill="both", expand=True)
            
            # Create figure
            fig = Figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            
            # Plot scores
            scores = pca.pca_ratings[:, [pc1, pc2]]
            scatter = ax.scatter(
                scores[:, 0],
                scores[:, 1],
                s=float(self.point_size.get()),
                alpha=0.6,
                c='blue',
                label='Scores'
            )
            
            # Add score labels if requested
            if self.show_labels.get():
                for i, txt in enumerate(pca.ratings_data.index):
                    ax.annotate(
                        txt,
                        (scores[i, 0], scores[i, 1]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        color='blue'
                    )
            
            # Add loadings
            loadings = pca.components_[[pc1, pc2]].T
            scale = float(self.vector_scale.get())
            scaled_loadings = loadings * scale
            
            # Add loading vectors if requested
            if self.show_vectors.get():
                for i, (x, y) in enumerate(scaled_loadings):
                    ax.arrow(
                        0, 0, x, y,
                        head_width=0.05,
                        head_length=0.1,
                        fc='red',
                        ec='red',
                        alpha=0.5
                    )
            
            # Add loading labels if requested
            if self.show_labels.get():
                for i, txt in enumerate(pca.original_dims):
                    ax.annotate(
                        txt,
                        (scaled_loadings[i, 0], scaled_loadings[i, 1]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        color='red'
                    )
            
            # Add confidence ellipses if requested
            if self.show_ellipses.get():
                self.add_confidence_ellipses(ax, pca, pc1, pc2)
            
            # Set labels and title
            var_exp = pca.explained_variance_ratio_
            ax.set_xlabel(f'PC{pc1+1} ({var_exp[pc1]:.1%} explained variance)')
            ax.set_ylabel(f'PC{pc2+1} ({var_exp[pc2]:.1%} explained variance)')
            ax.set_title('PCA Biplot')
            
            # Add legend if requested
            if self.show_legend.get():
                ax.legend(['Scores', 'Loadings'])
            
            # Add plot to GUI
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
            toolbar.update()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error creating biplot: {str(e)}")

    def add_confidence_ellipses(self, ax, pca, pc1, pc2):
        """Add confidence ellipses to the plot"""
        from scipy.stats import chi2
        
        # Get the scores for the selected PCs
        scores = pca.pca_ratings[:, [pc1, pc2]]
        
        # Calculate the covariance matrix
        cov = np.cov(scores.T)
        
        # Calculate eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eig(cov)
        
        # Sort eigenvalues and eigenvectors
        sort_indices = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[sort_indices]
        eigenvecs = eigenvecs[:, sort_indices]
        
        # Create theta values for ellipse
        theta = np.linspace(0, 2*np.pi, 100)
        
        # Plot ellipses for different confidence levels
        chi2_vals = [chi2.ppf(p, df=2) for p in [0.68, 0.95, 0.99]]
        labels = ['68%', '95%', '99%']
        
        for chi2_val, label in zip(chi2_vals, labels):
            # Calculate ellipse points
            ellipse_x = (np.sqrt(chi2_val * eigenvals[0]) * 
                        (np.cos(theta) * eigenvecs[0, 0] + np.sin(theta) * eigenvecs[0, 1]))
            ellipse_y = (np.sqrt(chi2_val * eigenvals[1]) * 
                        (np.cos(theta) * eigenvecs[1, 0] + np.sin(theta) * eigenvecs[1, 1]))
            
            # Plot ellipse
            ax.plot(ellipse_x, ellipse_y, '--', label=f'{label} Confidence')

# If you want to run directly from this file
if __name__ == "__main__":
    gui = ValueDimensionPCAGui()
    gui.run()
