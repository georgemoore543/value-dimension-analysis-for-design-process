import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import os
from typing import List, Tuple
import plotly.graph_objects as go
import numpy as np
from dash import html, dcc, Input, Output, State
from sklearn.impute import SimpleImputer
import io

class ValueDimensionPCAGui:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PCA Analysis File Selection")
        self.root.geometry("800x600")
        
        self.ratings_paths = []
        self.dims_paths = []
        self.ratings_data = []  # Store DataFrames
        self.dims_data = []     # Store DataFrames
        
        self.create_widgets()
        
    def create_widgets(self):
        # File count frame
        count_frame = ttk.LabelFrame(self.root, text="Number of Files", padding="10")
        count_frame.pack(fill="x", padx=10, pady=5)
        
        # Ratings count
        ttk.Label(count_frame, text="Number of ratings spreadsheets:").grid(row=0, column=0, padx=5, pady=5)
        self.ratings_count = ttk.Spinbox(count_frame, from_=1, to=10, width=5)
        self.ratings_count.grid(row=0, column=1, padx=5, pady=5)
        
        # Dimensions count
        ttk.Label(count_frame, text="Number of dimension spreadsheets:").grid(row=1, column=0, padx=5, pady=5)
        self.dims_count = ttk.Spinbox(count_frame, from_=1, to=10, width=5)
        self.dims_count.grid(row=1, column=1, padx=5, pady=5)
        
        # Button to proceed to file selection
        ttk.Button(count_frame, text="Set File Count", command=self.setup_file_selection).grid(row=2, column=0, columnspan=2, pady=10)
        
        # File selection frames
        self.files_frame = ttk.Frame(self.root)
        self.files_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Summary text
        self.summary_text = tk.Text(self.root, height=10, width=80)
        self.summary_text.pack(padx=10, pady=5)
        
        # Proceed button (initially disabled)
        self.proceed_button = ttk.Button(self.root, text="Proceed with Analysis", command=self.proceed, state="disabled")
        self.proceed_button.pack(pady=10)
        
    def setup_file_selection(self):
        # Clear previous file selection widgets
        for widget in self.files_frame.winfo_children():
            widget.destroy()
        
        self.ratings_paths = []
        self.dims_paths = []
        
        # Create ratings file selection
        ratings_frame = ttk.LabelFrame(self.files_frame, text="Ratings Spreadsheets", padding="10")
        ratings_frame.pack(fill="x", pady=5)
        
        for i in range(int(self.ratings_count.get())):
            self.create_file_selector(ratings_frame, f"Ratings file {i+1}", "ratings", i)
        
        # Create dimensions file selection
        dims_frame = ttk.LabelFrame(self.files_frame, text="Dimensions Spreadsheets", padding="10")
        dims_frame.pack(fill="x", pady=5)
        
        for i in range(int(self.dims_count.get())):
            self.create_file_selector(dims_frame, f"Dimensions file {i+1}", "dims", i)
    
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
            df = pd.read_excel(path)
            
            # Validate column names for ratings files against dimensions
            if file_type == "ratings" and self.dims_data:
                dims_df = self.dims_data[min(index, len(self.dims_data)-1)]  # Get corresponding dims file
                value_dims = dims_df['value_dimension'].tolist()  # Assuming this is the column name
                
                # Check if all value dimensions are present in ratings columns
                missing_dims = [dim for dim in value_dims if dim not in df.columns]
                extra_cols = [col for col in df.columns if col not in value_dims and col != 'prompt']
                
                if missing_dims or extra_cols:
                    message = "Column name mismatch detected:\n\n"
                    if missing_dims:
                        message += f"Missing dimensions: {', '.join(missing_dims)}\n\n"
                    if extra_cols:
                        message += f"Extra columns: {', '.join(extra_cols)}\n\n"
                    message += "Would you like to proceed anyway?"
                    
                    if not messagebox.askyesno("Column Validation", message, icon='warning'):
                        raise ValueError("Column name validation failed")
            
            # Check for missing values
            missing_rows = df.isnull().any(axis=1).sum()
            missing_cols = df.isnull().any(axis=0).sum()
            
            if missing_rows > 0 or missing_cols > 0:
                message = f"Missing values detected:\n\n" \
                         f"Rows with missing values: {missing_rows}\n" \
                         f"Columns with missing values: {missing_cols}\n\n" \
                         f"How would you like to proceed?"
                
                choice = messagebox.askquestion("Missing Values",
                    message,
                    type='yesnocancel',
                    icon='warning',
                    detail="Yes = Remove rows with missing values\n" \
                          "No = Remove columns with missing values\n" \
                          "Cancel = Impute missing values with mean")
                
                if choice == 'yes':
                    df = df.dropna(axis=0)
                    messagebox.showinfo("Info", f"Removed {missing_rows} rows with missing values.")
                elif choice == 'no':
                    df = df.dropna(axis=1)
                    messagebox.showinfo("Info", f"Removed {missing_cols} columns with missing values.")
                elif choice == 'cancel':
                    imputer = SimpleImputer(strategy='mean')
                    df = pd.DataFrame(
                        imputer.fit_transform(df),
                        columns=df.columns,
                        index=df.index
                    )
                    messagebox.showinfo("Info", "Missing values have been imputed with mean values.")
            
            # Store path and dataframe in appropriate list
            if file_type == "ratings":
                if len(self.ratings_paths) <= index:
                    self.ratings_paths.append(path)
                    self.ratings_data.append(df)
                else:
                    self.ratings_paths[index] = path
                    self.ratings_data[index] = df
            else:
                if len(self.dims_paths) <= index:
                    self.dims_paths.append(path)
                    self.dims_data.append(df)
                else:
                    self.dims_paths[index] = path
                    self.dims_data[index] = df
            
            self.update_summary()
            self.show_data_preview(df, file_type, index)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file: {str(e)}")
    
    def show_data_preview(self, df, file_type, index):
        """
        Enhanced data preview with custom row count and column filtering
        """
        preview_window = tk.Toplevel(self.root)
        preview_window.title(f"Data Preview - {file_type.capitalize()} File {index + 1}")
        preview_window.geometry("1000x600")
        
        # Control frame
        control_frame = ttk.Frame(preview_window, padding="5")
        control_frame.pack(fill="x", padx=10, pady=5)
        
        # Row count selector
        ttk.Label(control_frame, text="Preview Rows:").pack(side="left", padx=5)
        row_count = ttk.Spinbox(control_frame, from_=1, to=100, width=5)
        row_count.set(5)  # Default value
        row_count.pack(side="left", padx=5)
        
        # Column filter
        ttk.Label(control_frame, text="Filter Columns:").pack(side="left", padx=5)
        column_var = tk.StringVar()
        column_filter = ttk.Combobox(control_frame, textvariable=column_var, values=['All'] + list(df.columns))
        column_filter.set('All')
        column_filter.pack(side="left", padx=5)
        
        # Create preview frame
        preview_frame = ttk.Frame(preview_window, padding="10")
        preview_frame.pack(fill="both", expand=True)
        
        # Add preview text
        preview_text = tk.Text(preview_frame, wrap=tk.NONE)
        preview_text.pack(fill="both", expand=True)
        
        # Add scrollbars
        y_scrollbar = ttk.Scrollbar(preview_frame, orient="vertical", command=preview_text.yview)
        y_scrollbar.pack(side="right", fill="y")
        x_scrollbar = ttk.Scrollbar(preview_frame, orient="horizontal", command=preview_text.xview)
        x_scrollbar.pack(side="bottom", fill="x")
        
        preview_text.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
        
        def update_preview():
            preview_text.configure(state='normal')
            preview_text.delete(1.0, tk.END)
            
            # Filter dataframe based on selected column
            filtered_df = df if column_filter.get() == 'All' else df[[column_filter.get()]]
            
            # Show dataframe info
            preview_text.insert(tk.END, "=== DataFrame Info ===\n\n")
            buffer = io.StringIO()
            filtered_df.info(buf=buffer)
            preview_text.insert(tk.END, buffer.getvalue() + "\n\n")
            
            # Show selected number of rows
            n_rows = int(row_count.get())
            preview_text.insert(tk.END, f"=== First {n_rows} Rows ===\n\n")
            preview_text.insert(tk.END, filtered_df.head(n_rows).to_string() + "\n\n")
            
            # Show basic statistics
            preview_text.insert(tk.END, "=== Basic Statistics ===\n\n")
            preview_text.insert(tk.END, filtered_df.describe().to_string())
            
            preview_text.configure(state='disabled')
        
        # Update button
        ttk.Button(control_frame, text="Update Preview", command=update_preview).pack(side="left", padx=20)
        
        # Close button
        ttk.Button(preview_window, text="Close", command=preview_window.destroy).pack(pady=10)
        
        # Initial preview
        update_preview()
    
    def update_summary(self):
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, "=== Selected Files Summary ===\n\n")
        
        # Show ratings files
        self.summary_text.insert(tk.END, "Ratings Spreadsheets:\n")
        for i, path in enumerate(self.ratings_paths, 1):
            if os.path.exists(path):
                size = os.path.getsize(path) / (1024 * 1024)  # Convert to MB
                df = pd.read_excel(path)
                self.summary_text.insert(tk.END, 
                    f"{i}. {os.path.basename(path)} ({size:.2f} MB)\n"
                    f"   Columns: {', '.join(df.columns)}\n"
                    f"   Rows: {len(df)}\n\n"
                )
        
        # Show dimensions files
        self.summary_text.insert(tk.END, "Dimensions Spreadsheets:\n")
        for i, path in enumerate(self.dims_paths, 1):
            if os.path.exists(path):
                size = os.path.getsize(path) / (1024 * 1024)  # Convert to MB
                df = pd.read_excel(path)
                self.summary_text.insert(tk.END, 
                    f"{i}. {os.path.basename(path)} ({size:.2f} MB)\n"
                    f"   Columns: {', '.join(df.columns)}\n"
                    f"   Rows: {len(df)}\n\n"
                )
        
        # Enable proceed button if all files are selected
        expected_total = int(self.ratings_count.get()) + int(self.dims_count.get())
        actual_total = len(self.ratings_paths) + len(self.dims_paths)
        self.proceed_button['state'] = 'normal' if actual_total == expected_total else 'disabled'
    
    def proceed(self):
        # Here you would typically initialize your PCA analysis
        pca = ValueDimensionPCA()
        try:
            pca.load_data(self.ratings_paths, self.dims_paths)
            messagebox.showinfo("Success", "Data loaded successfully!")
            self.root.destroy()  # Close the GUI
        except Exception as e:
            messagebox.showerror("Error", f"Error loading data: {str(e)}")
    
    def run(self):
        self.root.mainloop()

    def create_3d_scatter(self, data, labels, title, prompts=None, color=None):
        """
        Enhanced 3D scatter plot with hover info and animation capabilities
        """
        fig = go.Figure(data=[go.Scatter3d(
            x=data[:, 0],
            y=data[:, 1],
            z=data[:, 2],
            mode='markers',
            text=prompts if prompts is not None else labels,
            hovertemplate="<b>Prompt:</b> %{text}<br>" +
                         "<b>PC1:</b> %{x:.2f}<br>" +
                         "<b>PC2:</b> %{y:.2f}<br>" +
                         "<b>PC3:</b> %{z:.2f}<br>" +
                         "<extra></extra>",
            marker=dict(
                size=8,
                color=color if color else data[:, 0],
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(
                    title='PC1',
                    rangeslider=dict(visible=True)
                ),
                yaxis=dict(
                    title='PC2',
                    rangeslider=dict(visible=True)
                ),
                zaxis=dict(
                    title='PC3',
                    rangeslider=dict(visible=True)
                ),
            ),
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(
                            label='Play',
                            method='animate',
                            args=[None, {'frame': {'duration': 500, 'redraw': True},
                                       'fromcurrent': True,
                                       'transition': {'duration': 300,
                                                    'easing': 'quadratic-in-out'}}]
                        ),
                        dict(
                            label='Pause',
                            method='animate',
                            args=[[None], {'frame': {'duration': 0, 'redraw': False},
                                         'mode': 'immediate',
                                         'transition': {'duration': 0}}]
                        )
                    ]
                )
            ]
        )
        
        # Add animation frames
        frames = [
            go.Frame(
                data=[go.Scatter3d(
                    x=data[:, 0] * np.cos(theta) - data[:, 1] * np.sin(theta),
                    y=data[:, 0] * np.sin(theta) + data[:, 1] * np.cos(theta),
                    z=data[:, 2]
                )],
                name=f'frame{i}'
            )
            for i, theta in enumerate(np.linspace(0, 2*np.pi, 30))
        ]
        fig.frames = frames
        
        return fig

    def save_visualization(self, fig, filename, spreadsheet_names):
        """
        Save visualization with formatted filename
        """
        # Create abbreviated spreadsheet names
        abbrev_names = '_'.join([name.split('.')[0][:3] for name in spreadsheet_names])
        
        # Determine visualization type
        if isinstance(fig, go.Figure):
            if any(trace.type == 'scatter3d' for trace in fig.data):
                viz_type = '3d'
            elif any(trace.type == 'heatmap' for trace in fig.data):
                viz_type = 'hmap'
            else:
                viz_type = 'scat'
        
        # Format filename
        formatted_filename = f"{abbrev_names}_{viz_type}_{filename}.png"
        
        # Save figure
        fig.write_image(formatted_filename)
        return formatted_filename

class PCADashboard:
    def setup_layout(self):
        """
        Enhanced dashboard layout with new features
        """
        # Previous layout code remains...
        
        # Add axis range controls
        axis_controls = html.Div([
            html.Div([
                html.Label("PC1 Range:"),
                dcc.RangeSlider(
                    id='pc1-range',
                    min=-10,
                    max=10,
                    step=0.1,
                    value=[-5, 5],
                    marks={i: str(i) for i in range(-10, 11, 2)}
                )
            ]),
            html.Div([
                html.Label("PC2 Range:"),
                dcc.RangeSlider(
                    id='pc2-range',
                    min=-10,
                    max=10,
                    step=0.1,
                    value=[-5, 5],
                    marks={i: str(i) for i in range(-10, 11, 2)}
                )
            ]),
            html.Div([
                html.Label("PC3 Range:"),
                dcc.RangeSlider(
                    id='pc3-range',
                    min=-10,
                    max=10,
                    step=0.1,
                    value=[-5, 5],
                    marks={i: str(i) for i in range(-10, 11, 2)}
                )
            ])
        ])
        
        self.app.layout.children.insert(-1, axis_controls)

    def setup_callbacks(self):
        """
        Enhanced callbacks with new features
        """
        @self.app.callback(
            [Output('quadrant1', 'figure'),
             Output('quadrant2', 'figure'),
             Output('quadrant3', 'figure'),
             Output('quadrant4', 'figure'),
             Output('scatter-matrix-container', 'children'),
             Output('scatter-matrix-container', 'style')],
            [Input('q1-viz-type', 'value'),
             Input('dataset-filter', 'value'),
             Input('lda-topics-slider', 'value'),
             Input('viz-mode', 'value'),
             Input('color-input', 'value'),
             Input('layout-style', 'value'),
             Input('show-scatter-matrix', 'value'),
             Input('pc1-range', 'value'),
             Input('pc2-range', 'value'),
             Input('pc3-range', 'value')]
        )
        def update_visualizations(viz_type, selected_datasets, n_topics,
                                viz_mode, color, layout_style, show_matrix,
                                pc1_range, pc2_range, pc3_range):
            # Previous visualization code...
            
            # Update axis ranges for 3D plots
            if viz_type == '3d':
                for fig in figures:
                    fig.update_layout(
                        scene=dict(
                            xaxis_range=pc1_range,
                            yaxis_range=pc2_range,
                            zaxis_range=pc3_range
                        )
                    )
            
            return figures + [scatter_matrix, matrix_style]

        @self.app.callback(
            Output('download-dataframe', 'data'),
            Input('save-viz-button', 'n_clicks'),
            State('q1-viz-type', 'value'),  # Add states for current visualization state
            prevent_initial_call=True
        )
        def save_visualization(n_clicks, viz_type):
            if n_clicks:
                # Get current figures
                figures = [
                    self.app.get_asset_url(f'quadrant{i+1}')
                    for i in range(4)
                ]
                
                # Save each figure
                filenames = []
                for i, fig in enumerate(figures):
                    filename = self.pca_analysis.save_visualization(
                        fig,
                        f'quadrant{i+1}',
                        self.pca_analysis.spreadsheet_names  # Add this attribute to store loaded spreadsheet names
                    )
                    filenames.append(filename)
                
                return filenames
