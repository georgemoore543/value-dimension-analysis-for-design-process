import sys
import os
print("Python path:", sys.path)
print("Current working directory:", os.getcwd())

sys.dont_write_bytecode = True

print("About to import from pca_value_dim...")
from pca_value_dim import ValueDimensionPCAGui
print("Import completed")

from run_pca_naming import generate_pca_names
import tkinter as tk
from tkinter import messagebox

class PCAGui:
    def __init__(self):
        # ... existing initialization ...
        
        # Add button for PCA name generation
        self.name_gen_button = tk.Button(
            self.button_frame,
            text="Generate Component Names",
            command=self.generate_component_names,
            state=tk.DISABLED  # Enable only after PCA is run
        )
        self.name_gen_button.pack(side=tk.LEFT, padx=5)

    def run_pca(self):
        """Existing PCA analysis method"""
        try:
            # ... existing PCA code ...
            
            # Enable the name generation button after successful PCA
            self.name_gen_button.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"PCA Analysis failed: {str(e)}")

    def generate_component_names(self):
        """Handle PCA name generation"""
        try:
            # Show processing indicator
            self.status_label.config(text="Generating component names...")
            self.root.update()

            # Get results from the name generation
            results_df = generate_pca_names(
                pca_results=self.pca_results,  # Your PCA results
                prompts_df=self.pca_instance.prompts_df,    # Your prompts DataFrame
                n_components=self.n_components # Number of components
            )

            if results_df is not None:
                # Display results in the GUI
                self.display_component_names(results_df)
                messagebox.showinfo(
                    "Success", 
                    f"Generated names for {len(results_df)} components"
                )
            else:
                messagebox.showerror(
                    "Error", 
                    "Failed to generate component names"
                )

        except Exception as e:
            messagebox.showerror(
                "Error", 
                f"Name generation failed: {str(e)}"
            )
        finally:
            self.status_label.config(text="Ready")

    def display_component_names(self, results_df):
        """Display the generated names in the GUI"""
        # Create or update results window
        if not hasattr(self, 'results_window'):
            self.results_window = tk.Toplevel(self.root)
            self.results_window.title("PCA Component Names")
        
        # Clear existing content
        for widget in self.results_window.winfo_children():
            widget.destroy()
        
        # Create scrollable frame
        canvas = tk.Canvas(self.results_window)
        scrollbar = tk.Scrollbar(self.results_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Display results
        for _, row in results_df.iterrows():
            frame = tk.Frame(scrollable_frame)
            frame.pack(fill="x", padx=5, pady=5)
            
            tk.Label(
                frame, 
                text=f"PC {row['pc_num']}: {row['name']}", 
                font=("Arial", 10, "bold")
            ).pack(anchor="w")
            
            tk.Label(
                frame, 
                text=row['explanation'], 
                wraplength=400
            ).pack(anchor="w")
            
            tk.Frame(frame, height=1, bg="gray").pack(fill="x", pady=5)

        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

if __name__ == "__main__":
    print("Creating GUI...")
    gui = ValueDimensionPCAGui()
    print("Running GUI...")
    gui.run() 