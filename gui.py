from tkinter import ttk, messagebox, filedialog
import tkinter as tk
from typing import Dict
from llm_handler import LLMHandler

class ValueDimensionPCAGui:
    def __init__(self, master):
        # ... existing initialization code ...
        
        # Initialize LLM handler
        self.llm_handler = None
        self.generated_names = []
        
        # Add LLM interface
        self.setup_llm_interface()

    def setup_llm_interface(self):
        """Create the LLM naming interface"""
        # Create LLM frame
        self.llm_frame = ttk.LabelFrame(self.window, text="LLM Component Naming", padding="10")
        self.llm_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

        # API Configuration
        api_frame = ttk.LabelFrame(self.llm_frame, text="Configuration", padding="5")
        api_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        ttk.Label(api_frame, text="OpenAI API Key:").grid(row=0, column=0, padx=5, pady=2)
        self.api_key_var = tk.StringVar(value=self.config.get('openai_api_key'))
        self.api_key_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, show="*")
        self.api_key_entry.grid(row=0, column=1, padx=5, pady=2)
        
        # Number of PCs to name
        ttk.Label(api_frame, text="Number of PCs to name:").grid(row=1, column=0, padx=5, pady=2)
        self.n_pcs_var = tk.IntVar(value=3)
        self.n_pcs_spinbox = ttk.Spinbox(api_frame, from_=1, to=10, textvariable=self.n_pcs_var)
        self.n_pcs_spinbox.grid(row=1, column=1, padx=5, pady=2)

        # Custom prompt
        prompt_frame = ttk.LabelFrame(self.llm_frame, text="Prompt Template", padding="5")
        prompt_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        self.prompt_text = tk.Text(prompt_frame, height=6, width=50)
        self.prompt_text.insert("1.0", self.config.get('default_prompt_template'))
        self.prompt_text.grid(row=0, column=0, padx=5, pady=5)

        # Buttons
        button_frame = ttk.Frame(self.llm_frame)
        button_frame.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        self.generate_button = ttk.Button(
            button_frame, 
            text="Generate Names", 
            command=self.generate_pc_names
        )
        self.generate_button.grid(row=0, column=0, padx=5)
        
        self.regenerate_button = ttk.Button(
            button_frame, 
            text="Regenerate", 
            command=lambda: self.generate_pc_names(regenerate=True),
            state="disabled"
        )
        self.regenerate_button.grid(row=0, column=1, padx=5)
        
        self.export_button = ttk.Button(
            button_frame, 
            text="Export Results", 
            command=self.export_results,
            state="disabled"
        )
        self.export_button.grid(row=0, column=2, padx=5)

        # Results display
        self.results_text = tk.Text(self.llm_frame, height=10, width=50)
        self.results_text.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.llm_frame, 
            mode='determinate', 
            variable=self.progress_var
        )
        self.progress_bar.grid(row=4, column=0, padx=5, pady=5, sticky="ew")
        self.progress_bar.grid_remove()  # Hide initially

    def generate_pc_names(self, regenerate=False):
        """Handle the generation of PC names using LLM"""
        if not self.api_key_var.get():
            messagebox.showerror("Error", "Please enter your OpenAI API key")
            return
        
        # Save API key to config
        self.config.set('openai_api_key', self.api_key_var.get())
        
        # Initialize LLM handler with current config
        self.llm_handler = LLMHandler(self.config)
        
        # Get number of PCs to name
        n_pcs = self.n_pcs_var.get()
        
        # Clear previous results if not regenerating
        if not regenerate:
            self.results_text.delete('1.0', tk.END)
            self.generated_names = []
        
        # Update UI state
        self.generate_button.config(state="disabled")
        self.progress_var.set(0)
        self.progress_bar.grid()  # Show progress bar
        
        # Process each PC
        for i in range(n_pcs):
            def callback(result, pc_idx=i):
                """Callback function to handle results from LLM"""
                if 'error' in result:
                    self.handle_error(result['error'])
                    return
                
                self.generated_names.append(result)
                self.display_result(result, pc_idx)
                self.progress_var.set((pc_idx + 1) * 100 / n_pcs)
                
                if pc_idx == n_pcs - 1:  # Last PC
                    self.generation_complete()
            
            # Prepare data for current PC
            pc_data = self.llm_handler.prepare_pc_data(self.pca_instance, i)
            custom_prompt = self.prompt_text.get('1.0', tk.END).strip()
            
            try:
                # Generate name in separate thread
                self.llm_handler.generate_name(pc_data, custom_prompt, callback)
            except Exception as e:
                self.handle_error(f"Error starting generation: {str(e)}")
                return

    def display_result(self, result: Dict[str, str], pc_idx: int):
        """Display a single PC naming result"""
        self.results_text.insert(tk.END, f"\nPrincipal Component {pc_idx + 1}:\n")
        self.results_text.insert(tk.END, f"Name: {result['name']}\n")
        self.results_text.insert(tk.END, f"Explanation: {result['explanation']}\n")
        self.results_text.insert(tk.END, "-" * 50 + "\n")
        self.results_text.see(tk.END)  # Scroll to show latest result

    def generation_complete(self):
        """Handle completion of name generation"""
        self.progress_bar.grid_remove()  # Hide progress bar
        self.generate_button.config(state="normal")
        self.regenerate_button.config(state="normal")
        self.export_button.config(state="normal")
        messagebox.showinfo("Success", "PC names generated successfully!")

    def handle_error(self, error_message: str):
        """Handle errors during generation"""
        self.progress_bar.grid_remove()
        self.generate_button.config(state="normal")
        self.regenerate_button.config(state="disabled")
        self.export_button.config(state="disabled")
        messagebox.showerror("Error", error_message)

    def export_results(self):
        """Export generated names to a file"""
        if not self.generated_names:
            messagebox.showerror("Error", "No results to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                export_path = self.llm_handler.export_results(self.generated_names, file_path)
                messagebox.showinfo("Success", f"Results exported to {export_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")