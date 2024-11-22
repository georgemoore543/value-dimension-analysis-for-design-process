import tkinter as tk
from tkinter import ttk
from .usage_monitor import UsageMonitor

class MainWindow(tk.Tk):
    def __init__(self, llm_handler):
        super().__init__()
        
        self.title("PCA Name Generator")
        
        # Create main container
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left panel for main functionality
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create right panel for usage monitoring
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add usage monitor to right panel
        self.usage_monitor = UsageMonitor(right_panel, llm_handler)
        self.usage_monitor.pack(fill=tk.BOTH, expand=True)
        
        # Add other widgets to left panel
        # ... your existing GUI code ... 