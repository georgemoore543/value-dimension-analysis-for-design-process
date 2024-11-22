import tkinter as tk
from tkinter import ttk
import threading
from typing import Optional
from datetime import datetime

class UsageMonitor(ttk.Frame):
    """Widget to display API usage and costs"""
    
    def __init__(self, parent, llm_handler):
        super().__init__(parent)
        self.llm_handler = llm_handler
        self.update_interval = 1000  # Update every second
        
        self._create_widgets()
        self._start_monitor()

    def _create_widgets(self):
        """Create and layout the usage monitoring widgets"""
        # Current Session Stats
        session_frame = ttk.LabelFrame(self, text="Current Session")
        session_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Token Usage
        self.token_label = ttk.Label(session_frame, text="Total Tokens: 0")
        self.token_label.pack(anchor=tk.W, padx=5, pady=2)
        
        # Cost Display
        self.cost_label = ttk.Label(session_frame, text="Total Cost: $0.00")
        self.cost_label.pack(anchor=tk.W, padx=5, pady=2)
        
        # Calls Counter
        self.calls_label = ttk.Label(session_frame, text="API Calls: 0")
        self.calls_label.pack(anchor=tk.W, padx=5, pady=2)
        
        # Average Stats
        avg_frame = ttk.LabelFrame(self, text="Averages")
        avg_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.avg_tokens_label = ttk.Label(avg_frame, text="Tokens per Call: 0")
        self.avg_tokens_label.pack(anchor=tk.W, padx=5, pady=2)
        
        self.avg_cost_label = ttk.Label(avg_frame, text="Cost per Call: $0.00")
        self.avg_cost_label.pack(anchor=tk.W, padx=5, pady=2)
        
        # Last Call Info
        last_call_frame = ttk.LabelFrame(self, text="Last API Call")
        last_call_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.last_call_label = ttk.Label(last_call_frame, text="No calls yet")
        self.last_call_label.pack(anchor=tk.W, padx=5, pady=2)

    def _start_monitor(self):
        """Start the usage monitoring update loop"""
        def update_stats():
            if not self.winfo_exists():
                return
                
            summary = self.llm_handler.get_usage_summary()
            
            # Update session stats
            self.token_label.config(text=f"Total Tokens: {summary['total_tokens']:,}")
            self.cost_label.config(text=f"Total Cost: ${summary['total_cost']:.4f}")
            self.calls_label.config(text=f"API Calls: {summary['total_calls']}")
            
            # Update averages
            self.avg_tokens_label.config(
                text=f"Tokens per Call: {summary['average_tokens_per_call']:.1f}"
            )
            self.avg_cost_label.config(
                text=f"Cost per Call: ${summary['average_cost_per_call']:.4f}"
            )
            
            # Update last call info
            if self.llm_handler.usage_log:
                last_usage = self.llm_handler.usage_log[-1]
                last_call_time = last_usage.timestamp.strftime("%H:%M:%S")
                self.last_call_label.config(
                    text=f"Time: {last_call_time} | "
                    f"Tokens: {last_usage.total_tokens} | "
                    f"Cost: ${last_usage.estimated_cost:.4f}"
                )
            
            # Schedule next update
            self.after(self.update_interval, update_stats)
            
        # Start the update loop
        self.after(0, update_stats) 