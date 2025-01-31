import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
import numpy as np

def select_file():
    """Open a file dialog to select an Excel file."""
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select Excel File",
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    return file_path

def calculate_metrics(df, num_dimensions, max_rating):
    """Calculate average ratings and normalized raw sum of squares for each prompt."""
    # Calculate mean rating for each prompt
    means = df.mean(axis=1)
    
    # Calculate normalized sum of squares
    # Divide by (num_dimensions * max_rating^2) since we're dealing with squared values
    normalization_factor = float(num_dimensions) * (float(max_rating) ** 2)
    sum_squares = df.apply(lambda row: sum(row**2), axis=1) / normalization_factor
    
    # Create DataFrame with results
    results = pd.DataFrame({
        'Prompt': df.index,
        'Mean_Rating': means,
        'Sum_Squares': sum_squares
    })
    
    # Sort by mean rating (descending) and add rank
    results = results.sort_values('Mean_Rating', ascending=False)
    results['Rank'] = range(1, len(results) + 1)
    
    return results

def get_plot_info():
    """Get plot title and framework information from user."""
    # Get number of frameworks to compare
    while True:
        try:
            num_frameworks = int(input("How many frameworks would you like to compare? "))
            if num_frameworks > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get dataset name once
    dataset_name = input("Please enter the name of the dataset: ")
    
    # Get info for each framework
    frameworks = []
    for i in range(num_frameworks):
        print(f"\nFramework {i+1}:")
        framework_name = input("Please enter the name of the value dimensions framework: ")
        num_dimensions = input("Please enter the number of dimensions used in the framework: ")
        max_rating = input("Please enter the maximum possible rating value for this framework: ")
        frameworks.append({
            'name': framework_name,
            'dimensions': num_dimensions,
            'max_rating': max_rating,
            'color': ['blue', 'red', 'green', 'purple', 'orange', 'brown'][i]
        })
    
    return dataset_name, frameworks

def validate_prompts(dataframes):
    """Validate that prompts match across all dataframes."""
    if len(dataframes) <= 1:
        return True
    
    base_prompts = set(dataframes[0].index)
    for i, df in enumerate(dataframes[1:], 2):
        if set(df.index) != base_prompts:
            print(f"\nWarning: Prompts in framework {i} do not exactly match the prompts in framework 1.")
            return False
    return True

def create_plot(results_list, dataset_name, frameworks):
    """Create and display the scatter plot for multiple frameworks."""
    plt.figure(figsize=(12, 8))
    
    # Plot data for each framework
    for results, framework in zip(results_list, frameworks):
        # Plot regular points
        plt.scatter(results['Rank'], results['Sum_Squares'],
                   color=framework['color'], alpha=0.6,
                   label=f"{framework['dimensions']}-dimensional, {framework['name']} framework")
        
        # Plot top 10 as stars
        top_10 = results[results['Rank'] <= 10]
        plt.scatter(top_10['Rank'], top_10['Sum_Squares'],
                   color=framework['color'], marker='*', s=200,
                   zorder=2)
    
    plt.xlabel('Rank')
    plt.ylabel('Normalized Sum of Squares')
    
    # Set title based on number of frameworks
    if len(frameworks) <= 2:
        title = f"Rankings for {dataset_name} based on the "
        title += " vs ".join([f"{f['dimensions']}-dimensional, {f['name']} framework" 
                             for f in frameworks])
    else:
        title = f"Rankings for {dataset_name} across several value dimension frameworks"
    
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    return plt.gcf()

def save_plot():
    """Ask user if they want to save the plot and handle saving."""
    root = Tk()
    root.withdraw()
    save = input("Would you like to save the plot? (yes/no): ").lower()
    
    if save.startswith('y'):
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), 
                      ("All files", "*.*")]
        )
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

def main():
    try:
        # Get plot information from user
        dataset_name, frameworks = get_plot_info()
        
        # Get data for each framework
        results_list = []
        dataframes = []
        
        for i, framework in enumerate(frameworks, 1):
            print(f"\nPlease select the Excel file for {framework['name']} framework")
            file_path = select_file()
            if not file_path:
                print("No file selected. Exiting...")
                return
            
            # Read and process data
            df = pd.read_excel(file_path)
            prompts = df.iloc[:, 1]
            ratings = df.iloc[:, 2:-1]
            ratings.index = prompts
            
            # Convert numeric columns
            for col in ratings.columns:
                ratings[col] = pd.to_numeric(ratings[col], errors='coerce')
            
            dataframes.append(ratings)
            # Pass number of dimensions and max rating to calculate_metrics
            results = calculate_metrics(ratings, 
                                     framework['dimensions'], 
                                     framework['max_rating'])
            results_list.append(results)
        
        # Validate prompts across frameworks
        validate_prompts(dataframes)
        
        # Create and display plot
        fig = create_plot(results_list, dataset_name, frameworks)
        plt.show()
        
        # Handle plot saving
        save_plot()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 