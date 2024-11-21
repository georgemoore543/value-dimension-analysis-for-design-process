from pca_value_dim import ValueDimensionPCAGui, ValueDimensionPCA, PCADashboard

def main():
    try:
        # Start with the GUI file selection
        gui = ValueDimensionPCAGui()
        gui.run()  # This will block until file selection is complete
        
        # Get the selected file paths from the GUI
        ratings_paths = gui.ratings_paths
        dims_paths = gui.dims_paths
        
        if not ratings_paths or not dims_paths:
            print("No files were selected. Exiting...")
            return
            
        # Initialize PCA analysis with selected files
        pca = ValueDimensionPCA()
        pca.load_data(ratings_paths, dims_paths)
        
        # Perform the necessary calculations
        print("\nPerforming calculations...")
        pca.normalize_data()
        pca.calculate_cosine_similarity()
        pca.perform_pca()
        
        print("\nStarting dashboard...")
        # Create and run the dashboard
        dashboard = PCADashboard(pca)
        dashboard.run_server()
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main() 