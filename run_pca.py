import sys
import os
print("Python path:", sys.path)
print("Current working directory:", os.getcwd())

sys.dont_write_bytecode = True

print("About to import from pca_value_dim...")
from pca_value_dim import ValueDimensionPCAGui
print("Import completed")

if __name__ == "__main__":
    print("Creating GUI...")
    gui = ValueDimensionPCAGui()
    print("Running GUI...")
    gui.run() 