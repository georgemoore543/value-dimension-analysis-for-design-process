import numpy as np
from sklearn.decomposition import PCA

class PCAAnalyzer:
    def __init__(self, data, prompts):
        self.data = data
        self.prompts = prompts
        self.pca = None
        self.components = None
        
    def run_pca(self, n_components):
        self.pca = PCA(n_components=n_components)
        self.components = self.pca.fit_transform(self.data) 