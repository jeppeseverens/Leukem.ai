class FeatureSelection(BaseEstimator, TransformerMixin):
    """
    Transformer for selecting top variable genes across multiple studies using MAD.

    This transformer selects the top 'n_genes' per study based on the median absolute deviation (MAD)
    and then uses the intersection of these genes across predefined studies.
    """
    def __init__(self):
        self.study_per_patient = None
        self.n_genes = None

    def _compute_top_genes(self, X_arr):
        # Pre-allocate arrays for better memory efficiency
        n_samples, n_genes = X_arr.shape
        medians = np.empty(n_genes, dtype=X_arr.dtype)
        mad = np.empty(n_genes, dtype=X_arr.dtype)
        
        # Compute medians in one pass
        np.median(X_arr, axis=0, out=medians)
        
        # Compute MAD in one pass using in-place operations
        # Subtract median and take absolute value in one step
        np.abs(X_arr - medians, out=X_arr)  # In-place operation
        np.median(X_arr, axis=0, out=mad)  # In-place operation
        
        # Use argpartition with kth parameter for better performance
        # This is O(n) instead of O(n log n) for full sorting
        kth = n_genes - self.n_genes
        top_indices = np.argpartition(mad, kth)[kth:]
        
        # Convert to set for faster intersection operations later
        return set(top_indices)

    def fit(self, X, y=None, study_per_patient=None, n_genes = 2000):
        self.n_genes = n_genes
        self.study_per_patient = study_per_patient

        if self.study_per_patient is None:
            raise ValueError("study_per_patient must be provided.")
        
        selected_studies = [
            "BEATAML1.0-COHORT",
            "AAML0531",
            "AAML1031",
            "TCGA-LAML",
            "LEUCEGENE"
        ]
        top_genes_by_study = {}
        
        # Pre-allocate memory for study masks
        study_masks = {}
        for study in selected_studies:
            study_masks[study] = self.study_per_patient == study
        
        for study in selected_studies:
            mask = study_masks[study]
            if mask.sum() == 0:
                continue
            X_study_arr = X[mask, :]
            top_genes_by_study[study] = self._compute_top_genes(X_study_arr)

        # Compute the intersection of top genes and preserve the original order
        intersect_genes = set.intersection(*top_genes_by_study.values())
        self.mvgs_ = [i for i in intersect_genes]
        
        return self

    def transform(self, X, y=None):
        """
        Reduce the dataset to only include the selected genes.
        """
        return X[:, self.mvgs_] 

class FeatureSelection2(BaseEstimator, TransformerMixin):
    """
    Transformer for selecting top variable genes across multiple studies using MAD.

    This transformer selects the top 'n_genes' per study based on the median absolute deviation (MAD)
    and then uses the intersection of these genes across predefined studies.
    """
    def __init__(self):
        self.study_per_patient = None
        self.n_genes = None

    def _compute_top_genes(self, X_arr):
        # Pre-allocate arrays for better memory efficiency
        n_samples, n_genes = X_arr.shape
        medians = np.empty(n_genes, dtype=X_arr.dtype)
        mad = np.empty(n_genes, dtype=X_arr.dtype)
        
        # Compute medians in one pass
        np.median(X_arr, axis=0, out=medians)
        
        # Compute MAD in one pass using in-place operations
        # Subtract median and take absolute value in one step
        np.abs(X_arr - medians, out=X_arr)  # In-place operation
        np.median(X_arr, axis=0, out=mad)  # In-place operation
        
        # Use argpartition with kth parameter for better performance
        # This is O(n) instead of O(n log n) for full sorting
        kth = n_genes - self.n_genes
        top_indices = np.argpartition(mad, kth)[kth:]
        
        # Convert to set for faster intersection operations later
        return set(top_indices)

    def _process_study(self, study, X, mask):
        if mask.sum() == 0:
            return study, None
        X_study_arr = X[mask, :]
        return study, self._compute_top_genes(X_study_arr)

    def fit(self, X, y=None, study_per_patient=None, n_genes = 2000):
        self.n_genes = n_genes
        self.study_per_patient = study_per_patient

        if self.study_per_patient is None:
            raise ValueError("study_per_patient must be provided.")
        
        selected_studies = [
            "BEATAML1.0-COHORT",
            "AAML0531",
            "AAML1031",
            "TCGA-LAML",
            "LEUCEGENE"
        ]
        top_genes_by_study = {}
        
        # Pre-allocate memory for study masks
        study_masks = {}
        for study in selected_studies:
            study_masks[study] = self.study_per_patient == study
        
        # Use ThreadPoolExecutor for parallel processing
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        # Create a thread-local storage for numpy operations
        thread_local = threading.local()
        
        def init_thread():
            # Each thread gets its own numpy random state
            thread_local.np_random = np.random.RandomState()
        
        # Process studies in parallel
        with ThreadPoolExecutor(max_workers=len(selected_studies), initializer=init_thread) as executor:
            # Submit all tasks
            future_to_study = {
                executor.submit(self._process_study, study, X, study_masks[study]): study 
                for study in selected_studies
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_study):
                study = future_to_study[future]
                try:
                    study, result = future.result()
                    if result is not None:
                        top_genes_by_study[study] = result
                except Exception as exc:
                    print(f'{study} generated an exception: {exc}')

        # Compute the intersection of top genes and preserve the original order
        intersect_genes = set.intersection(*top_genes_by_study.values())
        self.mvgs_ = [i for i in intersect_genes]
        
        return self

    def transform(self, X, y=None):
        """
        Reduce the dataset to only include the selected genes.
        """
        return X[:, self.mvgs_] 