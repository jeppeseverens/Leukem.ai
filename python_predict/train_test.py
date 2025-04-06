import pandas as pd
import os
from datetime import datetime
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_predict
from sklearn.metrics import cohen_kappa_score, accuracy_score, matthews_corrcoef
import numpy as np
import pandas as pd


import os
import pandas as pd
import numpy as np
from datetime import datetime

def load_data(directory):
    """
    Loads data from CSV files in a given directory based on file types and dates,
    returning NumPy arrays.

    This function searches the provided directory for CSV files starting with specific
    file types (e.g. 'meta', 'GDC_counts', and 'RGAs'). It extracts the date from the
    filename and selects the newest file for each file type. The data is then loaded
    using pandas and converted to NumPy arrays.

    Parameters
    ----------
    directory : str
        Path to the directory containing the CSV files.

    Returns
    -------
    studies : np.ndarray
        NumPy array loaded from the 'Studies' column of the newest 'meta' file.
    X : np.ndarray
        NumPy array of gene count data loaded from the newest 'GDC_counts' file.
        Rows correspond to samples/observations, columns to genes/features.
    y : np.ndarray
        NumPy array of the target variable ('ICC_Subtype') loaded from the newest 'RGAs' file.
    """
    # List all files in the directory.
    files = os.listdir(directory)

    def extract_date(filename, file_type):
        """
        Extracts the date from the filename if it starts with the specified file_type.

        If the filename does not start with the provided file_type or the date format
        is incorrect, returns the minimum datetime value so that it is not selected
        as the newest file.

        Parameters
        ----------
        filename : str
            The name of the file.
        file_type : str
            The prefix type to match in the filename.

        Returns
        -------
        datetime
            The extracted date from the filename or datetime.min if not matching or error.
        """
        # Ensure the filename starts with the given file type.
        if not filename.lower().startswith(file_type.lower()):
            return datetime.min
        try:
            # Extract the date part from the filename (assumes format: <type>_<...>_<date>.csv)
            date_str = filename.split('_')[-1].replace('.csv', '')
            return datetime.strptime(date_str, '%d%b%Y')
        except (ValueError, IndexError):
             # Handle cases where splitting or date parsing fails
             return datetime.min

    # Filter for CSV files first to avoid errors with non-CSV files
    csv_files = [f for f in files if f.lower().endswith('.csv')]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {directory}")

    # Find the newest file for each type based on the extracted date.
    meta_file = max(csv_files, key=lambda x: extract_date(x, 'meta'))
    counts_file = max(csv_files, key=lambda x: extract_date(x, 'GDC_counts'))
    rgas_file = max(csv_files, key=lambda x: extract_date(x, 'RGAs'))

    # Construct full paths
    meta_path = os.path.join(directory, meta_file)
    counts_path = os.path.join(directory, counts_file)
    rgas_path = os.path.join(directory, rgas_file)

    # Load CSV data into pandas DataFrames/Series.
    X_df = pd.read_csv(counts_path, index_col=0, engine='c')
    studies_series = pd.read_csv(meta_path)["Studies"]
    y_series = pd.read_csv(rgas_path, index_col=0)["ICC_Subtype"]

    # --- Convert to NumPy arrays ---
    # .values returns the underlying numpy array representation
    studies = studies_series.values
    X = X_df.transpose().values
    y = y_series.values

    print(f"  Studies: {len(studies)}")
    print(f"  X shape: {X.shape}")
    print(f"  y: {len(y)}")
    
    # Check if the number of samples aligns after loading
    if not (len(studies) == X.shape[0] == len(y)):
        raise ValueError("Loaded data dimensions do not align.")
    
    return studies, X, y


from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'kappa': make_scorer(cohen_kappa_score),
    'mcc': make_scorer(matthews_corrcoef)
}

def run_inner_cv_scikeras(X, y, study_per_patient, pipeline, param_grid,k = 2, n_jobs = 1, inner_state = 42):
    # ---------------------------------------------------------------------------
    # SET UP CROSS-VALIDATION STRATEGIES
    # ---------------------------------------------------------------------------
    
    inner_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=inner_state)
    outer_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=inner_state)

    # Container for inner cross-validation results.
    inner_predictions = {}

    # ---------------------------------------------------------------------------
    # OUTER CROSS-VALIDATION LOOP
    # ---------------------------------------------------------------------------
    for outer_train_idx, outer_test_idx in outer_cv.split(X, y):
        # Split data into outer training and test sets.
        X_train, y_train = X.iloc[outer_train_idx], y.iloc[outer_train_idx]
        
        # Reindex study metadata to match training indices.
        study_train = study_per_patient.reindex(X_train.index)
        pipeline.set_params(feature_selection__study_per_patient=study_train)
        
        # -----------------------------------------------------------------------
        # INNER CROSS-VALIDATION FOR EACH PARAMETER COMBINATION
        # -----------------------------------------------------------------------
        for params in ParameterGrid(param_grid):
            pipeline.set_params(**params)
            
            #predicted_classes = cross_val_predict(
            #    pipeline, X_train, y_train,
            #    cv=inner_cv, method='predict', n_jobs=n_jobs
            #)
            
            cv_results = cross_validate(
                pipeline, 
                X_train, 
                y_train, 
                cv=inner_cv, 
                scoring=scoring, 
                return_estimator=True, 
                n_jobs=n_jobs
            )
            
            epoch_counts = []

            for estimator in cv_results['estimator']:
                n_epochs = len(estimator._final_estimator.history_["val_loss"])
                epoch_counts.append(n_epochs)

            mean_epochs = np.mean(epoch_counts)
            mean_kappa = np.mean(cv_results['test_kappa'])
            mean_mcc = np.mean(cv_results['test_mcc'])
            mean_accuracy = np.mean(cv_results['test_accuracy'])
            
            # Compile the results for this parameter configuration.
            results_dict = {
                'params': params,
                'indices_inner_fold': outer_train_idx,
                'true_class': y_train,
                'mean_epochs': mean_epochs,
                'kappa': mean_kappa,
                'mcc': mean_mcc,
                'accuracy_score': mean_accuracy,
                'estimators': cv_results['estimator']
            }
            
            key = tuple(sorted(params.items()))
            if key not in inner_predictions:
                inner_predictions[key] = []
            inner_predictions[key].append(results_dict)
    return inner_predictions

def run_inner_cv(X, y, study_per_patient, pipeline, param_grid,k = 2, n_jobs = 1, inner_state = 42):
    # ---------------------------------------------------------------------------
    # SET UP CROSS-VALIDATION STRATEGIES
    # ---------------------------------------------------------------------------
    
    inner_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=inner_state)
    outer_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=inner_state)

    # Container for inner cross-validation results.
    inner_predictions = {}

    # ---------------------------------------------------------------------------
    # OUTER CROSS-VALIDATION LOOP
    # ---------------------------------------------------------------------------
    for outer_train_idx, outer_test_idx in outer_cv.split(X, y):
        # Split data into outer training and test sets.
        X_train, y_train = X.iloc[outer_train_idx], y.iloc[outer_train_idx]
        
        # Reindex study metadata to match training indices.
        study_train = study_per_patient.reindex(X_train.index)
        pipeline.set_params(feature_selection__study_per_patient=study_train)
        
        # -----------------------------------------------------------------------
        # INNER CROSS-VALIDATION FOR EACH PARAMETER COMBINATION
        # -----------------------------------------------------------------------
        for params in ParameterGrid(param_grid):
            pipeline.set_params(**params)
            
            predicted_classes = cross_val_predict(
                pipeline, X_train, y_train,
                cv=inner_cv, method='predict', n_jobs=n_jobs
            )
            
            # Compute inner CV probability predictions.
            #inner_proba_preds = cross_val_predict(
            #    pipeline, X_train, y_train,
            #    cv=inner_cv, method='predict_proba', n_jobs=n_jobs
            #)
            # Create a DataFrame for probability predictions with proper class names.
            #inner_preds_df = pd.DataFrame(inner_proba_preds, columns=class_order, index=X_train.index)
            
            # Determine predicted classes by selecting the class with maximum probability.
            #predicted_classes = inner_preds_df.idxmax(axis=1)
            
            
            # Compile the results for this parameter configuration.
            results_dict = {
                'params': params,
                'indices_inner_fold': outer_train_idx,
                #'inner_preds_proba': inner_preds_df,
                'predicted_class': predicted_classes,
                'true_class': y_train,
                'kappa': cohen_kappa_score(y_train, predicted_classes),
                'mcc': matthews_corrcoef(y_train, predicted_classes),
                'accuracy_score': accuracy_score(y_train, predicted_classes)
            }
            
            key = tuple(sorted(params.items()))
            if key not in inner_predictions:
                inner_predictions[key] = []
            inner_predictions[key].append(results_dict)
    return inner_predictions

def cv_to_extracted_dict(inner_predictions):
    fold_data = []
    for result in inner_predictions:
        proba = result['inner_preds_proba'].values  # shape: (n_samples, n_classes)
        class_labels = result['inner_preds_proba'].columns.values
        true_labels = result['true_class'].values
        raw_preds = result['predicted_class'].values
        
        # Precompute maximum probabilities and indices for each sample.
        max_probs = proba.max(axis=1)
        max_indices = proba.argmax(axis=1)
        
        fold_data.append({
            'max_probs': max_probs,
            'max_indices': max_indices,
            'class_labels': class_labels,
            'true_labels': true_labels,
            'raw_preds': raw_preds  
        })
    return fold_data