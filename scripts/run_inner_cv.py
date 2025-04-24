import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import train_test, transformers, classifiers

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import datetime
import pandas as pd
import argparse

def main():
    # Parsing
    parser = argparse.ArgumentParser(description="Run XGBOOST or SVM cross-validation with adjustable n_jobs.")

    # Define argument configurations
    arg_configs = {
        'model_type': {
            'type': str,
            'help': 'Model type to use'
        },
        'n_jobs': {
            'type': int,
            'default': 12,
            'help': 'Number of jobs to run in parallel for cross-validation (default: 12)'
        },
        'k_out': {
            'type': int,
            'default': 5,
            'help': 'Number of inner folds (default: 5)'
        },
        'k_in': {
            'type': int,
            'default': 5,
            'help': 'Number of outer folds (default: 5)'
        }
    }

    # Add arguments from configurations
    for arg_name, config in arg_configs.items():
        parser.add_argument(
            f'--{arg_name}',
            **config
        )

    args = parser.parse_args() # Parse the command-line arguments
    n_jobs = args.n_jobs # Get the value of n_jobs from the parsed arguments

    # Get the current date and time in string format
    time = datetime.datetime.now()
    time = time.strftime("%Y%m%d_%H%M%S")

    # Create the output directory if it doesn't exist
    output_dir = f"out/{args.model_type}"
    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare data
    print("Loading and preparing data")
    X, y, study_labels = train_test.load_data("data")
    X, y, study_labels = train_test.filter_data(X, y, study_labels, min_n = 20)
    y, label_mapping = train_test.encode_labels(y)

    # Define the model and parameter grid   
    if args.model_type == "XGBOOST":
        model = classifiers.WeightedXGBClassifier
        param_grid = {
            'n_genes': [1000, 2000, 3000],
            'class_weight': [True, False],
            'max_depth': [2, 3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1.0],
            'reg_lambda': [0, 1.0, 10.0]
        }
        param_grid = {
            'n_genes': [1000],
            'class_weight': [True, False]
        }
    elif args.model_type == "SVM":
        from sklearn.svm import SVC
        model = SVC()
        param_grid = {
            'n_genes': [1000, 2000, 3000],
            'C': [0.1, 1, 10, 100, 1000],  
            'gamma': ['auto', 'scale', 0.0001, 0.001, 0.01, 0.1],  
            'class_weight': ["balanced", None],
            'probability': [True]
        }
    else:
        raise ValueError(f"Model type {args.model_type} not supported")

    # Define the pipeline
    pipe = Pipeline([
        ('DEseq2', transformers.DESeq2RatioNormalizer()),
        ('feature_selection', transformers.FeatureSelection()),
        ('scaler', StandardScaler())
    ])

    # Get the number of inner and outer folds
    k_out = args.k_out
    k_in = args.k_in

    # Start the inner cross-validation process
    print("Starting inner cross-validation process.")
    # Iterate through different multiclass classification strategies
    # standard: Uses the classifier's default multiclass handling
    # OvO: One-vs-One strategy - trains binary classifier between each pair of classes
    # OvR: One-vs-Rest strategy - trains binary classifier for each class against all others
    for multi_type in ["standard", "OvO", "OvR"]:
        print(f"Running {multi_type} strategy...")
        # Run inner cross-validation with current multiclass strategy
        df = train_test.run_inner_cv(X, y, study_labels, model, param_grid, n_jobs, pipe, 
                                   multi_type=multi_type, k_out=k_out, k_in=k_in)
        
        # Convert encoded labels back to original class names
        df = train_test.restore_labels(df, label_mapping)
        
        # Save results to CSV file with model type, strategy and timestamp
        df.to_csv(f"{output_dir}/{args.model_type}_inner_cv_{multi_type}_{time}.csv")   
        print(f"Finished {multi_type} strategy.")
    
    print("Cross-validation process finished.")

if __name__ == "__main__":
    main()