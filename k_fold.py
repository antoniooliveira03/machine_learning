from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import time
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import KFold, StratifiedKFold
import joblib

from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np

import play_song as song


def k_fold(df, features, target, k = 5, model = LogisticRegression()):
    
    start_time = time.time()
    
    X = df[features]
    y = df[target]
    
    kf = KFold(n_splits= k, shuffle=True, random_state=1)
    predictions = []
    
    for train_idx, val_idx in kf.split(X):

        ### SPLIT
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        original_columns = X_train.columns
        
        ### PIPELINE
        
        pipeline = Pipeline([
        ('imputer', FunctionTransformer(n.custom_impute, validate=False)), 
        ('log_transform', FunctionTransformer(n.log_transform, validate=False)),  
        ('scaler', RobustScaler()),
        ])
        
        X_train = pipeline.fit_transform(X_train, y_train)
        X_val = pipeline.transform(X_val)
        X_train = pd.DataFrame(X_train, columns=original_columns)
        X_val = pd.DataFrame(X_val, columns=original_columns)

        
        # fit model
        model = model
        model.fit(X_train, y_train)

        # make predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        # compute metrics
        print(classification_report(y_train, train_pred))
        print(classification_report(y_val, val_pred))

        # save predictions and best model's parameters

        predictions.append({'Train Predictions': train_pred, 'Validation Predictions': val_pred})
    
    model_name = type(model).__name__
    print(model_name)
    joblib.dump(model, f'./models/{model_name}.joblib')  

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(elapsed_time)
    song.play_('audio.mp3')
        
    return predictions

## K-Fold 2 

def k_fold(df, features, target, model_name, k = 5, 
           model = LogisticRegression(), patience=2):
    
    start_time = time.time()
    
    X = df[features]
    y = df[target]
    
    kf = StratifiedKFold(n_splits= k, shuffle=True, random_state=1)
    predictions = []
    
    # Initialize variables for early stopping
    best_macro_avg = 0  # Track the best macro average score
    no_improvement_count = 0  # Counter for early stopping

    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        
        print(f'----------FOLD {fold}----------')
        ### SPLIT
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        original_columns = X_train.columns
        
        ### PIPELINE
        
        pipeline = Pipeline([
        ('imputer', FunctionTransformer(n.custom_impute, validate=False)), 
        ('log_transform', FunctionTransformer(n.log_transform, validate=False)),  
        ('scaler', RobustScaler()),
        ])
        
        X_train = pipeline.fit_transform(X_train, y_train)
        X_val = pipeline.transform(X_val)
        X_train = pd.DataFrame(X_train, columns=original_columns)
        X_val = pd.DataFrame(X_val, columns=original_columns)

        
        # fit model
        model = model
        model.fit(X_train, y_train)

        # make predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        # Compute metrics
        train_report = classification_report(y_train, train_pred, output_dict=True)
        val_report = classification_report(y_val, val_pred, output_dict=True)
        
        print(f"Fold {fold} - Training Report:\n", classification_report(y_train, train_pred))
        print(f"Fold {fold} - Validation Report:\n", classification_report(y_val, val_pred))
        
        val_macro_avg = val_report['macro avg']['f1-score']
        
        if val_macro_avg > best_macro_avg:
            best_macro_avg = val_macro_avg
            no_improvement_count = 0
            
            # Save the best model 
            joblib.dump(model, f'./models/{model_name}.joblib')
        else:
            no_improvement_count += 1
            print(f"No improvement for {no_improvement_count} fold(s)")

        if no_improvement_count >= patience:
            print(f"Early stopping at fold {fold} due to no improvement in macro average for {patience} folds")
            break
        
        
        # save predictions and best model's parameters
        predictions.append({'Train Predictions': train_pred, 'Validation Predictions': val_pred})
    
    
    # Time
    end_time = time.time()
    elapsed_time = round((end_time - start_time) / 60, 2)
    print(f'This run took {elapsed_time} minutes')
    
    # Play Warning Song
    song.play_('audio.mp3')
        
    return predictions


#from imblearn.over_sampling import SMOTE
#from imblearn.under_sampling import RandomUnderSampler
#from imblearn.pipeline import Pipeline as ImbPipeline

'''
def k_fold(df, features, target, model_name, 
           param_distributions = None, random_search = False, 
           k = 5, model = LogisticRegression(), patience = 2,
           use_smote = False, undersample = False, 
           smote_strategy = None, under_strategy = None):
    
    
    df: The dataset containing both features and target columns.
    
    features: List of column names to use as features for model training.
    
    target: Column name of the target variable to predict.
    
    model_name: Name under which to save the best-performing model.
    
    param_distributions: Dictionary of hyperparameters for RandomizedSearchCV; used only if random_search is enabled.
    
    random_search: Boolean to enable random search hyperparameter tuning. Default is False.
    
    k: Number of folds for cross-validation. Default is 5.
    
    model: The model to train. Default is LogisticRegression.
    
    patience: Number of consecutive folds without improvement needed to trigger early stopping. Default is 2.
    
    use_smote: Boolean to apply SMOTE to the minority class in the training set. Default is False.
    
    undersample: Boolean to apply undersampling to the majority class in the training set. Default is False.
    
    smote_strategy: Ratio of the minority class after SMOTE. Only used if use_smote is True.
    
    under_strategy: Ratio of the majority class after undersampling. Only used if undersample is True.
    
    
    
    start_time = time.time()
    
    X = df[features]
    y = df[target]
    
    kf = StratifiedKFold(n_splits= k, shuffle=True, random_state=1)
    predictions = []
    
    # Initialize variables for early stopping
    best_macro_avg = 0  
    no_improvement_count = 0

    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        
        print(f'----------FOLD {fold}----------')
        ### SPLIT
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
#        original_columns = X_train.columns
        
        ### PIPELINE
#         steps = []
#         # Add smote or undersampling as needed
#         if use_smote:
#             steps.append(('smote', SMOTE(sampling_strategy=smote_strategy, random_state=1)))
            
#         if undersample:
#             steps.append(('undersampler', RandomUnderSampler(sampling_strategy=under_strategy, random_state=1)))


        ### PIPELINE       
        pipeline = Pipeline([
            ('imputer', FunctionTransformer(n.custom_impute, validate=False)), 
            ('log_transform', FunctionTransformer(n.log_transform, validate=False)),  
            ('scaler', RobustScaler()),
         #   *steps,
            ('classifier', model)
        ])
        

        ### RANDOM SEARCH
        if random_search:
            pass
#             random_search = RandomizedSearchCV(
#                 estimator=pipeline,
#                 param_distributions=param_distributions,
#                 n_iter=50, 
#                 scoring='f1_macro',
#                 cv=3,
#                 verbose=2,
#                 random_state=1,
#                 n_jobs=-1
#             )

#             random_search_cv.fit(X_train, y_train)
#             pipeline = random_search.best_estimator_
        else:
            pipeline.fit(X_train, y_train)

    
        # make predictions
        train_pred = pipeline.predict(X_train)
        val_pred = pipeline.predict(X_val)

        # Compute metrics
        train_report = classification_report(y_train, train_pred, output_dict=True)
        val_report = classification_report(y_val, val_pred, output_dict=True)
        
        print(f"Fold {fold} - Training Report:\n", classification_report(y_train, train_pred))
        print(f"Fold {fold} - Validation Report:\n", classification_report(y_val, val_pred))
        
        val_macro_avg = val_report['macro avg']['f1-score']
        
        if val_macro_avg > best_macro_avg:
            best_macro_avg = val_macro_avg
            no_improvement_count = 0
            
            # Save the best model 
            joblib.dump(pipeline, f'./models/{model_name}.joblib')
        else:
            no_improvement_count += 1
            print(f"No improvement for {no_improvement_count} fold(s)")

        if no_improvement_count >= patience:
            print(f"Early stopping at fold {fold} due to no improvement in macro average for {patience} folds")
            break
        
        
        # save predictions and best model's parameters
        predictions.append({'Train Predictions': train_pred, 'Validation Predictions': val_pred})
    
    
    # Time
    end_time = time.time()
    elapsed_time = round((end_time - start_time) / 60, 2)
    print(f'This run took {elapsed_time} minutes')
    
    # Play Warning Song
    song.play_('audio.mp3')
        
    return predictions
'''