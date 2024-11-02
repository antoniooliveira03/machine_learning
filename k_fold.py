from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import time
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import KFold
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