# LASSO
from sklearn.linear_model import Lasso

# RFE
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report

# OTHER
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# LASSO
def lasso_feature_importance(X, y, num, alpha = 0.01, color = 'Blue'):
    X_num = X[num] 
    
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_num, y)
    
    importance = pd.Series(lasso.coef_, index=X_num.columns)
    
    non_zero_importance = importance[importance != 0]
    
    non_zero_importance.sort_values().plot(kind="barh", color=color)
    
    plt.title("Lasso Feature Importance")
    plt.xlabel("Coefficient Value")
    plt.show()
    
    selected_features = non_zero_importance.index
    
    return selected_features

# RFE
def rfe(X, y, num, n_features, model=None):
    
    X_num = X[num]
    
    results = {}
    
    for feature in n_features:
        
        # Perform RFE to select features
        rfe = RFE(estimator=model, n_features_to_select=feature)
        rfe.fit(X_num, y)

        # Get selected features
        selected_features = X_num.columns[rfe.support_]
        
        # Print the number of selected features
        print(f"Trying with {feature} features: {len(selected_features)} selected features.")
        
        # Model predictions and classification report on the training set with selected features
        y_pred = rfe.predict(X_num)
        print(f"Classification Report for {feature} features:")
        print(classification_report(y, y_pred))
        
        # Store the results
        results[feature] = selected_features
        
    return results