from sklearn.linear_model import Lasso

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
