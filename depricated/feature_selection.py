# LASSO
from sklearn.linear_model import Lasso

# RFE
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report

# CHI-SQUARED
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

# MUTUAL INFORMATION 
from sklearn.feature_selection import mutual_info_classif

# OTHER
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# LASSO
def lasso(X, y, num, alpha = 0.01, color = 'Blue'):
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
    
    return selected_features.to_list()

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

# CORRELATION MATRIX

def correlation_matrix(X, num, threshold=0.8, cmap = 'Blues'):
    
    X_num = X[num]
    
    # Compute the absolute correlation matrix
    corr_matrix = X_num.corr().abs()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.show()
    
    # Set to track features to drop
    to_drop = set()
    
    # Iterate over columns to find highly correlated pairs
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            feature1 = corr_matrix.columns[i]
            feature2 = corr_matrix.columns[j]
            
            # Check if the correlation is above the threshold
            if corr_matrix.iloc[i, j] > threshold:
                # Add the second feature to the drop list
                to_drop.add(feature2)
    
    # List of selected features: all features except those in to_drop
    selected_features = [feature for feature in X_num.columns if feature not in to_drop]
    
    return selected_features

# UNIVARIATE TEST
def var(X, num, threshold=0.01):
    X_num = X[num]
    # Calculate variance for each feature
    variances = X_num.var()
    
    # Select features with variance above the threshold
    selected_features = variances[variances > threshold].index
    
    # Plot the variances as horizontal bars
    plt.figure(figsize=(10, 8))
    sns.barplot(x=variances.values, y=variances.index, orient='h')
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold = {threshold}')
    plt.title('Feature Variance')
    plt.xlabel('Variance')
    plt.ylabel('Features')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return selected_features.tolist()

######################## CATEG ########################

# CHI-SQUARED
def chi_squared(X, y, categ, threshold=0.05):
    X_categ = X[categ]
    
    # Scale the features
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_categ), 
                            columns=X_categ.columns)
    
    # Fit the chi-squared selector
    chi2_selector = SelectKBest(chi2, k='all')
    chi2_selector.fit(X_scaled, y)

    # Get Chi-squared scores
    chi2_scores = chi2_selector.scores_

    # Create a DataFrame for scores
    scores_df = pd.DataFrame({
        'Feature': X_categ.columns,
        'Chi2 Score': chi2_scores
    })

    # Filter features based on the threshold
    selected_features = scores_df[scores_df['Chi2 Score'] > threshold]['Feature']
    
    # Plot the Chi-squared scores
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Chi2 Score', y='Feature', data=scores_df)
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold = {threshold}')
    plt.title('Chi-squared Scores for Features')
    plt.xlabel('Chi-squared Score')
    plt.ylabel('Features')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return selected_features.tolist()

# MUTUAL INFORMATION 
def mutual_information(X, y, categ, threshold=0.01):
    X_categ = X[categ]
    
    # Calculate Mutual Information
    mi_scores = mutual_info_classif(X_categ, y, discrete_features='auto', random_state=42)
    
    # Create a DataFrame for better visualization
    mi_scores_df = pd.DataFrame(mi_scores, index=X_categ.columns, columns=["Mutual Information"])
    mi_scores_df = mi_scores_df.sort_values(by="Mutual Information", ascending=False)

    # Filter features based on the threshold
    selected_features = mi_scores_df[mi_scores_df["Mutual Information"] >= threshold].index.tolist()
    
    # Plot the Mutual Information scores
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Mutual Information', y=mi_scores_df.index, data=mi_scores_df)
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold = {threshold}')
    plt.title('Mutual Information Scores for Features')
    plt.xlabel('Mutual Information Score')
    plt.ylabel('Features')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return selected_features
