import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

def correlation_matrix(X, cmap='YlOrBr'):
    
    # Compute the absolute correlation matrix
    corr_matrix = X.corr().abs()

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.show()

def chi_squared(X_categ, y, threshold=0.05):
    
    # Scale the features
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_categ), 
                            columns=X_categ.columns)
    
    # Fit the chi-squared selector
    chi2_selector = SelectKBest(chi2, k='all')
    chi2_selector.fit(X_scaled, y)

    # Get Chi-squared scores and p-values
    chi2_scores = chi2_selector.scores_
    p_values = chi2_selector.pvalues_

    # Create a DataFrame for scores and p-values
    scores_df = pd.DataFrame({
        'Feature': X_categ.columns,
        'Chi2 Score': chi2_scores,
        'p-value': p_values
    })

    # Filter features based on the p-value threshold
    selected_features = scores_df[scores_df['p-value'] < threshold]['Feature']
    
    # Extract non-selected features
    non_selected_features = scores_df[scores_df['p-value'] >= threshold]

    # Plot the Chi-squared scores
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Chi2 Score', y='Feature', data=scores_df.sort_values(by='Chi2 Score', ascending=False), color='orange')
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'p-value Threshold = {threshold}')
    plt.title('Chi-squared Scores for Features')
    plt.xlabel('Chi-squared Score')
    plt.ylabel('Features')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print(f"\nInitial Features: {len(X_categ.columns.tolist())}\n")
    print(X_categ.columns.tolist())
    print(f"\nDecision for Categorical Features (p-value < threshold): {len(selected_features.tolist())}\n")
    print(selected_features.tolist())

    # Display non-selected features with their p-values and Chi-squared scores
    print("\nNon-Selected Features (p-value >= threshold):\n")
    print(non_selected_features[['Feature', 'Chi2 Score', 'p-value']])

def mutual_info(X, y, threshold=0.1):
    
    # Calculate MI scores
    mi_scores = mutual_info_classif(X, y, random_state=0)
    mi_scores_series = pd.Series(mi_scores, index=X.columns)
    
    # Select features based on the threshold
    selected_features = mi_scores_series[mi_scores_series >= threshold].index.tolist()
    
    # Plot the MI scores for visualization
    plt.figure(figsize=(10, 6))
    mi_scores_series.sort_values(ascending=True).plot(kind='barh', color='orange')
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
    plt.xlabel("Mutual Information Score")
    plt.ylabel("Features")
    plt.title("Mutual Information Scores for Features")
    plt.legend()
    plt.show()
    
    print(f"\nInitial Features: {len(X.columns.tolist())} \n")
    print(X.columns.tolist())
    print(f"\nDecision for Categorical Features (MI Score >= {threshold}): {len(selected_features)} \n")
    print(selected_features)

def rfe(X, y, n_features, model=None):
    
    best_score = 0
    best_features = []

    results = {}
    
    for feature in n_features:
        
        # Perform RFE to select features
        rfe = RFE(estimator=model, n_features_to_select=feature)
        rfe.fit(X, y)

        # Get selected features
        selected_features = X.columns[rfe.support_]
        
        # Model predictions and classification report on the training set with selected features
        y_pred = rfe.predict(X)
        print(f"Classification Report for {feature} features:\n")
        print(classification_report(y, y_pred))
        
        # Calculate the macro average F1 score
        macro_f1 = f1_score(y, y_pred, average='macro')
        print(f"Macro Avg F1 Score for {feature} features: {macro_f1:.4f}\n")
        
        # Store the results
        results[feature] = selected_features
        
        # Check if this is the best score
        if macro_f1 > best_score:
            best_score = macro_f1
            best_features = selected_features.tolist()  
    
    return best_features


def lasso(X, y, alpha = 0.01, color = 'orange'):
    
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    
    importance = pd.Series(lasso.coef_, index=X.columns)
    
    non_zero_importance = importance[importance != 0]
    
    importance.sort_values().plot(kind="barh", color=color)
    
    plt.title("Lasso Feature Importance")
    plt.xlabel("Coefficient Value")
    plt.show()
    
    selected_features = non_zero_importance.index
    
    print(f"\nInitial Features: {len(X.columns)}\n")
    print(X.columns.tolist())
    print(f"\nDecision for Numerical Features (lasso ≠ 0): {len(selected_features.tolist())}\n")
    print(selected_features.tolist())

def plot_feature_importance(X_num, X_categ, y, n_estimators=250, random_state=42,
                            threshold=5):
    # Concatenate scaled and categorical features
    X_comb = pd.concat([X_num, X_categ], axis=1)

    # Initialize the ExtraTreesClassifier with given parameters
    clf = ExtraTreesClassifier(n_estimators=n_estimators,
                               random_state=random_state)

    # Fit the model on the training data
    clf.fit(X_comb, y)

    # Calculate feature importances
    feature_importance = clf.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())

    # Sort the indices of features based on importance
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.barh(pos, feature_importance[sorted_idx], align='center', color='orange')
    plt.yticks(pos, X_comb.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance Using ExtraTreesClassifier')
    # Draw a line at the 5% importance threshold
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'{threshold}% Importance Threshold')
    plt.legend()

    plt.show()

    print(f"\nInitial Features: {len(X_comb.columns)}\n")
    print(X_comb.columns.tolist())

    # Identify important features
    important_features = X_comb.columns[feature_importance >= threshold]

    # Split important features into numerical and categorical
    important_num_features = [f for f in important_features if f in X_num.columns]
    important_categ_features = [f for f in important_features if f in X_categ.columns]

    print("\nImportant Numerical Features:")
    print(important_num_features)

    print("\nImportant Categorical Features:")
    print(important_categ_features)
