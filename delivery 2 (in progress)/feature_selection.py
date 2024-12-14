import pandas as pd
import numpy as np

# Plots
import matplotlib.pyplot as plt
import seaborn as sns

# Feature Selection
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE

# Scaler
from sklearn.preprocessing import MinMaxScaler

# Model
from sklearn.linear_model import LogisticRegression

# Metrics
from sklearn.metrics import classification_report, f1_score

def correlation_matrix(X, cmap='YlOrBr'):
    
    # Correlation matrix
    corr_matrix = X.corr().abs()

    # Plot Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.show()

def chi_squared(X_categ, y, threshold=0.05):
    
    # Scale 
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_categ), 
                            columns=X_categ.columns)
    
    # Fit 
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

    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Chi2 Score', y='Feature', data=scores_df.sort_values(by='Chi2 Score', ascending=False), color='orange')
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'p-value Threshold = {threshold}')
    plt.title('Chi-squared Scores for Features')
    plt.xlabel('Chi-squared Score')
    plt.ylabel('Features')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Print Results
    print(f"\nInitial Features: {len(X_categ.columns.tolist())}\n")
    print(X_categ.columns.tolist())
    print(f"\nDecision for Categorical Features (p-value < threshold): {len(selected_features.tolist())}\n")
    print(selected_features.tolist())
    print("\nNon-Selected Features (p-value >= threshold):\n")
    print(non_selected_features[['Feature', 'Chi2 Score', 'p-value']])

def mutual_info(X, y, threshold=0.1):
    
    # MI scores
    mi_scores = mutual_info_classif(X, y, random_state=0)
    mi_scores_series = pd.Series(mi_scores, index=X.columns)
    
    # Select features based on the threshold
    selected_features = mi_scores_series[mi_scores_series >= threshold].index.tolist()
    
    # Plot
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


def rfe(X_train, y_train, X_val, y_val, n_features, model=None):
    
    best_score = 0
    best_features = []

    results = {}
    
    for feature in n_features:
        
        # Fit
        rfe = RFE(estimator=model, n_features_to_select=feature)
        rfe.fit(X_train, y_train)

        # Get selected features
        selected_features = X_train.columns[rfe.support_]
        
        print('-------------TRAIN-------------')

        # Predictions for Train
        y_pred = rfe.predict(X_train)
        print(f"Classification Report for {feature} features:\n")
        print(classification_report(y_train, y_pred))
        
        # Metrics
        macro_f1 = f1_score(y_train, y_pred, average='macro')
        print(f"Macro Avg F1 Score for {feature} features: {macro_f1:.4f}\n")
        
        print('----------VALIDATION----------')

        # Predictions for Validation
        y_val_pred = rfe.predict(X_val)
        print(f"Classification Report for {feature} features:\n")
        print(classification_report(y_val, y_val_pred))
        
        # Metrics
        macro_f1_val = f1_score(y_val, y_val_pred, average='macro')
        print(f"Macro Avg F1 Score for {feature} features: {macro_f1_val:.4f}\n")
        
        # Best score
        if macro_f1_val > best_score:
            best_score = macro_f1_val
            best_features = selected_features.tolist()  
    
    return best_features


def lasso(X, y, alpha = 0.01, color = 'orange'):
    
    # Fit
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    
    # Get Feature Importance
    importance = pd.Series(lasso.coef_, index=X.columns)
    importance.sort_values().plot(kind="barh", color=color)
    non_zero_importance = importance[importance != 0]
    selected_features = non_zero_importance.index

    # Plot
    plt.title("Lasso Feature Importance")
    plt.xlabel("Coefficient Value")
    plt.show()
    
    # Print Results
    print(f"\nInitial Features: {len(X.columns)}\n")
    print(X.columns.tolist())
    print(f"\nDecision for Numerical Features (lasso ≠ 0): {len(selected_features.tolist())}\n")
    print(selected_features.tolist())

def plot_feature_importance(X_num, X_categ, y, n_estimators=250, random_state=42,
                            threshold=5):
    
    # Combine Numeric and Categroical
    X_comb = pd.concat([X_num, X_categ], axis=1)

    # Fit
    clf = ExtraTreesClassifier(n_estimators=n_estimators,
                               random_state=random_state)
    clf.fit(X_comb, y)

    # Feature Importances
    feature_importance = clf.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())

    # Sort based on importance
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5

    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(pos, feature_importance[sorted_idx], align='center', color = 'orange')
    plt.yticks(pos, X_comb.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance Using ExtraTreesClassifier')

    # Draw a line at the defined threshold
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'{threshold}% Importance Threshold')
    plt.legend()
    plt.show()

    # Print Results 
    print(f"\nInitial Features: {len(X_comb.columns)}\n")
    print(X_comb.columns.tolist())

    # Split Numerical and Categorical for easier interpretation
    important_features = X_comb.columns[feature_importance >= threshold]
    important_num_features = [f for f in important_features if f in X_num.columns]
    important_categ_features = [f for f in important_features if f in X_categ.columns]

    print("\nImportant Numerical Features:")
    print(important_num_features)
    print("\nImportant Categorical Features:")
    print(important_categ_features)
