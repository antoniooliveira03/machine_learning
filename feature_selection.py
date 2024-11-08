import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def correlation_matrix(X, num, cmap='Blues'):
    X_num = X[num]
    
    # Compute the absolute correlation matrix
    corr_matrix = X_num.corr().abs()

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.show()


from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

def chi_squared(X, y, categ, threshold=0.05):
    X_categ = X[categ]
    
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
    sns.barplot(x='Chi2 Score', y='Feature', data=scores_df.sort_values(by='Chi2 Score', ascending=False))
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'p-value Threshold = {threshold}')
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



from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

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

from sklearn.linear_model import Lasso
def lasso(X, y, alpha = 0.01, color = 'lightblue'):
    
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



from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import numpy as np

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
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, X_comb.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance Using ExtraTreesClassifier')
    # Draw a line at the 5% importance threshold
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'{threshold}% Importance Threshold')
    plt.legend()

    plt.show()

    print(f"\nInitial Features: {len(X_comb.columns)}\n")
    print(X_comb.columns.tolist())
    important_features = X_comb.columns[feature_importance >= threshold]
    print(f"\nFeatures above 5% importance: {len(important_features)}\n")
    print(important_features.to_list())


