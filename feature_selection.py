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
    
    print("\nInitial Features:\n")
    print(X_categ.columns.tolist())
    print("\nFinal Decision for Categorical Features:\n")
    print(selected_features.tolist())


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

def rfe(X, y, num, n_features, model=None):
    
    X_num = X[num]
    best_score = 0
    best_features = []

    results = {}
    
    for feature in n_features:
        
        # Perform RFE to select features
        rfe = RFE(estimator=model, n_features_to_select=feature)
        rfe.fit(X_num, y)

        # Get selected features
        selected_features = X_num.columns[rfe.support_]
        
        # Model predictions and classification report on the training set with selected features
        y_pred = rfe.predict(X_num)
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

def lasso(X, y, num, alpha = 0.01, color = 'lightblue'):
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
    print(selected_features.to_list())